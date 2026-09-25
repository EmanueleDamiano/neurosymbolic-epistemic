from pathlib import Path
import os
os.chdir(Path(__file__).resolve().parent)
import argparse
parser = argparse.ArgumentParser(description="Run the fixed 50-query axiom experiment; resumes an existing run.")
parser.add_argument('--model', default='qwen3.5:4b')
parser.add_argument('--output', type=Path, default=Path('generated/balanced50.jsonl'))
args = parser.parse_args()
if args.output.resolve() == Path('results_qwen_3_5_4b_subclass_balanced50_axioms_ttl.jsonl').resolve():
    parser.error('Choose a new output; the reference run is immutable.')
## nuovo system prompt: il modello deve riportare esplicitamente la LISTA degli assiomi usati
## (formato a una riga per assioma, cosi' da poterli estrarre e confrontare con gli assiomi groundtruth
##  prodotti da subclass_justification.build_subclass_justification)

import re
import os
import json
import ollama

## onto context: stessa ontologia ttl del run precedente
with open("univ-bench-inst.txt") as f:
    lubm_onto = f.read()

thinking_modalities = {"off": "detailed thinking off", "on": "detailed thinking on"}
selected_modality = "on"

system_prompt_subclassOf_report_axioms = f"""You are an expert in semantic web with deep understanding of Ontology Web Language (OWL). {thinking_modalities[selected_modality]}

Your task is to analyze the provided OWL ontology and evaluate a natural language subsumption query (TBox reasoning) against it.

CRITICAL OWL SEMANTICS RULES:
1. If the query logically follows from the ontology axioms the final answer is "True".
2. If the query does not follow from the ontology axioms the final answer is "False".
   Class subsumption over the TBox is decidable: the absence of a derivation is a sound
   non-entailment result, not an "unknown" caused by the Open World Assumption.

OUTPUT FORMAT:
You must strictly return your final response inside the following XML tags. Do not add any text or conversational filler outside these tags.

<reasoning_trace>
[Write here your step-by-step logical deduction, referencing the specific OWL classes, properties, or axioms used]
</reasoning_trace>
<axioms>
[Write here the list of the ontology axioms you used, one per line, following the rules below]
</axioms>
<final_answer>
[Strictly write only ONE of these: True, False]
</final_answer>

RULES FOR THE <axioms> SECTION:
- Write ONE axiom per line and nothing else on that line: no numbering, no bullet, no comment, no prose.
- Every line must use EXACTLY one of these two templates:
    :ClassA rdfs:subClassOf :ClassB
    :ClassA owl:equivalentClass [intersectionOf ( :ClassB ... )]
  Use the first one when the ontology asserts an explicit rdfs:subClassOf between two NAMED classes.
  Use the second one when :ClassA is defined by an owl:equivalentClass whose intersection contains the named class :ClassB.
- Write class names as the local name prefixed by ':' exactly as spelled in the ontology (e.g. :FullProfessor). Never write a full IRI and never write a blank node identifier.
- List ONLY the axioms that are strictly necessary to derive the answer (the minimal justification), in the order in which they are used in the derivation chain. Do not add axioms that are true but not needed.
- Do NOT write inference steps here (transitivity, "therefore X is a subclass of Y", ...): those belong to <reasoning_trace>. This section must contain only axioms asserted in the ontology.
- Never invent an axiom: every line must be literally verifiable in the ONTOLOGY CONTEXT.
- If no axiom supports the query, write exactly: NONE
"""


import time
from subclass_justification import (load_graph, build_named_subclass_edges,
                                    build_subclass_justification)

_model = args.model                       # stesso modello del run precedente
sample_file = "subclass_balanced_sample_50.json"
output_file = str(args.output)
args.output.parent.mkdir(parents=True, exist_ok=True)

reasoning_pattern = r"<reasoning_trace>(.*?)</reasoning_trace>"
axioms_pattern = r"<axioms>(.*?)</axioms>"
answer_pattern = r"<final_answer>(.*?)</final_answer>"


# il parser e' bracket-aware: unisce le righe quando il modello va a capo dentro
# una restrizione anonima spezzando un singolo assioma su piu' righe
from axiom_match import parse_axioms_block


def generate_ollama_ans(model, system_prompt, prompt, **kwargs):
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    return ollama.chat(model=model, messages=messages, options=kwargs)


# rileggiamo il campione dal file salvato: cosi' il run usa esattamente
# il campione persistito in locale (riproducibile anche a kernel riavviato)
with open(sample_file) as f:
    balanced_sample = json.load(f)

g = load_graph("univ-bench-inst.ttl")
edges = build_named_subclass_edges(g)

processed_questions = set()
if os.path.exists(output_file):
    with open(output_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                processed_questions.add(json.loads(line)["question"])
            except Exception:
                continue
print(f"Trovate {len(processed_questions)} query gia' elaborate.", flush=True)

with open(output_file, "a", encoding="utf-8") as f:
    for i, row in enumerate(balanced_sample):
        user_query = row["question"]
        if user_query in processed_questions:
            print(f"[{i}] skipped", flush=True)
            continue

        answer_gt = row["answer"]
        gold = build_subclass_justification(g, edges, row["sub"], row["sup"])

        user_prompt = f"""ONTOLOGY CONTEXT:
{lubm_onto}

USER QUERY:
{user_query}

Analyze the context and query, then output the requested XML structure.
"""
        t0 = time.time()
        try:
            resp = generate_ollama_ans(
                model=_model,
                system_prompt=system_prompt_subclassOf_report_axioms,
                prompt=user_prompt,
                temperature=0.6,
            )
            content = resp["message"]["content"]
            thinking = getattr(resp["message"], "thinking", None)

            reasoning_match = re.search(reasoning_pattern, content, re.DOTALL)
            axioms_match = re.search(axioms_pattern, content, re.DOTALL)
            answer_match = re.search(answer_pattern, content, re.DOTALL)

            axioms_raw = axioms_match.group(1).strip() if axioms_match else None

            record = {
                "sample_id": i,
                "orig_index": row["orig_index"],
                "question": user_query,
                "sub": row["sub"],
                "sup": row["sup"],
                "reasoning_type": row["reasoning_type"],
                "answer_gt": answer_gt,
                "reasoning_trace": reasoning_match.group(1).strip() if reasoning_match else None,
                "axioms_raw": axioms_raw,
                "axioms": parse_axioms_block(axioms_raw),
                "final_answer": answer_match.group(1).strip() if answer_match else None,
                "gt_axioms": gold["axioms"],
                "gt_steps": gold["steps"],
                "thinking": thinking,
                "raw_response": content,
                "elapsed_s": round(time.time() - t0, 2),
            }
        except Exception as e:
            record = {
                "sample_id": i,
                "orig_index": row["orig_index"],
                "question": user_query,
                "sub": row["sub"],
                "sup": row["sup"],
                "reasoning_type": row["reasoning_type"],
                "answer_gt": answer_gt,
                "reasoning_trace": None,
                "axioms_raw": None,
                "axioms": None,
                "final_answer": None,
                "gt_axioms": gold["axioms"],
                "gt_steps": gold["steps"],
                "thinking": None,
                "raw_response": None,
                "elapsed_s": round(time.time() - t0, 2),
                "error": str(e),
            }

        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()
        processed_questions.add(user_query)
        print(f"[{i}] processed ({record['elapsed_s']}s) -> {record['final_answer']}", flush=True)
