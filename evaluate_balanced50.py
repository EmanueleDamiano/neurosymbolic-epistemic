from pathlib import Path
import os
os.chdir(Path(__file__).resolve().parent)
import argparse
import matplotlib
matplotlib.use('Agg')
parser = argparse.ArgumentParser(description="Evaluate saved answers without LLM calls.")
parser.add_argument('--input', type=Path, default=Path('results_qwen_3_5_4b_subclass_balanced50_axioms_ttl.jsonl'))
parser.add_argument('--output-dir', type=Path, default=Path('generated/evaluation'))
parser.add_argument('--model-label', default='qwen3.5:4b')
args = parser.parse_args()
args.output_dir.mkdir(parents=True, exist_ok=True)
_model = args.model_label
## valutazione: risposta finale + confronto assiomi citati vs assiomi groundtruth
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
from subclass_justification import load_graph, build_named_subclass_edges
from axiom_match import parse_axioms_block, canonical_set, canonical_axiom

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)

output_file = str(args.input)
with open(output_file) as f:
    bal_results = [json.loads(line) for line in f if line.strip()]
print("risposte raccolte:", len(bal_results))

# ------------------------------------------------------------------
# NOTA SUL CONFRONTO
# Il template gold di subclass_justification usa un placeholder ('...') al posto delle
# restrizioni anonime:   :Employee owl:equivalentClass [intersectionOf ( :Person ... )]
# mentre il modello di norma le espande per intero. Le due scritture denotano lo STESSO
# assioma, quindi il confronto per stringa esatta misurerebbe la formattazione e non il
# contenuto. Si confrontano quindi le forme CANONICHE (axiom_match.canonical_axiom);
# la metrica per stringa esatta resta calcolata come riferimento.
# ------------------------------------------------------------------
g = load_graph("univ-bench-inst.ttl")
edges = build_named_subclass_edges(g)

ontology_canon = {
    canonical_axiom(e["axiom"])
    for node, lst in edges.items()
    if not node.endswith("__restrictions__")
    for e in lst
}
ontology_canon.discard(None)
print("assiomi (archi verso classi nominate) nell'ontologia:", len(ontology_canon))

norm_str = lambda a: " ".join(str(a).split()).lower()

# ------------------------------------------------------------------
# 1. accuratezza della final answer
# ------------------------------------------------------------------
_gt = [str(r["answer_gt"]).strip().lower() for r in bal_results]
_pred = [str(r["final_answer"]).strip().lower() for r in bal_results]
print("\naccuracy sul campione bilanciato:", accuracy_score(_gt, _pred))

labels = sorted(set(_gt) | set(_pred))
cm = confusion_matrix(_gt, _pred, labels=labels)
plt.figure(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
plt.xlabel("Predicted label"); plt.ylabel("True label")
plt.title(f"{_model} - subclassOf campione bilanciato (ttl)")
plt.tight_layout(); plt.savefig(args.output_dir / 'confusion_matrix.png'); plt.close()

# ------------------------------------------------------------------
# 2. metriche sugli assiomi citati
# ------------------------------------------------------------------
rows = []
for r in bal_results:
    cited_lines = parse_axioms_block(r["axioms_raw"])          # re-parse bracket-aware
    cited, unparsed, declarations = canonical_set(cited_lines)  # declaration = rumore
    gold, _, _ = canonical_set(r["gt_axioms"])

    tp = len(cited & gold)
    precision = tp / len(cited) if cited else (1.0 if not gold else 0.0)
    recall = tp / len(gold) if gold else None                   # indefinito sui negativi
    f1 = (2 * precision * recall / (precision + recall)
          if recall not in (None, 0) and precision else (None if recall is None else 0.0))

    hallucinated = [a for a in cited if a not in ontology_canon]

    # metrica per stringa esatta, solo come riferimento
    cited_str = {norm_str(a) for a in (cited_lines or [])}
    gold_str = {norm_str(a) for a in (r["gt_axioms"] or [])}
    tp_str = len(cited_str & gold_str)
    precision_str = tp_str / len(cited_str) if cited_str else (1.0 if not gold_str else 0.0)

    rows.append({
        "sample_id": r["sample_id"],
        "question": r["question"],
        "reasoning_type": r["reasoning_type"],
        "answer_gt": r["answer_gt"],
        "final_answer": r["final_answer"],
        "answer_ok": str(r["answer_gt"]).lower() == str(r["final_answer"]).lower(),
        "n_cited": len(cited),
        "n_gold": len(gold),
        "axiom_precision": precision,
        "axiom_recall": recall,
        "axiom_f1": f1,
        "exact_match": cited == gold,
        "n_hallucinated": len(hallucinated),
        "groundedness": (1 - len(hallucinated) / len(cited)) if cited else None,
        "n_declarations": len(declarations),   # ':X rdf:type owl:Class' -> rumore, non giustifica
        "n_unparsed": len(unparsed),
        "axiom_precision_stringa_esatta": precision_str,
        "hallucinated_axioms": "; ".join(str(a) for a in hallucinated),
    })

axiom_eval_df = pd.DataFrame(rows)

print("\n--- formato ---")
print("risposte con blocco <axioms> presente:",
      f"{sum(r['axioms_raw'] is not None for r in bal_results)}/{len(bal_results)}")
print("righe non interpretabili come assioma:", int(axiom_eval_df["n_unparsed"].sum()))

print("\n--- per reasoning_type (matching semantico) ---")
print(axiom_eval_df.groupby("reasoning_type").agg(
    n=("sample_id", "size"),
    answer_acc=("answer_ok", "mean"),
    axiom_precision=("axiom_precision", "mean"),
    axiom_recall=("axiom_recall", "mean"),
    exact_match=("exact_match", "mean"),
    groundedness=("groundedness", "mean"),
    mean_cited=("n_cited", "mean"),
    mean_gold=("n_gold", "mean"),
).round(3))

# sui subClassOf_negative il golden set e' VUOTO per costruzione (nessun assioma giustifica
# la subsumption): il recall e' indefinito e la precision misura la sovra-citazione.
pos = axiom_eval_df[axiom_eval_df["reasoning_type"] != "subClassOf_negative"]
print("\n--- solo casi positivi (direct + inferred) ---")
print("axiom precision (semantico):", round(pos["axiom_precision"].mean(), 3))
print("axiom recall    (semantico):", round(pos["axiom_recall"].mean(), 3))
print("axiom F1        (semantico):", round(pos["axiom_f1"].mean(), 3))
print("exact match     (semantico):", round(pos["exact_match"].mean(), 3))
print("[riferimento] precision per stringa esatta:",
      round(pos["axiom_precision_stringa_esatta"].mean(), 3))

neg = axiom_eval_df[axiom_eval_df["reasoning_type"] == "subClassOf_negative"]
print("\n--- solo casi negativi (golden set vuoto) ---")
print("casi in cui il modello non cita nulla:", int((neg["n_cited"] == 0).sum()), "/", len(neg))
print("media assiomi citati (sovra-citazione):", round(neg["n_cited"].mean(), 2))

print("\n--- assiomi allucinati (citati ma non presenti nell'ontologia) ---")
hal = axiom_eval_df[axiom_eval_df["n_hallucinated"] > 0]
print(hal[["sample_id", "question", "reasoning_type", "hallucinated_axioms"]].to_string(index=False)
      if len(hal) else "nessuno")

axiom_eval_df.to_csv(args.output_dir / 'axiom_eval.csv', index=False)
axiom_eval_df.head(10)
