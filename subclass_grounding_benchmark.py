"""Frozen positive subclass queries, full OWL context, verified support and withdrawal."""
from __future__ import annotations

import argparse
from collections import Counter, deque
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import platform
import random
import re
import shutil
import subprocess
import time
import urllib.error
import urllib.request
import uuid

import jsonschema
import owlready2
import rdflib
from rdflib import RDF, RDFS, OWL, Namespace, URIRef
from rdflib.compare import isomorphic, to_isomorphic

ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "benchmarks" / "subclass_grounding_v1"
NS = Namespace("http://swat.cse.lehigh.edu/onto/univ-bench.owl#")
OPTIONS = {"temperature": 0, "seed": 42, "num_ctx": 32768, "num_predict": 768}
SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string", "enum": ["entailed", "not_entailed"]},
        "axioms": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "subject": {"type": "string", "pattern": "^https?://[A-Za-z0-9_:/#.-]+$"},
                "predicate": {"type": "string", "const": str(RDFS.subClassOf)},
                "object": {"type": "string", "pattern": "^https?://[A-Za-z0-9_:/#.-]+$"},
            },
            "required": ["subject", "predicate", "object"], "additionalProperties": False,
        }},
    },
    "required": ["answer", "axioms"], "additionalProperties": False,
}
INSTRUCTIONS = """Evaluate the subclass query against the CURRENT ONTOLOGY using OWL semantics.
The complete ontology is provided in Turtle. Do not use outside domain knowledge.
entailed means that the query follows from the current ontology.
not_entailed means that it does not follow, not that its negation is true.
A new current ontology replaces the previous ontology completely. Axioms absent from the new ontology no longer apply.
For entailed, return a sufficient set of asserted named-class rdfs:subClassOf axioms supporting the query.
For this task, such a set exists whenever the query is entailed. Alternative valid proofs are acceptable.
Use full absolute URIs for subject, predicate and object. Do not report inferred conclusions as asserted axioms.
For not_entailed, return an empty axioms array.
Return only the answer and axioms as JSON matching this schema:
""" + json.dumps(SCHEMA, sort_keys=True)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def json_bytes(value) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8")


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def triple_json(triple):
    return dict(zip(("subject", "predicate", "object"), map(str, triple)))


def json_triple(value):
    return tuple(URIRef(value[k]) for k in ("subject", "predicate", "object"))


def copy_graph(graph):
    other = rdflib.Graph()
    for prefix, namespace in graph.namespaces():
        other.bind(prefix, namespace)
    for triple in graph:
        other.add(triple)
    return other


class Oracle:
    """HermiT classification cache, isolated World for every distinct graph."""
    def __init__(self):
        java = shutil.which("java")
        if not java:
            raise RuntimeError("Java required by HermiT; put the JDK on PATH.")
        owlready2.JAVA_EXE = java
        self.cache = {}

    def entails(self, graph, query):
        if any(graph.triples((None, OWL.imports, None))):
            raise ValueError("External OWL imports are not allowed in this frozen benchmark.")
        key = to_isomorphic(graph).graph_digest()
        if key not in self.cache:
            world = owlready2.World()
            try:
                world.get_ontology("http://swat.cse.lehigh.edu/onto/univ-bench.owl").load(
                    fileobj=io.BytesIO(graph.serialize(format="xml", encoding="utf-8")))
                # Inconsistent ontologies raise; never turn reasoner errors into negatives.
                owlready2.sync_reasoner(world, debug=0)
                self.cache[key] = {
                    c.iri: {a.iri for a in c.ancestors() if hasattr(a, "iri")}
                    for c in world.classes()
                }
            finally:
                world.close()
        closure = self.cache[key]
        sub, sup = map(str, query)
        if sub not in closure or sup not in closure:
            raise ValueError("Query class absent from classified graph.")
        if str(OWL.Nothing) in closure[sub]:
            raise ValueError("Unsatisfiable query subject: vacuous entailment excluded.")
        return sup in closure[sub]


def named_path(graph, query):
    """Candidate proof only; full OWL entailment is always checked with HermiT."""
    start, end = map(URIRef, query)
    queue, seen = deque([(start, [])]), {start}
    while queue:
        current, path = queue.popleft()
        if current == end:
            return path
        for parent in sorted(graph.objects(current, RDFS.subClassOf), key=str):
            if isinstance(parent, URIRef) and parent not in seen:
                seen.add(parent)
                queue.append((parent, path + [(current, RDFS.subClassOf, parent)]))
    return None


def query_count(value):
    """Accept a positive integer or 'all'; never silently cap the requested size."""
    if value == "all":
        return "all"
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise ValueError("n_queries must be a positive integer or 'all'.")
    try:
        count = int(value)
    except ValueError as exc:
        raise ValueError("n_queries must be a positive integer or 'all'.") from exc
    if count < 1:
        raise ValueError("n_queries must be positive.")
    return count


def select_verified_cases(eligible, n_queries="all", seed=20260929):
    n_queries = query_count(n_queries)
    if not eligible:
        raise ValueError("No eligible queries passed the reasoner checks.")
    if n_queries != "all" and n_queries > len(eligible):
        raise ValueError(f"Requested {n_queries} queries, but only {len(eligible)} are eligible. Use 'all' or a smaller number.")
    selected = sorted(eligible, key=lambda item: item[0]["query"])
    random.Random(seed).shuffle(selected)
    if n_queries != "all":
        selected = selected[:n_queries]
    # Sampling is uniform without replacement; display order is by proof depth.
    return sorted(selected, key=lambda item: (item[0]["explicit_proof_depth"], item[0]["query"]))


def build_dataset(destination=DATASET, seed=20260929, n_queries="all"):
    """Verify the full positive pool, then freeze all or N seeded sampled queries."""
    n_queries = query_count(n_queries)
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError("Dataset already exists. Use a new version directory; never overwrite it.")
    source = ROOT / "univ-bench-inst.ttl"
    query_source = ROOT / "owlapy_generated_qa.json"
    graph = rdflib.Graph().parse(source, format="turtle")
    rows = json.loads(query_source.read_text(encoding="utf-8"))
    oracle = Oracle()
    candidates, seen, excluded = [], set(), []
    for index, row in enumerate(rows):
        if row.get("answer") is not True or not row.get("reasoning_type", "").startswith("subClassOf"):
            continue
        query = (str(NS[row["sub"]]), str(NS[row["sup"]]))
        if query in seen:
            continue
        seen.add(query)
        proof = named_path(graph, query)
        if proof:
            candidates.append((query, proof, index, row))
        else:
            excluded.append({"source_index": index, "reason": "no nonempty explicit named-subclass proof"})
    eligible = []
    for query, proof, index, row in sorted(candidates, key=lambda c: c[0]):
        if not oracle.entails(graph, query):
            excluded.append({"source_index": index, "reason": "positive label not confirmed"})
            continue
        chosen = None
        for edge in reversed(proof):
            withdrawn = copy_graph(graph)
            withdrawn.remove(edge)
            if not oracle.entails(withdrawn, query):
                chosen = (edge, withdrawn)
                break
        if chosen is None:
            excluded.append({"source_index": index, "reason": "no single proof edge kills entailment"})
            continue
        edge, withdrawn = chosen
        eligible.append(({
            "query": list(query), "source_index": index, "source_question": row["question"],
            "source_reasoning_type": row["reasoning_type"], "explicit_proof_depth": len(proof),
            "reference_support": [triple_json(t) for t in proof],
            "removed_axioms": [triple_json(edge)],
            "expected": {"original": "entailed", "withdrawal": "not_entailed"},
        }, withdrawn))
    selected = select_verified_cases(eligible, n_queries, seed)
    for index, (case, _) in enumerate(selected, 1):
        case["id"] = f"subclass_{index:02d}"
    destination.mkdir(parents=True)
    # Preserve snapshot bytes and hashes when committing on Windows.
    (destination / ".gitattributes").write_text(
        "*.ttl -text whitespace=cr-at-eol,-blank-at-eof\ndataset.json -text whitespace=cr-at-eol\n",
        encoding="utf-8")
    files = {}
    def snapshot(filename, g):
        text = g.serialize(format="turtle")
        if not isomorphic(g, rdflib.Graph().parse(data=text, format="turtle")):
            raise AssertionError("Serialization lost triples.")
        (destination / filename).write_text(text, encoding="utf-8")
        files[filename] = digest((destination / filename).read_bytes())
    snapshot("ontology_original.ttl", graph)
    cases = []
    for case, withdrawn in selected:
        case["snapshots"] = {"original": "ontology_original.ttl", "withdrawal": case["id"] + "_withdrawal.ttl"}
        snapshot(case["snapshots"]["withdrawal"], withdrawn)
        cases.append(case)
    data = {
        "version": destination.name, "schema_version": 2, "seed": seed, "cases": cases,
        "requested_n_queries": n_queries, "eligible_queries": len(eligible),
        "selected_queries": len(selected),
        "eligible_by_depth": dict(sorted(Counter(str(c[0]["explicit_proof_depth"]) for c in eligible).items())),
        "selected_by_depth": dict(sorted(Counter(str(c[0]["explicit_proof_depth"]) for c in selected).items())),
        "selection": "all eligible or seeded uniform sample without replacement; no per-depth quota or subject uniqueness constraint; single critical edge withdrawal",
        "positive_source_queries": len(seen), "candidate_queries": len(candidates),
        "selection_exclusions": excluded,
        "sources": {source.name: digest(source.read_bytes()), query_source.name: digest(query_source.read_bytes())},
        "files": files, "oracle": "HermiT via Owlready2", "owlready2": importlib.metadata.version("owlready2"),
    }
    save_json(destination / "dataset.json", data)
    return data


def load_dataset(directory=DATASET):
    directory = Path(directory)
    raw = (directory / "dataset.json").read_bytes()
    data = json.loads(raw)
    for filename, expected in data["files"].items():
        if digest((directory / filename).read_bytes()) != expected:
            raise ValueError(f"Frozen snapshot changed: {filename}")
    for filename, expected in data["sources"].items():
        if digest((ROOT / filename).read_bytes()) != expected:
            raise ValueError(f"Source changed: {filename}. Build a new dataset version.")
    graphs = {f: rdflib.Graph().parse(directory / f, format="turtle") for f in data["files"]}
    texts = {f: (directory / f).read_text(encoding="utf-8") for f in data["files"]}
    oracle = Oracle()
    original_source = rdflib.Graph().parse(ROOT / "univ-bench-inst.ttl", format="turtle")
    for case in data["cases"]:
        original, withdrawn = (graphs[case["snapshots"][c]] for c in ("original", "withdrawal"))
        if not isomorphic(original, original_source):
            raise ValueError("Original context is not the full source ontology.")
        expected_withdrawn = copy_graph(original)
        for a in case["removed_axioms"]:
            if json_triple(a) not in expected_withdrawn:
                raise ValueError("Removed axiom absent from original.")
            expected_withdrawn.remove(json_triple(a))
        if not isomorphic(expected_withdrawn, withdrawn):
            raise ValueError("Withdrawal changes more than the declared axioms.")
        for condition in ("original", "withdrawal"):
            actual = "entailed" if oracle.entails(graphs[case["snapshots"][condition]], case["query"]) else "not_entailed"
            if actual != case["expected"][condition]:
                raise ValueError(f"Oracle disagrees: {case['id']} {condition}")
    return data, graphs, texts, oracle, digest(raw)


def message(case, condition, texts):
    prefix = ("This is the initial current ontology. Evaluate the query." if condition == "original" else
              "This is the new current ontology. It replaces the previous ontology. Re-evaluate the same query.")
    return {"role": "user", "content": INSTRUCTIONS + "\n" + prefix
            + "\nCURRENT ONTOLOGY (Turtle):\n" + texts[case["snapshots"][condition]]
            + f"\nQUERY: Is <{case['query'][0]}> a subclass of <{case['query'][1]}>?"}


def api(host, endpoint, payload=None, timeout=600):
    request = urllib.request.Request(host + endpoint, data=None if payload is None else json_bytes(payload),
                                     headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"Ollama HTTP {exc.code}: {exc.read().decode('utf-8', errors='replace')}") from exc


def support_graph(triples, query):
    graph = rdflib.Graph()
    for uri in query:
        graph.add((URIRef(uri), RDF.type, OWL.Class))
    for sub, pred, sup in triples:
        graph.add((sub, pred, sup))
        graph.add((sub, RDF.type, OWL.Class))
        graph.add((sup, RDF.type, OWL.Class))
    return graph


def score_response(record, case, graphs, oracle):
    condition = record["condition"]
    graph = graphs[case["snapshots"][condition]]
    original = graphs[case["snapshots"]["original"]]
    row = {"case_id": case["id"], "condition": condition, "gold": case["expected"][condition],
           "answer": None, "valid_output": record["parsed"] is not None,
           "answer_correct": False, "citations_current": None, "support_sufficient": None,
           "support_minimal": None, "stale_axioms": [], "absent_axioms": [],
           "success": False, "error_kind": record.get("error_kind")}
    parsed = record["parsed"]
    if parsed is None:
        return row
    triples = {json_triple(a) for a in parsed["axioms"]}
    absent = triples - set(graph)
    stale = absent & set(original)
    row.update(answer=parsed["answer"], answer_correct=parsed["answer"] == row["gold"],
               citations_current=not absent,
               stale_axioms=[triple_json(t) for t in sorted(stale)],
               absent_axioms=[triple_json(t) for t in sorted(absent)])
    if parsed["answer"] == "entailed":
        sufficient = not absent and oracle.entails(support_graph(triples, case["query"]), case["query"])
        minimal = sufficient and all(not oracle.entails(support_graph(triples - {t}, case["query"]), case["query"])
                                      for t in triples)
        row.update(support_sufficient=sufficient, support_minimal=minimal,
                   success=row["answer_correct"] and sufficient)
    else:
        # Empty evidence is an output convention, NOT a proof of non-entailment.
        row["success"] = row["answer_correct"] and not triples
    return row


def evaluate(records, data, graphs, oracle):
    keys = [(r["case_id"], r["condition"]) for r in records]
    required = {(c["id"], stage) for c in data["cases"] for stage in ("original", "withdrawal")}
    if len(keys) != len(set(keys)) or set(keys) != required:
        raise ValueError("Evaluation requires exactly one record per case and condition.")
    cases = {c["id"]: c for c in data["cases"]}
    rows = [score_response(r, cases[r["case_id"]], graphs, oracle) for r in records]
    lookup = {(r["case_id"], r["condition"]): r for r in rows}
    pairs = [{"case_id": c["id"], "depth": c["explicit_proof_depth"],
              "original_success": lookup[c["id"], "original"]["success"],
              "withdrawal_success": lookup[c["id"], "withdrawal"]["success"],
              "pair_success": lookup[c["id"], "original"]["success"] and lookup[c["id"], "withdrawal"]["success"]}
             for c in data["cases"]]
    n = len(pairs)
    infrastructure = sum(r.get("error_kind") in ("infrastructure", "context", "dependency") for r in records)
    originals = [r for r in rows if r["condition"] == "original"]
    complete = infrastructure == 0
    successes = sum(p["pair_success"] for p in pairs)
    summary = {
        "n_pairs": n, "n_responses": len(rows), "complete_for_comparison": complete,
        "infrastructure_or_context_errors": infrastructure,
        "format_errors": sum(r.get("error_kind") == "format" for r in records),
        "pair_success_count": successes,
        "paired_grounded_revision_score": successes / n if complete else None,
        "original_grounded_score": sum(p["original_success"] for p in pairs) / n if complete else None,
        "withdrawal_score": sum(p["withdrawal_success"] for p in pairs) / n if complete else None,
        "original_answer_accuracy": sum(r["answer_correct"] for r in originals) / n if complete else None,
        "correct_original_without_valid_support_count": sum(r["answer_correct"] and not r["success"] for r in originals),
        "withdrawal_stale_citation_count": sum(bool(r["stale_axioms"]) for r in rows if r["condition"] == "withdrawal"),
    }
    by_depth = {}
    for depth in sorted({p["depth"] for p in pairs}):
        group = [p for p in pairs if p["depth"] == depth]
        by_depth[str(depth)] = {
            "n_pairs": len(group),
            "original_grounded_score": sum(p["original_success"] for p in group) / len(group) if complete else None,
            "withdrawal_score": sum(p["withdrawal_success"] for p in group) / len(group) if complete else None,
            "paired_grounded_revision_score": sum(p["pair_success"] for p in group) / len(group) if complete else None,
        }
    return {"summary": summary, "by_depth": by_depth, "pairs": pairs, "responses": rows}


def run_benchmark(model="qwen2.5-coder:7b", directory=DATASET, options=None):
    options = dict(OPTIONS if options is None else options)
    data, graphs, texts, oracle, dataset_hash = load_dataset(directory)
    host = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
    if not host.startswith(("http://", "https://")):
        host = "http://" + host
    available = api(host, "/api/tags")["models"]
    info = next((m for m in available if m["name"] == model), None)
    if not info:
        raise ValueError(f"Model not installed: {model}; available: {[m['name'] for m in available]}")
    model_show = api(host, "/api/show", {"model": model})
    context_limits = [v for k, v in model_show.get("model_info", {}).items() if k.endswith(".context_length")]
    if context_limits and min(context_limits) < options["num_ctx"]:
        raise ValueError("Model context limit is below the fixed protocol window.")
    # Keep model tags readable while avoiding invalid path characters on Windows.
    model_slug = re.sub(r"[^A-Za-z0-9._-]+", "_", model).strip("._-")[:100] or "model"
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + model_slug + "_" + uuid.uuid4().hex[:8]
    out = ROOT / "generated" / "subclass_grounding" / run_id
    out.mkdir(parents=True, exist_ok=False)
    fingerprint = digest(json_bytes({"dataset": dataset_hash, "instructions": INSTRUCTIONS, "schema": SCHEMA,
                                    "options": options, "evaluator": digest(Path(__file__).read_bytes())}))
    manifest = {"run_id": run_id, "model": info, "ollama_version": api(host, "/api/version"),
                "model_details": model_show, "dataset_sha256": dataset_hash, "protocol_fingerprint": fingerprint,
                "options": options, "instructions": INSTRUCTIONS, "schema": SCHEMA,
                "dataset": data, "platform": platform.platform(), "python": platform.python_version(),
                "java": subprocess.run([owlready2.JAVA_EXE, "-version"], capture_output=True, text=True).stderr,
                "packages": {p: importlib.metadata.version(p) for p in ("rdflib", "owlready2", "jsonschema")}}
    save_json(out / "manifest.json", manifest)
    # Freeze exact contexts alongside each run, so it remains inspectable independently.
    (out / "snapshots").mkdir()
    for filename, text in texts.items():
        (out / "snapshots" / filename).write_text(text, encoding="utf-8")
    records = []

    def append(record):
        records.append(record)
        with (out / "responses.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(record["case_id"], record["condition"], record["parsed"] or record["error"], flush=True)

    def call(case, condition, messages):
        record = {"case_id": case["id"], "condition": condition, "parsed": None,
                  "response": None, "error": None, "error_kind": None}
        payload = {"model": model, "messages": messages, "format": SCHEMA, "stream": False, "options": options}
        record["request"] = payload
        start = time.perf_counter()
        budget = len(json_bytes(messages)) + len(json_bytes(SCHEMA)) + 4096 + options["num_predict"]
        record["conservative_context_budget"] = budget
        if budget > options["num_ctx"]:
            record.update(error="Conservative context budget exceeded; request not sent.", error_kind="context")
        else:
            try:
                response = api(host, "/api/chat", payload)
                record["response"] = response
                ps = api(host, "/api/ps")
                record["ollama_ps"] = ps
                loaded = next((m for m in ps.get("models", []) if m.get("digest") == info["digest"]), {})
                context = loaded.get("context_length")
                if context is None or context < budget:
                    record.update(error="Actual context capacity not verified or insufficient.", error_kind="context")
                elif response.get("prompt_eval_count", 0) + options["num_predict"] >= options["num_ctx"]:
                    record.update(error="Prompt too close to context limit.", error_kind="context")
                else:
                    try:
                        if response.get("done_reason") == "length":
                            raise ValueError("Generation reached output token limit.")
                        parsed = json.loads(response["message"]["content"])
                        jsonschema.validate(parsed, SCHEMA)
                        record["parsed"] = parsed
                    except (ValueError, KeyError, jsonschema.ValidationError) as exc:
                        record.update(error=str(exc), error_kind="format")
            except Exception as exc:
                record.update(error=f"{type(exc).__name__}: {exc}", error_kind="infrastructure")
        record["elapsed_s"] = round(time.perf_counter() - start, 3)
        append(record)
        return record

    for case in data["cases"]:
        initial = message(case, "original", texts)
        first = call(case, "original", [initial])
        if first["response"] is not None and first["error_kind"] not in ("context", "infrastructure"):
            # Preserve actual output even when malformed. No gold answer, repaired answer or feedback.
            previous = {"role": "assistant", "content": first["response"]["message"]["content"]}
            call(case, "withdrawal", [initial, previous, message(case, "withdrawal", texts)])
        else:
            append({"case_id": case["id"], "condition": "withdrawal", "parsed": None,
                    "response": None, "error": "Missing usable initial call; withdrawal not run.",
                    "error_kind": "dependency", "elapsed_s": 0})
    result = evaluate(records, data, graphs, oracle)
    save_json(out / "evaluation.json", result)
    print(json.dumps(result["summary"], indent=2), flush=True)
    print("Run:", out, flush=True)
    return out, result


def compare_runs(directories):
    rows, fingerprints = [], set()
    for directory in map(Path, directories):
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        result = json.loads((directory / "evaluation.json").read_text(encoding="utf-8"))
        fingerprints.add(manifest["protocol_fingerprint"])
        if not result["summary"]["complete_for_comparison"]:
            raise ValueError(f"Incomplete run, excluded from comparison: {directory}")
        rows.append({"model": manifest["model"]["name"], "digest": manifest["model"]["digest"],
                     "run": str(directory), **result["summary"]})
    if len(fingerprints) > 1:
        raise ValueError("Different datasets, prompts, evaluator or options: scores are not directly comparable.")
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["build", "run"])
    parser.add_argument("--model", default="qwen2.5-coder:7b")
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--num-queries", type=query_count, default="all",
                        help="Build only: positive number of queries or 'all' (default).")
    parser.add_argument("--seed", type=int, default=20260929, help="Build sampling seed.")
    args = parser.parse_args()
    if args.command == "run" and (args.num_queries != "all" or args.seed != 20260929):
        parser.error("--num-queries and --seed configure build, not run; use a frozen dataset.")
    if args.command == "build":
        dataset = build_dataset(args.dataset, seed=args.seed, n_queries=args.num_queries)
        print(json.dumps({k: dataset[k] for k in ("version", "requested_n_queries", "eligible_queries", "selected_queries", "selected_by_depth")}, indent=2))
    else:
        run_benchmark(args.model, args.dataset)
