"""Generate, freeze and evaluate transitivity-withdrawal cases. No work at import time.

Only explicit function calls generate datasets, invoke HermiT or contact Ollama.
Positive property entailment is checked by inconsistency of O + not(query),
not by treating missing materialized property values as negative answers.
"""
from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import re
import subprocess
import sys
import tempfile
import time
import urllib.request
import uuid

import jsonschema
import rdflib
from rdflib import BNode, RDF, RDFS, OWL, URIRef
from rdflib.compare import to_isomorphic

ROOT = Path(__file__).resolve().parent
OPTIONS = {"temperature": 0, "seed": 42, "num_ctx": 32768, "num_predict": 768}
URI_PATTERN = r"^https?://[^\s<>]+$"
SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["answer", "axioms"],
    "properties": {
        "answer": {"type": "string", "enum": ["entailed", "not_entailed"]},
        "axioms": {"type": "array", "items": {
            "type": "object", "additionalProperties": False,
            "required": ["subject", "predicate", "object"],
            "properties": {key: {"type": "string", "pattern": URI_PATTERN}
                           for key in ("subject", "predicate", "object")}}},
    },
}
GENERATION_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["generated_onto", "target_property"],
    "properties": {
        "generated_onto": {"type": "string", "minLength": 1},
        "target_property": {"type": "string", "pattern": URI_PATTERN},
    },
}
INSTRUCTIONS = """Evaluate only the current ontology using OWL semantics.
Return JSON with answer (entailed or not_entailed) and axioms.
For entailed, cite asserted triples from the current ontology sufficient to entail
exactly the queried object property assertion. Include the transitivity declaration
when needed. Never cite an inferred conclusion as an asserted premise.
For not_entailed return axioms: []. Not entailed does not mean explicitly false.
A new current ontology replaces the previous one, including its property declarations.
Do not assume property characteristics from names or from the previous snapshot.
"""


@dataclass(frozen=True)
class GenerationConfig:
    generator: str = "procedural"
    domain: str = "only_symbolic"
    seed: int = 42
    path_length: int = 3
    n_distractor_triples: int = 10
    n_queries: int = 1
    max_generation_attempts: int = 3
    # If true, queries have an explicit shortcut and remain entailed after withdrawal.
    direct_assertion_control: bool = False

    def __post_init__(self):
        if self.generator not in {"procedural", "llm"}:
            raise ValueError("generator must be procedural or llm")
        if not isinstance(self.domain, str) or not self.domain:
            raise ValueError("domain must be a nonempty string")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError("seed must be an integer")
        for name, minimum in (("path_length", 2), ("n_distractor_triples", 0),
                              ("n_queries", 1), ("max_generation_attempts", 1)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if not isinstance(self.direct_assertion_control, bool):
            raise ValueError("direct_assertion_control must be boolean")


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def json_bytes(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8")


def save_json(path, value):
    Path(path).write_bytes(json.dumps(value, indent=2, ensure_ascii=False).encode("utf-8"))


def graph_id(graph):
    return digest(str(to_isomorphic(graph).graph_digest()).encode())


def copy_graph(graph):
    result = rdflib.Graph()
    for prefix, ns in graph.namespaces():
        result.bind(prefix, ns)
    for triple in graph:
        result.add(triple)
    return result


def triple_json(triple):
    return dict(zip(("subject", "predicate", "object"), map(str, triple)))


def json_triple(value):
    return tuple(URIRef(value[k]) for k in ("subject", "predicate", "object"))


def parse_ontology(obj):
    if obj.get("format") != "ttl":
        raise ValueError("Only Turtle input is supported")
    return rdflib.Graph().parse(data=obj["generated_onto"], format="turtle")


def generation_request(config):
    return (f"Generate a consistent OWL ontology in Turtle for domain {config.domain!r}. "
            "Use absolute HTTP(S) IRIs. Return a JSON object with generated_onto "
            "(a Turtle string) and target_property (an absolute IRI). Include a declared object "
            "property typed owl:TransitiveProperty; identify it using target_property "
            "in the returned object. Use only named entities: no blank nodes, "
            "anonymous restrictions, RDF lists or imports. Other named classes and "
            "properties may provide domain context. Do not materialize inferred facts. "
            "Individuals, benchmark chains, distractor facts and queries will be inserted "
            "by a separate builder; do not generate them or answer any benchmark query. "
            "For only_symbolic use neutral symbols without domain meanings.")


def make_ollama_generator(model, *, options=None, host=None, timeout_s=600, transport=None, think=None,
                          on_response=None):
    """Create the llm_generator callback. Construction does NOT contact Ollama.

    Only invoking the returned callback contacts an already installed model.
    It requests structured JSON containing Turtle and the target property, checks
    format/syntax, and records exact prompts, response, digest and actual options.
    Semantic validation and reference answers remain the oracle's responsibility.
    transport, when supplied, must have the same signature as api (including timeout).
    """
    if not isinstance(model, str) or not model.strip():
        raise ValueError("model must be an exact nonempty installed model name")
    if timeout_s <= 0:
        raise ValueError("timeout_s must be positive")
    if think is not None and not isinstance(think, (bool, str)):
        raise ValueError("think must be None, a boolean or a supported reasoning level")
    generator_options = {"temperature": 0.6, "num_ctx": 32768, "num_predict": 4096,
                         **(options or {})}
    for key in ("num_ctx", "num_predict"):
        if type(generator_options[key]) is not int or generator_options[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    configured_host = host

    def llm_generator(*, prompt, system_prompt, config):
        endpoint_host = configured_host or os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        if not endpoint_host.startswith(("http://", "https://")):
            endpoint_host = "http://" + endpoint_host
        send = transport or api

        def request(endpoint, payload=None):
            return send(endpoint_host, endpoint, payload, timeout=timeout_s)

        actual_options = {**generator_options}
        actual_options.setdefault("seed", config["seed"])
        # Config controls separate pipeline stages; it is metadata, not a gold label.
        user_prompt = prompt + "\nGENERATION CONFIGURATION:\n" + json.dumps(config, sort_keys=True)
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})
        payload = {"model": model, "messages": messages, "format": GENERATION_SCHEMA,
                   "stream": False, "options": actual_options}
        if think is not None:
            payload["think"] = think
        budget = len(json_bytes(messages)) + len(json_bytes(GENERATION_SCHEMA)) + 4096 + actual_options["num_predict"]
        if budget > actual_options["num_ctx"]:
            raise ValueError("Generator context budget exceeded; request not sent")
        info = next((m for m in request("/api/tags")["models"] if m["name"] == model), None)
        if info is None:
            raise ValueError(f"Generator model not installed: {model}")
        details = request("/api/show", {"model": model})
        limits = [v for k, v in details.get("model_info", {}).items() if k.endswith(".context_length")]
        if limits and min(limits) < actual_options["num_ctx"]:
            raise ValueError("Generator model capacity is below the requested window")
        version = request("/api/version")
        start = time.perf_counter()
        response = request("/api/chat", payload)
        elapsed_s = round(time.perf_counter() - start, 3)
        if on_response:
            on_response({"request": payload, "response": response, "model_info": info,
                         "model_details": details, "ollama_version": version, "elapsed_s": elapsed_s})
        if response.get("done_reason") == "length":
            raise ValueError("Ontology generation reached the output token limit")
        content = response["message"]["content"]
        parsed = json.loads(content)
        jsonschema.validate(parsed, GENERATION_SCHEMA)
        graph = rdflib.Graph().parse(data=parsed["generated_onto"], format="turtle")
        target = URIRef(parsed["target_property"])
        if (target, RDF.type, OWL.ObjectProperty) not in graph or (target, RDF.type, OWL.TransitiveProperty) not in graph:
            raise ValueError("Generated target lacks explicit object-property/transitivity declarations")
        if any(graph.triples((None, OWL.imports, None))):
            raise ValueError("Generated ontology must not contain imports")
        if any(isinstance(term, BNode) for triple in graph for term in triple):
            raise ValueError("Generated ontology contains unsupported blank-node axioms")
        ps = request("/api/ps")
        loaded = next((m for m in ps.get("models", []) if m.get("digest") == info["digest"]), {})
        context = loaded.get("context_length")
        if context is None or context < budget or response.get("prompt_eval_count", 0) + actual_options["num_predict"] >= actual_options["num_ctx"]:
            raise ValueError("Generator actual context capacity unverified or insufficient")
        return {**parsed, "provenance": {
            "generator": "ollama", "model": model, "model_info": info,
            "model_details": details, "ollama_version": version,
            "options": actual_options, "generation_config": dict(config),
            "request": payload, "response": response, "ollama_ps": ps,
            "conservative_context_budget": budget, "elapsed_s": elapsed_s,
        }}

    return llm_generator


def generate_onto(config=None, *, llm_generator=None, system_prompt=""):
    """Return Turtle and provenance. LLM generation requires an explicit callback.

    Callback contract: llm_generator(prompt=..., system_prompt=..., config=...)
    returns generated_onto (Turtle), target_property (IRI), provenance (model/options).
    Domain is metadata for the procedural generator; it does not generate domain prose.
    """
    config = config or GenerationConfig()
    requested = asdict(config)
    if config.generator == "llm":
        if llm_generator is None:
            raise ValueError("Supply llm_generator explicitly; there is no default LLM call")
        print("generating ontology using llm")
        generated = llm_generator(prompt=generation_request(config),
                                  system_prompt=system_prompt, config=requested)
        for key in ("generated_onto", "target_property", "provenance"):
            if key not in generated:
                raise ValueError(f"LLM generator omitted {key}")
        return {**generated, "format": "ttl", "domain": config.domain, "requested": requested}
    token = digest(json_bytes(requested))[:16]
    ns = rdflib.Namespace(f"https://example.org/transitivity/{token}#")
    graph = rdflib.Graph()
    graph.bind("ex", ns)
    graph.add((ns.ontology, RDF.type, OWL.Ontology))
    graph.add((ns.p, RDF.type, OWL.ObjectProperty))
    graph.add((ns.p, RDF.type, OWL.TransitiveProperty))
    return {"generated_onto": graph.serialize(format="turtle"), "format": "ttl",
            "domain": config.domain, "target_property": str(ns.p), "requested": requested,
            "provenance": {"generator": "procedural", "seed": config.seed}}


# A subprocess bounds reasoner time and isolates every graph, including failed loads.
_REASONER_WORKER = '''import json, sys, owlready2
world = owlready2.World()
try:
    world.get_ontology("https://example.org/transitivity/check").load(fileobj=open(sys.argv[1], "rb"))
    try:
        owlready2.sync_reasoner(world, debug=0)
        print(json.dumps({"consistent": True}))
    except owlready2.OwlReadyInconsistentOntologyError:
        print(json.dumps({"consistent": False}))
finally:
    world.close()
'''


class PropertyOracle:
    """HermiT consistency/refutation oracle; errors and timeouts always propagate."""
    def __init__(self, timeout_s=120):
        if timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        self.timeout_s = timeout_s
        self.cache = {}

    def consistent(self, graph):
        if any(graph.triples((None, OWL.imports, None))):
            raise ValueError("External imports are not supported in frozen cases")
        key = graph_id(graph)
        if key not in self.cache:
            with tempfile.TemporaryDirectory(prefix="transitivity-oracle-") as directory:
                path = Path(directory) / "ontology.rdf"
                path.write_bytes(graph.serialize(format="xml", encoding="utf-8"))
                proc = subprocess.run([sys.executable, "-c", _REASONER_WORKER, str(path)],
                                      capture_output=True, text=True, timeout=self.timeout_s)
            if proc.returncode:
                raise RuntimeError(f"HermiT failed: {proc.stderr.strip()}")
            # Unexpected output is an infrastructure error, never non-entailment.
            result = json.loads(proc.stdout.strip())
            if type(result.get("consistent")) is not bool:
                raise RuntimeError("Invalid reasoner result")
            self.cache[key] = result["consistent"]
        return self.cache[key]

    def entails(self, graph, query):
        if not self.consistent(graph):
            raise ValueError("Inconsistent ontology: vacuous entailment excluded")
        probe = copy_graph(graph)
        assertion = BNode()
        probe.add((assertion, RDF.type, OWL.NegativePropertyAssertion))
        probe.add((assertion, OWL.sourceIndividual, URIRef(query["subject"])))
        probe.add((assertion, OWL.assertionProperty, URIRef(query["property"])))
        probe.add((assertion, OWL.targetIndividual, URIRef(query["object"])))
        return not self.consistent(probe)


def validate_onto(onto_obj, oracle):
    """Check syntax, frozen dependencies, named object property and consistency.

    This is not a complete OWL 2 DL profile validator. Complex LLM ontologies need
    additional profile checks; reasoner failures are reported as validation errors.
    """
    report = {"valid": False, "errors": [], "metadata": {}}
    try:
        graph = parse_ontology(onto_obj)
        target = URIRef(onto_obj["target_property"])
        if not re.fullmatch(URI_PATTERN, str(target)):
            raise ValueError("target_property must be an absolute HTTP(S) IRI")
        if any(graph.triples((None, OWL.imports, None))):
            raise ValueError("Imports are not allowed")
        if (target, RDF.type, OWL.ObjectProperty) not in graph:
            raise ValueError("Target must be explicitly declared owl:ObjectProperty")
        if (target, RDF.type, OWL.TransitiveProperty) not in graph:
            raise ValueError("Target transitivity declaration is missing")
        # Keep the current structured evidence format lossless.
        if any(isinstance(term, BNode) for triple in graph for term in triple):
            raise ValueError("Blank-node axioms require an extended evidence format")
        characteristics = sorted({str(o) for s, p, o in graph
                                  if p == RDF.type and str(o).startswith(str(OWL))
                                  and str(o).endswith("Property")})
        report["metadata"] = {"n_triples": len(graph), "graph_sha256": graph_id(graph),
                              "property_characteristics": characteristics,
                              "n_named_individuals": len(set(graph.subjects(RDF.type, OWL.NamedIndividual)))}
        if not oracle.consistent(graph):
            raise ValueError("Ontology is inconsistent")
        report["valid"] = True
    except Exception as exc:
        report["errors"].append(f"{type(exc).__name__}: {exc}")
    return report


def add_individuals_and_facts(onto_obj, config):
    """Insert disjoint fresh chains; requested length is checked again in selection."""
    graph = parse_ontology(onto_obj)
    target = URIRef(onto_obj["target_property"])
    ns = rdflib.Namespace(f"https://example.org/case/{uuid.UUID(hex=digest(json_bytes(asdict(config)))[:32])}#")
    # An external generator may already use this namespace: never silently reuse nodes.
    if any(str(term).startswith(str(ns)) for triple in graph for term in triple):
        raise ValueError("Fresh individual namespace collides with generated ontology")
    endpoints = []
    edges = []
    for index in range(config.n_queries):
        nodes = [ns[f"q{index}_n{i}"] for i in range(config.path_length + 1)]
        for node in nodes:
            graph.add((node, RDF.type, OWL.NamedIndividual))
        edges.extend((a, target, b) for a, b in zip(nodes, nodes[1:]))
        if config.direct_assertion_control:
            edges.append((nodes[0], target, nodes[-1]))
        endpoints.append({"subject": str(nodes[0]), "property": str(target), "object": str(nodes[-1])})
    random.Random(config.seed).shuffle(edges)
    for edge in edges:
        graph.add(edge)
    if config.n_distractor_triples:
        graph.add((ns.otherProperty, RDF.type, OWL.ObjectProperty))
    for i in range(config.n_distractor_triples):
        a, b = ns[f"d{i}_a"], ns[f"d{i}_b"]
        graph.add((a, RDF.type, OWL.NamedIndividual))
        graph.add((b, RDF.type, OWL.NamedIndividual))
        graph.add((a, ns.otherProperty, b))
    return {**onto_obj, "generated_onto": graph.serialize(format="turtle"),
            "candidate_queries": endpoints, "ontology_id": graph_id(graph)}


def property_path(graph, query):
    start, prop, end = (URIRef(query[k]) for k in ("subject", "property", "object"))
    queue, seen = deque([(start, [])]), {start}
    while queue:
        node, path = queue.popleft()
        if node == end:
            return path
        for nxt in sorted(graph.objects(node, prop), key=str):
            if isinstance(nxt, URIRef) and nxt not in seen:
                seen.add(nxt)
                queue.append((nxt, path + [(node, prop, nxt)]))
    return None


def select_candidate_queries(onto_obj, config):
    graph = parse_ontology(onto_obj)
    candidates = []
    for query in onto_obj.get("candidate_queries", []):
        path = property_path(graph, query)
        if not path:
            continue
        if config.direct_assertion_control:
            if len(path) != 1:
                continue
        elif len(path) != config.path_length or len(path) < 2:
            continue
        support = list(path)
        if len(path) > 1:
            support.append((URIRef(query["property"]), RDF.type, OWL.TransitiveProperty))
        candidates.append({"query": query, "explicit_path_length": len(path),
                           "reference_support": [triple_json(t) for t in support]})
    if len(candidates) != config.n_queries:
        raise ValueError(f"Requested {config.n_queries} queries but found {len(candidates)} candidates")
    return candidates


def withdraw_transitivity(original_graph, target_property):
    retired = (URIRef(target_property), RDF.type, OWL.TransitiveProperty)
    if retired not in original_graph:
        raise ValueError("The selected transitivity declaration is not asserted")
    result = copy_graph(original_graph)
    result.remove(retired)
    if set(original_graph) - set(result) != {retired} or set(result) - set(original_graph):
        raise ValueError("Withdrawal must remove exactly one transitivity declaration")
    return result


def support_graph(triples, query):
    graph = rdflib.Graph()
    # Harmless role declarations; do not type subjects/objects as classes.
    graph.add((URIRef(query["property"]), RDF.type, OWL.ObjectProperty))
    graph.add((URIRef(query["subject"]), RDF.type, OWL.NamedIndividual))
    graph.add((URIRef(query["object"]), RDF.type, OWL.NamedIndividual))
    for triple in triples:
        graph.add(triple)
    return graph


def verify_case(original_graph, withdrawal_graph, candidate, oracle, *, preserve=False):
    """Raise on all unverified cases; never replace reasoner errors with a negative."""
    query = candidate["query"]
    expected_graph = withdraw_transitivity(original_graph, query["property"])
    if set(expected_graph) != set(withdrawal_graph):
        raise ValueError("Snapshots differ by more than the selected declaration")
    if not oracle.consistent(original_graph) or not oracle.consistent(withdrawal_graph):
        raise ValueError("Both snapshots must be consistent")
    if not oracle.entails(original_graph, query):
        raise ValueError("Original query is not entailed")
    after = oracle.entails(withdrawal_graph, query)
    if after != preserve:
        raise ValueError("Unexpected entailment after withdrawal")
    triples = {json_triple(t) for t in candidate["reference_support"]}
    if not triples <= set(original_graph) or not oracle.entails(support_graph(triples, query), query):
        raise ValueError("Reference support is absent or insufficient")
    return {"consistent_original": True, "consistent_withdrawal": True,
            "original_entailed": True, "withdrawal_entailed": after,
            "reference_support_verified": True, "exact_withdrawal_verified": True}


def prepare_cases(configs, oracle, *, llm_generator=None, system_prompt=""):
    """Explicit preparation, bounded retries, ontology groups and rejection logs."""
    cases, snapshots, attempts = [], {}, []
    ontology_ids = set()
    for index, config in enumerate(configs):
        attempt_system_prompt = system_prompt
        for attempt in range(config.max_generation_attempts):
            try:
                obj = generate_onto(config, llm_generator=llm_generator, system_prompt=attempt_system_prompt)
                validation = validate_onto(obj, oracle)
                if not validation["valid"]:
                    raise ValueError("; ".join(validation["errors"]))
                obj = add_individuals_and_facts(obj, config)
                validation = validate_onto(obj, oracle)
                if not validation["valid"]:
                    raise ValueError("; ".join(validation["errors"]))
                original = parse_ontology(obj)
                ontology_id = graph_id(original)
                if ontology_id in ontology_ids:
                    raise ValueError("Duplicate ontology: use a different generation seed/configuration")
                withdrawn = withdraw_transitivity(original, obj["target_property"])
                candidates = select_candidate_queries(obj, config)
                verified = [verify_case(original, withdrawn, c, oracle,
                                        preserve=config.direct_assertion_control) for c in candidates]
                names = {stage: f"ontology_{index:04d}_{stage}.ttl" for stage in ("original", "withdrawal")}
                snapshots[names["original"]] = original.serialize(format="turtle")
                snapshots[names["withdrawal"]] = withdrawn.serialize(format="turtle")
                for qindex, (candidate, report) in enumerate(zip(candidates, verified)):
                    cases.append({"id": f"transitive_{index:04d}_{qindex:03d}",
                                  "ontology_id": ontology_id, "snapshots": names,
                                  **candidate, "expected": {"original": "entailed", "withdrawal":
                                      "entailed" if config.direct_assertion_control else "not_entailed"},
                                  "removed_axioms": [triple_json((URIRef(obj["target_property"]), RDF.type, OWL.TransitiveProperty))],
                                  "requested": asdict(config), "observed": validation["metadata"],
                                  "provenance": obj["provenance"], "verification": report})
                ontology_ids.add(ontology_id)
                attempts.append({"config_index": index, "attempt": attempt + 1, "accepted": True})
                break
            except Exception as exc:
                attempts.append({"config_index": index, "attempt": attempt + 1,
                                 "accepted": False, "error": f"{type(exc).__name__}: {exc}"})
                if config.generator == "llm":
                    attempt_system_prompt = system_prompt + "\nPrevious generation was rejected: " + str(exc) + "\nCorrect this issue while preserving the requested constraints."
        else:
            raise RuntimeError("Generation attempts exhausted: " + json.dumps(attempts, ensure_ascii=False))
    if not cases:
        raise ValueError("No verified cases")
    return {"version": "transitive_withdrawal_v1", "cases": cases,
            "generation_attempts": attempts}, snapshots


def save_dataset(data, snapshots, destination):
    """Freeze a new dataset directory; no overwrite of existing artifacts."""
    destination = Path(destination)
    if not data.get("cases") or not all(c.get("verification", {}).get("exact_withdrawal_verified") for c in data["cases"]):
        raise ValueError("Only verified nonempty datasets can be frozen")
    hashes = {name: digest(text.encode("utf-8")) for name, text in snapshots.items()}
    if any(Path(name).name != name for name in snapshots):
        raise ValueError("Snapshot filenames must be simple basenames")
    for case in data["cases"]:
        for name in case["snapshots"].values():
            if name not in hashes:
                raise ValueError(f"Missing snapshot {name}")
    destination.mkdir(parents=True, exist_ok=False)
    for name, content in snapshots.items():
        (destination / name).write_bytes(content.encode("utf-8"))
    save_json(destination / "dataset.json", {**data, "snapshot_sha256": hashes})
    return destination


def generate_ontology_batch(configs, model, destination, *, options=None, think=None,
                            system_prompt="", oracle=None, llm_generator=None, on_progress=print):
    """Generate and save each ontology immediately, without evaluating an LLM.

    A batch contains separate frozen datasets, raw generator artifacts and an index.
    Failure leaves a readable partial batch, and never silently changes the model
    or experimental parameters. Existing directories are never overwritten.
    """
    configs = list(configs)
    if not configs or any(c.generator != "llm" for c in configs):
        raise ValueError("Supply a nonempty sequence of LLM generation configurations")
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    oracle = oracle or PropertyOracle()
    active_folder = None

    def record_raw_response(artifact):
        attempts_dir = active_folder / "generation_attempts"
        number = len(list(attempts_dir.glob("raw_*.json"))) + 1
        save_json(attempts_dir / f"raw_{number:02d}.json", artifact)

    generator = llm_generator or make_ollama_generator(model, options=options, think=think,
                                                       on_response=record_raw_response)
    manifest = {"version": "transitive_ontology_batch_v1", "status": "running", "model": model,
                "generation_options": dict(options or {}), "think": think,
                "system_prompt": system_prompt, "configs": [asdict(c) for c in configs],
                "requested_n_ontologies": len(configs), "items": [],
                "created_at_utc": datetime.now(timezone.utc).isoformat()}
    save_json(destination / "batch.json", manifest)
    try:
        for index, config in enumerate(configs):
            slug = re.sub(r"[^A-Za-z0-9._-]+", "_", config.domain).strip("._-") or "domain"
            folder = destination / f"{index + 1:02d}_{slug}"
            folder.mkdir()
            (folder / "generation_attempts").mkdir()
            active_folder = folder
            generated_objects = []

            def recording_generator(**kwargs):
                generated = generator(**kwargs)
                attempt = len(generated_objects) + 1
                generated_objects.append(generated)
                save_json(folder / "generation_attempts" / f"attempt_{attempt:02d}.json", generated)
                (folder / "generation_attempts" / f"attempt_{attempt:02d}.ttl").write_bytes(
                    generated["generated_onto"].encode("utf-8"))
                return generated

            if on_progress:
                on_progress(f"[{index + 1}/{len(configs)}] Generazione {config.domain}, seed={config.seed}")
            data, snapshots = prepare_cases([config], oracle, llm_generator=recording_generator,
                                           system_prompt=system_prompt)
            save_dataset(data, snapshots, folder / "dataset")
            (folder / "base.ttl").write_bytes(generated_objects[-1]["generated_onto"].encode("utf-8"))
            save_json(folder / "generation.json", generated_objects[-1])
            case = data["cases"][0]
            item = {"index": index + 1, "domain": config.domain, "seed": config.seed,
                    "directory": folder.name, "dataset_directory": f"{folder.name}/dataset",
                    "ontology_id": case["ontology_id"], "n_queries": len(data["cases"]),
                    "path_length": case["explicit_path_length"],
                    "n_distractor_triples": config.n_distractor_triples,
                    "n_original_triples": case["observed"]["n_triples"],
                    "model_digest": case["provenance"].get("model_info", {}).get("digest"),
                    "verification": case["verification"]}
            manifest["items"].append(item)
            save_json(destination / "batch.json", manifest)
            if on_progress:
                on_progress(f"[{index + 1}/{len(configs)}] Salvata e verificata: {folder}")
        manifest["status"] = "complete"
        manifest["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
        save_json(destination / "batch.json", manifest)
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        save_json(destination / "batch.json", manifest)
        raise
    return destination, manifest


def load_dataset(directory, *, oracle=None):
    directory = Path(directory)
    raw = (directory / "dataset.json").read_bytes()
    data = json.loads(raw)
    if data.get("version") != "transitive_withdrawal_v1" or not data.get("cases"):
        raise ValueError("Unsupported or empty dataset")
    texts, graphs = {}, {}
    for name, expected_hash in data["snapshot_sha256"].items():
        if Path(name).name != name:
            raise ValueError("Invalid snapshot filename")
        content = (directory / name).read_bytes()
        if digest(content) != expected_hash:
            raise ValueError(f"Snapshot hash mismatch: {name}")
        texts[name] = content.decode("utf-8")
        graphs[name] = rdflib.Graph().parse(data=texts[name], format="turtle")
    ids = [c["id"] for c in data["cases"]]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate case IDs")
    for case in data["cases"]:
        original, withdrawn = (graphs[case["snapshots"][stage]] for stage in ("original", "withdrawal"))
        if set(withdraw_transitivity(original, case["query"]["property"])) != set(withdrawn):
            raise ValueError("Invalid snapshot pair")
        if case["expected"]["original"] != "entailed" or case["expected"]["withdrawal"] not in {"entailed", "not_entailed"}:
            raise ValueError("Invalid expected answers")
        if oracle is not None:
            verify_case(original, withdrawn, case, oracle,
                        preserve=case["expected"]["withdrawal"] == "entailed")
    return data, graphs, texts, digest(raw)


def build_messages(case, texts, previous_response=None):
    query = case["query"]
    def user(stage):
        intro = "Initial current ontology." if stage == "original" else "New current ontology: it replaces the previous ontology. Re-evaluate the same query."
        return {"role": "user", "content": intro + "\nCURRENT ONTOLOGY (Turtle):\n" +
                texts[case["snapshots"][stage]] + "\nQUERY: Does the ontology entail " +
                f"<{query['subject']}> <{query['property']}> <{query['object']}>?"}
    messages = [{"role": "system", "content": INSTRUCTIONS}, user("original")]
    if previous_response is not None:
        messages.extend([{"role": "assistant", "content": previous_response}, user("withdrawal")])
    return messages


def api(host, endpoint, payload=None, timeout=600):
    request = urllib.request.Request(host.rstrip("/") + endpoint,
        data=None if payload is None else json_bytes(payload), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def run_case(model, case, texts, options=None, *, host=None, model_info=None, transport=api, on_record=None):
    """Two real calls only when explicitly invoked; append each record immediately."""
    options = dict(OPTIONS if options is None else options)
    host = host or os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    if not host.startswith(("http://", "https://")):
        host = "http://" + host
    records = []
    previous = None
    for stage in ("original", "withdrawal"):
        record = {"case_id": case["id"], "condition": stage, "parsed": None,
                  "response": None, "request": None, "error": None, "error_kind": None}
        start = time.perf_counter()
        if stage == "withdrawal" and previous is None:
            record.update(error="No usable initial call", error_kind="dependency")
        else:
            messages = build_messages(case, texts, previous)
            record["request"] = {"model": model, "messages": messages, "format": SCHEMA,
                                 "stream": False, "options": options}
            budget = len(json_bytes(messages)) + len(json_bytes(SCHEMA)) + 4096 + options["num_predict"]
            record["conservative_context_budget"] = budget
            if budget > options["num_ctx"]:
                record.update(error="Conservative context budget exceeded; request not sent", error_kind="context")
            else:
                try:
                    response = transport(host, "/api/chat", record["request"])
                    record["response"] = response
                    content = response["message"]["content"]
                    if not isinstance(content, str):
                        raise ValueError("Missing textual response")
                    ps = transport(host, "/api/ps")
                    record["ollama_ps"] = ps
                    loaded = next((m for m in ps.get("models", []) if
                        (model_info is not None and m.get("digest") == model_info["digest"]) or
                        (model_info is None and m.get("name") == model)), {})
                    actual_context = loaded.get("context_length")
                    if actual_context is None or actual_context < budget or response.get("prompt_eval_count", 0) + options["num_predict"] >= options["num_ctx"]:
                        record.update(error="Actual context capacity unverified or insufficient", error_kind="context")
                    else:
                        # Preserve raw text even when it fails schema validation.
                        if stage == "original":
                            previous = content
                        try:
                            if response.get("done_reason") == "length":
                                raise ValueError("Output token limit reached")
                            parsed = json.loads(content)
                            jsonschema.validate(parsed, SCHEMA)
                            record["parsed"] = parsed
                        except (ValueError, jsonschema.ValidationError) as exc:
                            record.update(error=str(exc), error_kind="format")
                except Exception as exc:
                    record.update(error=f"{type(exc).__name__}: {exc}", error_kind="infrastructure")
                    if stage == "original":
                        previous = None
        record["elapsed_s"] = round(time.perf_counter() - start, 3)
        records.append(record)
        if on_record:
            on_record(record)
    return records


def evaluate_case(records, case, graphs, oracle):
    keys = [(r["case_id"], r["condition"]) for r in records]
    if len(keys) != 2 or set(keys) != {(case["id"], s) for s in ("original", "withdrawal")}:
        raise ValueError("Exactly one record per case and condition is required")
    rows = []
    original = graphs[case["snapshots"]["original"]]
    for record in records:
        graph = graphs[case["snapshots"][record["condition"]]]
        row = {"case_id": case["id"], "condition": record["condition"],
               "answer": None, "gold": case["expected"][record["condition"]],
               "answer_correct": False, "citations_current": None, "support_sufficient": None,
               "support_minimal": None, "stale_axioms": [], "absent_axioms": [],
               "success": False, "error_kind": record.get("error_kind")}
        if record.get("parsed") is not None:
            parsed = record["parsed"]
            jsonschema.validate(parsed, SCHEMA)
            triples = {json_triple(t) for t in parsed["axioms"]}
            absent = triples - set(graph)
            row.update(answer=parsed["answer"], answer_correct=parsed["answer"] == row["gold"],
                       citations_current=not absent, absent_axioms=[triple_json(t) for t in sorted(absent)],
                       stale_axioms=[triple_json(t) for t in sorted(absent & set(original))])
            if parsed["answer"] == "entailed":
                sufficient = not absent and oracle.entails(support_graph(triples, case["query"]), case["query"])
                minimal = sufficient and all(not oracle.entails(support_graph(triples - {t}, case["query"]), case["query"]) for t in triples)
                row.update(support_sufficient=sufficient, support_minimal=minimal,
                           success=row["answer_correct"] and sufficient)
            else:
                row["success"] = row["answer_correct"] and not triples
        rows.append(row)
    lookup = {r["condition"]: r for r in rows}
    pair = {"case_id": case["id"], "ontology_id": case["ontology_id"],
            "path_length": case["explicit_path_length"],
            "withdrawal_expected": case["expected"]["withdrawal"],
            "original_success": lookup["original"]["success"],
            "withdrawal_success": lookup["withdrawal"]["success"],
            "pair_success": all(r["success"] for r in rows)}
    return {"pair": pair, "responses": rows}


def evaluate(records, data, graphs, oracle):
    required = {(c["id"], stage) for c in data["cases"] for stage in ("original", "withdrawal")}
    keys = [(r["case_id"], r["condition"]) for r in records]
    if len(keys) != len(set(keys)) or set(keys) != required:
        raise ValueError("Missing, duplicated or unexpected response records")
    evaluated = [evaluate_case([r for r in records if r["case_id"] == c["id"]], c, graphs, oracle) for c in data["cases"]]
    pairs = [e["pair"] for e in evaluated]
    rows = [r for e in evaluated for r in e["responses"]]
    errors = sum(r.get("error_kind") in {"infrastructure", "context", "dependency"} for r in records)
    complete = errors == 0
    def scores(group):
        def rate(field, items):
            return sum(p[field] for p in items) / len(items) if complete and items else None
        changes = [p for p in group if p["withdrawal_expected"] == "not_entailed"]
        preserved = [p for p in group if p["withdrawal_expected"] == "entailed"]
        return {"n_pairs": len(group), "n_withdrawal_pairs": len(changes), "n_preservation_pairs": len(preserved),
                "pair_success_count": sum(p["pair_success"] for p in group),
                "paired_grounded_revision_score": rate("pair_success", group),
                "original_grounded_score": rate("original_success", group),
                "updated_grounded_score": rate("withdrawal_success", group),
                "withdrawal_score": rate("withdrawal_success", changes),
                "preservation_score": rate("withdrawal_success", preserved)}
    originals = [r for r in rows if r["condition"] == "original"]
    summary = {**scores(pairs), "n_ontologies": len({p["ontology_id"] for p in pairs}),
               "n_responses": len(rows), "complete_for_comparison": complete,
               "infrastructure_or_context_errors": errors,
               "format_errors": sum(r.get("error_kind") == "format" for r in records),
               "original_answer_accuracy": sum(r["answer_correct"] for r in originals) / len(originals) if complete else None,
               "correct_original_without_valid_support_count": sum(r["answer_correct"] and not r["success"] for r in originals),
               "withdrawal_stale_citation_count": sum(bool(r["stale_axioms"]) for r in rows if r["condition"] == "withdrawal")}
    return {"summary": summary, "pairs": pairs, "responses": rows,
            "by_path_length": {str(depth): scores([p for p in pairs if p["path_length"] == depth])
                               for depth in sorted({p["path_length"] for p in pairs})}}


def run_benchmark(model, dataset_directory, *, options=None, output_root=None, oracle=None, host=None):
    """Explicit experiment entry point. Never called automatically by the notebook."""
    options = dict(OPTIONS if options is None else options)
    oracle = oracle or PropertyOracle()
    data, graphs, texts, dataset_hash = load_dataset(dataset_directory, oracle=oracle)
    host = host or os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    if not host.startswith(("http://", "https://")):
        host = "http://" + host
    info = next((m for m in api(host, "/api/tags")["models"] if m["name"] == model), None)
    if info is None:
        raise ValueError(f"Model not installed: {model}")
    details = api(host, "/api/show", {"model": model})
    limits = [v for k, v in details.get("model_info", {}).items() if k.endswith(".context_length")]
    if limits and min(limits) < options["num_ctx"]:
        raise ValueError("Model capacity is below the requested window")
    version = api(host, "/api/version")
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", model).strip("._-")[:100] or "model"
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + slug + "_" + uuid.uuid4().hex[:8]
    fingerprint = digest(json_bytes({"dataset": dataset_hash, "options": options,
        "instructions": INSTRUCTIONS, "schema": SCHEMA, "evaluator": digest(Path(__file__).read_bytes())}))
    out = Path(output_root or ROOT / "generated" / "transitive_property") / run_id
    out.mkdir(parents=True, exist_ok=False)
    save_json(out / "manifest.json", {"run_id": run_id, "model": info, "model_details": details,
        "ollama_version": version, "dataset_sha256": dataset_hash, "dataset": data,
        "protocol_fingerprint": fingerprint, "options": options, "instructions": INSTRUCTIONS,
        "schema": SCHEMA, "python": platform.python_version(), "platform": platform.platform(),
        "oracle": {"method": "HermiT inconsistency of ontology + negative query", "timeout_s": getattr(oracle, "timeout_s", None)},
        "java": subprocess.run(["java", "-version"], capture_output=True, text=True).stderr,
        "packages": {p: importlib.metadata.version(p) for p in ("rdflib", "owlready2", "jsonschema")}})
    (out / "snapshots").mkdir()
    for name, content in texts.items():
        (out / "snapshots" / name).write_bytes(content.encode("utf-8"))
    records = []
    def append(record):
        records.append(record)
        with (out / "responses.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    for case in data["cases"]:
        run_case(model, case, texts, options, host=host, model_info=info, on_record=append)
    result = evaluate(records, data, graphs, oracle)
    save_json(out / "evaluation.json", result)
    return out, result
