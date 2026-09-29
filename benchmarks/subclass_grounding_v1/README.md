# LUBM subclass grounding pilot v1

Six frozen query pairs from the existing `owlapy_generated_qa.json`. Each original
query is entailed by the complete repository ontology `univ-bench-inst.ttl`.
The paired withdrawal removes exactly one asserted subclass axiom, and HermiT
confirms that the query is no longer entailed. Every other triple is preserved.

Selection is independent of LLM results: 76 positive subclass queries in the source,
55 with a nonempty path of explicit named-class subclass inclusions of length 1–3.
The fixed random seed is 20260929; selection allocates the scarce depth-3 cases first,
then depth 2, then depth 1, with distinct subject classes and two cases per depth.
Depth measures the reference explicit path, not the minimum size of every possible OWL proof.
The original dataset's `direct`/`inferred` labels are retained for provenance but do
not determine the asserted/inferred distinction or the strata of this benchmark.

The original snapshot contains all 319 source triples. Every withdrawal contains 318.
The evaluator checks graph isomorphism against the source and declared removals,
as well as entailment and consistency with HermiT before model calls.
Named-class proof paths are used to propose examples, never to establish OWL non-entailment.

`dataset.json` records the source row, query, expected labels, removed axiom,
reference support, snapshot filenames and hashes. Reference support is never
sent to the model and is not used as an exact-match target. Alternative valid
supports are accepted. The full original and withdrawn snapshots are sent verbatim.

Do not edit the frozen files after collecting results. Create a new dataset version
for new queries or a changed selection policy. The pilot has repeated support edges
and is not a sample of independent problems or a representative OWL benchmark.
All withdrawals change entailment; selective stability controls are outside v1.
