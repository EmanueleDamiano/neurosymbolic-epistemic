# LUBM subclass grounding all eligible queries

Frozen dataset created without LLM calls. From 76 positive subclass queries in the
existing source JSON, 55 have nonempty explicit named-subclass paths and 52 pass
the full HermiT checks for original entailment and non-entailment after a single
critical asserted edge is removed. All excluded rows and reasons are in dataset.json.

Depth distribution: 33 one-edge proofs, 15 two-edge proofs, 4 three-edge proofs.
There are no quotas by depth and no distinct-subject restriction. Depth describes
the reference explicit path, not the size of every possible OWL proof. The original
source labels are retained only for provenance. All 319 original triples are sent
to the model; every withdrawal preserves the other 318 triples.

The dataset manifest records requested size, seed, eligible and selected counts,
reference proofs, removals and snapshot hashes. Do not overwrite frozen datasets.
To use fewer cases, build a new dataset with --num-queries N and an optional --seed.
The selection is uniform without replacement from the verified pool, then sorted
by depth for presentation. Multiple queries may share classes and proof edges.

The six-case v1 dataset is preserved unchanged. Results from runs with different
datasets or code fingerprints must not be combined as if they used the same protocol.
