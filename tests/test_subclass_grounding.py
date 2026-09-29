"""Small semantic checks for the benchmark evaluator; requires Java/HermiT, no LLM."""
import unittest
from owlready2 import OwlReadyInconsistentOntologyError
from rdflib import RDF, RDFS, OWL, Namespace
import rdflib
from subclass_grounding_benchmark import Oracle, copy_graph, triple_json, score_response, evaluate


class GroundingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.oracle = Oracle()
        cls.ns = Namespace("http://test.example/#")

    def setUp(self):
        u = self.ns
        self.edges = [(u.A, RDFS.subClassOf, u.B), (u.B, RDFS.subClassOf, u.C),
                      (u.A, RDFS.subClassOf, u.D), (u.D, RDFS.subClassOf, u.C)]
        original = rdflib.Graph()
        for c in (u.A, u.B, u.C, u.D):
            original.add((c, RDF.type, OWL.Class))
        for t in self.edges:
            original.add(t)
        withdrawn = copy_graph(original)
        withdrawn.remove(self.edges[1])
        withdrawn.remove(self.edges[3])
        self.graphs = {"original": original, "withdrawal": withdrawn}
        self.case = {"id": "test", "query": [str(u.A), str(u.C)], "explicit_proof_depth": 2,
                     "snapshots": {"original": "original", "withdrawal": "withdrawal"},
                     "expected": {"original": "entailed", "withdrawal": "not_entailed"}}

    def record(self, edges, answer="entailed", condition="original"):
        return {"case_id": "test", "condition": condition, "error_kind": None,
                "parsed": {"answer": answer, "axioms": [triple_json(t) for t in edges]}}

    def score(self, record):
        return score_response(record, self.case, self.graphs, self.oracle)

    def test_alternative_proofs_both_pass(self):
        self.assertTrue(self.score(self.record(self.edges[:2]))["success"])
        self.assertTrue(self.score(self.record(self.edges[2:]))["success"])

    def test_present_but_insufficient_support_fails(self):
        row = self.score(self.record(self.edges[:1]))
        self.assertTrue(row["answer_correct"])
        self.assertTrue(row["citations_current"])
        self.assertFalse(row["success"])

    def test_inferred_but_not_asserted_citation_fails(self):
        row = self.score(self.record([(self.ns.A, RDFS.subClassOf, self.ns.C)]))
        self.assertFalse(row["citations_current"])
        self.assertFalse(row["success"])

    def test_retired_axiom_is_distinguished(self):
        row = self.score(self.record(self.edges[:2], condition="withdrawal"))
        self.assertEqual(row["stale_axioms"], [triple_json(self.edges[1])])
        self.assertFalse(row["success"])

    def test_empty_negative_is_correct_only_when_oracle_agrees(self):
        self.assertTrue(self.score(self.record([], "not_entailed", "withdrawal"))["success"])
        self.assertFalse(self.score(self.record([], "not_entailed", "original"))["success"])

    def test_extra_evidence_is_valid_but_not_minimal(self):
        row = self.score(self.record(self.edges))
        self.assertTrue(row["success"])
        self.assertFalse(row["support_minimal"])

    def test_pair_score_not_conditioned_on_correct_initial_answer(self):
        records = [self.record([], "not_entailed"), self.record([], "not_entailed", "withdrawal")]
        result = evaluate(records, {"cases": [self.case]}, self.graphs, self.oracle)
        self.assertEqual(result["summary"]["paired_grounded_revision_score"], 0)
        self.assertEqual(result["summary"]["withdrawal_score"], 1)

    def test_infrastructure_failure_makes_score_unavailable(self):
        records = [self.record([], "not_entailed"), self.record([], "not_entailed", "withdrawal")]
        records[0].update(parsed=None, error_kind="infrastructure")
        result = evaluate(records, {"cases": [self.case]}, self.graphs, self.oracle)
        self.assertIsNone(result["summary"]["paired_grounded_revision_score"])
        self.assertFalse(result["summary"]["complete_for_comparison"])

    def test_inconsistent_graph_raises(self):
        graph = copy_graph(self.graphs["original"])
        graph.add((self.ns.A, OWL.disjointWith, self.ns.C))
        graph.add((self.ns.x, RDF.type, self.ns.A))
        with self.assertRaises(OwlReadyInconsistentOntologyError):
            self.oracle.entails(graph, self.case["query"])


if __name__ == "__main__":
    unittest.main()
