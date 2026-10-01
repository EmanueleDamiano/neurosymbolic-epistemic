"""Offline contract tests: no Java, model calls or persisted experimental datasets."""
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from rdflib import RDF, OWL, URIRef
import transitive_property_benchmark as benchmark


class TinyOracle:
    """Test double for ONLY named property assertions and transitivity in fixtures."""
    def consistent(self, graph):
        return True

    def entails(self, graph, query):
        prop = URIRef(query['property'])
        triple = tuple(URIRef(query[k]) for k in ('subject', 'property', 'object'))
        return triple in graph or ((prop, RDF.type, OWL.TransitiveProperty) in graph
                                   and bool(benchmark.property_path(graph, query)))


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.oracle = TinyOracle()
        self.config = benchmark.GenerationConfig(n_distractor_triples=2)
        self.obj = benchmark.add_individuals_and_facts(benchmark.generate_onto(self.config), self.config)
        self.graph = benchmark.parse_ontology(self.obj)
        self.candidate = benchmark.select_candidate_queries(self.obj, self.config)[0]
        self.withdrawn = benchmark.withdraw_transitivity(self.graph, self.obj['target_property'])
        self.data, self.texts = benchmark.prepare_cases([self.config], self.oracle)
        self.case = self.data['cases'][0]
        self.graphs = {name: benchmark.rdflib.Graph().parse(data=text, format='turtle')
                       for name, text in self.texts.items()}

    def record(self, stage, answer, triples):
        return {'case_id': self.case['id'], 'condition': stage, 'error_kind': None,
                'parsed': {'answer': answer, 'axioms': [benchmark.triple_json(t) for t in triples]}}

    def test_exact_removal_does_not_mutate_original(self):
        target = URIRef(self.obj['target_property'])
        self.assertEqual(set(self.graph) - set(self.withdrawn), {(target, RDF.type, OWL.TransitiveProperty)})
        self.assertIn((target, RDF.type, OWL.ObjectProperty), self.withdrawn)
        self.assertTrue(benchmark.verify_case(self.graph, self.withdrawn, self.candidate, self.oracle)['original_entailed'])

    def test_wrong_target_and_extra_changes_rejected(self):
        with self.assertRaises(ValueError):
            benchmark.withdraw_transitivity(self.graph, 'https://example.org/other')
        changed = benchmark.copy_graph(self.withdrawn)
        changed.remove(benchmark.json_triple(self.candidate['reference_support'][0]))
        with self.assertRaises(ValueError):
            benchmark.verify_case(self.graph, changed, self.candidate, self.oracle)

    def test_still_entailed_with_shortcut_is_rejected_for_withdrawal(self):
        graph = benchmark.copy_graph(self.graph)
        graph.add(tuple(URIRef(self.candidate['query'][k]) for k in ('subject', 'property', 'object')))
        with self.assertRaisesRegex(ValueError, 'Unexpected entailment'):
            benchmark.verify_case(graph, benchmark.withdraw_transitivity(graph, self.obj['target_property']),
                                  self.candidate, self.oracle)

    def test_generation_is_reproducible_with_exact_query_count(self):
        again = benchmark.add_individuals_and_facts(benchmark.generate_onto(self.config), self.config)
        self.assertEqual(set(self.graph), set(benchmark.parse_ontology(again)))
        self.assertEqual(self.candidate['explicit_path_length'], 3)
        config = benchmark.GenerationConfig(n_queries=4, path_length=4)
        obj = benchmark.add_individuals_and_facts(benchmark.generate_onto(config), config)
        self.assertEqual(len(benchmark.select_candidate_queries(obj, config)), 4)

    def test_unverified_generation_has_bounded_retries(self):
        class Unavailable(TinyOracle):
            def consistent(self, graph):
                raise TimeoutError('reasoner unavailable')
        with self.assertRaisesRegex(RuntimeError, 'Generation attempts exhausted'):
            benchmark.prepare_cases([benchmark.GenerationConfig(max_generation_attempts=2)], Unavailable())
        report = benchmark.validate_onto(self.obj, Unavailable())
        self.assertFalse(report['valid'])
        self.assertIn('TimeoutError', report['errors'][0])

    def test_no_implicit_llm_generator(self):
        with self.assertRaisesRegex(ValueError, 'Supply llm_generator'):
            benchmark.generate_onto(benchmark.GenerationConfig(generator='llm'))

    def test_dataset_round_trip_and_tamper_detection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'dataset'
            benchmark.save_dataset(self.data, self.texts, root)
            loaded, _, _, _ = benchmark.load_dataset(root, oracle=self.oracle)
            self.assertEqual(loaded['cases'], self.data['cases'])
            with self.assertRaises(FileExistsError):
                benchmark.save_dataset(self.data, self.texts, root)
            snapshot = root / self.case['snapshots']['withdrawal']
            snapshot.write_bytes(snapshot.read_bytes() + b'\n# tampered\n')
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                benchmark.load_dataset(root)

    def test_good_pair_and_missing_transitivity_support(self):
        proof = [benchmark.json_triple(t) for t in self.case['reference_support']]
        records = [self.record('original', 'entailed', proof), self.record('withdrawal', 'not_entailed', [])]
        good = benchmark.evaluate(records, self.data, self.graphs, self.oracle)
        self.assertEqual(good['summary']['paired_grounded_revision_score'], 1)
        records[0] = self.record('original', 'entailed', [t for t in proof if t[2] != OWL.TransitiveProperty])
        bad = benchmark.evaluate(records, self.data, self.graphs, self.oracle)
        self.assertEqual(bad['summary']['original_answer_accuracy'], 1)
        self.assertEqual(bad['summary']['original_grounded_score'], 0)

    def test_retired_rule_and_inferred_premise_rejected(self):
        proof = [benchmark.json_triple(t) for t in self.case['reference_support']]
        records = [self.record('original', 'entailed', proof), self.record('withdrawal', 'entailed', proof)]
        evaluated = benchmark.evaluate(records, self.data, self.graphs, self.oracle)
        self.assertEqual(evaluated['summary']['withdrawal_stale_citation_count'], 1)
        triple = tuple(URIRef(self.case['query'][k]) for k in ('subject', 'property', 'object'))
        records[0] = self.record('original', 'entailed', [triple])
        evaluated = benchmark.evaluate(records, self.data, self.graphs, self.oracle)
        self.assertFalse(evaluated['responses'][0]['citations_current'])

    def test_infrastructure_is_not_a_negative_answer(self):
        records = [self.record('original', 'not_entailed', []), self.record('withdrawal', 'not_entailed', [])]
        records[0].update(parsed=None, error_kind='infrastructure')
        summary = benchmark.evaluate(records, self.data, self.graphs, self.oracle)['summary']
        self.assertFalse(summary['complete_for_comparison'])
        self.assertIsNone(summary['withdrawal_score'])
        with self.assertRaises(ValueError):
            benchmark.evaluate(records + [records[0]], self.data, self.graphs, self.oracle)

    def test_controls_have_separate_denominator(self):
        config = benchmark.GenerationConfig(direct_assertion_control=True)
        data, texts = benchmark.prepare_cases([config], self.oracle)
        case = data['cases'][0]
        proof = case['reference_support']
        records = [{'case_id': case['id'], 'condition': stage, 'error_kind': None,
                    'parsed': {'answer': 'entailed', 'axioms': proof}} for stage in ('original', 'withdrawal')]
        graphs = {name: benchmark.rdflib.Graph().parse(data=text, format='turtle') for name, text in texts.items()}
        summary = benchmark.evaluate(records, data, graphs, self.oracle)['summary']
        self.assertEqual(summary['preservation_score'], 1)
        self.assertIsNone(summary['withdrawal_score'])

    def test_messages_contain_raw_previous_answer_but_no_gold_support(self):
        raw = 'not JSON: authentic first output'
        messages = benchmark.build_messages(self.case, self.texts, raw)
        self.assertEqual(messages[2]['content'], raw)
        self.assertIn(self.texts[self.case['snapshots']['withdrawal']], messages[-1]['content'])
        self.assertNotIn('reference_support', messages[-1]['content'])

    def test_runner_preserves_malformed_text_and_logs_progress(self):
        chat_requests, logged = [], []
        def transport(host, endpoint, payload=None):
            if endpoint == '/api/ps':
                return {'models': [{'name': 'fake:1', 'context_length': 32768}]}
            chat_requests.append(payload)
            content = 'malformed' if len(chat_requests) == 1 else json.dumps({'answer': 'not_entailed', 'axioms': []})
            return {'message': {'content': content}, 'prompt_eval_count': 100}
        records = benchmark.run_case('fake:1', self.case, self.texts, transport=transport, on_record=logged.append)
        self.assertEqual(records[0]['error_kind'], 'format')
        self.assertEqual(chat_requests[1]['messages'][2]['content'], 'malformed')
        self.assertEqual(len(logged), 2)

    def test_context_failure_skips_second_call(self):
        calls = []
        def transport(*args):
            calls.append(args)
            raise AssertionError('Should not contact transport')
        records = benchmark.run_case('fake:1', self.case, self.texts,
            options={**benchmark.OPTIONS, 'num_ctx': 10}, transport=transport)
        self.assertEqual(calls, [])
        self.assertEqual([r['error_kind'] for r in records], ['context', 'dependency'])

    def test_unverified_actual_context_is_an_error_not_a_model_failure(self):
        def transport(host, endpoint, payload=None):
            if endpoint == '/api/ps':
                return {'models': [{'name': 'fake:1', 'context_length': None}]}
            return {'message': {'content': '{"answer":"entailed","axioms":[]}'}}
        records = benchmark.run_case('fake:1', self.case, self.texts, transport=transport)
        self.assertEqual([r['error_kind'] for r in records], ['context', 'dependency'])

    def test_run_artifacts_with_mocked_model_and_oracle(self):
        proof = self.case['reference_support']
        chats = []
        def transport(host, endpoint, payload=None):
            if endpoint == '/api/tags':
                return {'models': [{'name': 'fake:1', 'digest': 'test-digest'}]}
            if endpoint == '/api/show':
                return {'model_info': {'fake.context_length': 32768}}
            if endpoint == '/api/version':
                return {'version': 'test'}
            if endpoint == '/api/ps':
                return {'models': [{'digest': 'test-digest', 'context_length': 32768}]}
            chats.append(payload)
            parsed = {'answer': 'entailed', 'axioms': proof} if len(chats) == 1 else {'answer': 'not_entailed', 'axioms': []}
            return {'message': {'content': json.dumps(parsed)}, 'prompt_eval_count': 100}
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / 'dataset'
            benchmark.save_dataset(self.data, self.texts, dataset)
            with patch.object(benchmark, 'api', side_effect=transport), \
                 patch.object(benchmark.subprocess, 'run', return_value=subprocess.CompletedProcess([], 0, '', 'java-test')):
                # run_case has a definition-time transport default; inject it explicitly.
                original_run_case = benchmark.run_case
                def fake_run_case(*args, **kwargs):
                    return original_run_case(*args, **kwargs, transport=transport)
                with patch.object(benchmark, 'run_case', side_effect=fake_run_case):
                    out, result = benchmark.run_benchmark('fake:1', dataset, oracle=self.oracle,
                                                         output_root=Path(directory) / 'runs')
            self.assertIn('_fake_1_', out.name)
            self.assertEqual(result['summary']['paired_grounded_revision_score'], 1)
            self.assertEqual(len((out / 'responses.jsonl').read_text().splitlines()), 2)
            self.assertTrue((out / 'manifest.json').is_file())
            self.assertTrue((out / 'evaluation.json').is_file())
            self.assertEqual(set(p.name for p in (out / 'snapshots').iterdir()), set(self.texts))


class OllamaGeneratorTests(unittest.TestCase):
    def setUp(self):
        self.calls = []
        self.config = benchmark.GenerationConfig(generator='llm', domain='organizations', seed=123)
        self.generated = {
            'generated_onto': '@prefix owl: <http://www.w3.org/2002/07/owl#> . '
                              '<https://test/p> a owl:ObjectProperty, owl:TransitiveProperty .',
            'target_property': 'https://test/p',
        }

    def transport(self, host, endpoint, payload=None, timeout=600):
        self.calls.append((host, endpoint, payload, timeout))
        if endpoint == '/api/tags':
            return {'models': [{'name': 'generator:1', 'digest': 'generator-digest'}]}
        if endpoint == '/api/show':
            return {'model_info': {'test.context_length': 32768}}
        if endpoint == '/api/version':
            return {'version': 'offline-test'}
        if endpoint == '/api/ps':
            return {'models': [{'digest': 'generator-digest', 'context_length': 32768}]}
        if endpoint == '/api/chat':
            return {'message': {'content': json.dumps(self.generated)}, 'prompt_eval_count': 100}
        raise AssertionError(endpoint)

    def generator(self, **kwargs):
        return benchmark.make_ollama_generator('generator:1', transport=self.transport, **kwargs)

    def test_factory_is_lazy_and_callback_records_authentic_generation(self):
        generator = self.generator(host='localhost:11434', timeout_s=90)
        self.assertEqual(self.calls, [])
        obj = benchmark.generate_onto(self.config, llm_generator=generator, system_prompt='Generate ontology only.')
        self.assertEqual(obj['target_property'], self.generated['target_property'])
        provenance = obj['provenance']
        self.assertEqual(provenance['options']['seed'], 123)
        self.assertEqual(provenance['model_info']['digest'], 'generator-digest')
        self.assertEqual(json.loads(provenance['response']['message']['content']), self.generated)
        request = provenance['request']
        self.assertEqual(request['format'], benchmark.GENERATION_SCHEMA)
        self.assertEqual(request['messages'][0]['content'], 'Generate ontology only.')
        self.assertTrue(all(c[0] == 'http://localhost:11434' and c[3] == 90 for c in self.calls))
        self.assertEqual(sum(c[1] == '/api/chat' for c in self.calls), 1)

    def test_callback_integrates_with_pipeline_and_keeps_config_seeds_distinct(self):
        data, _ = benchmark.prepare_cases([self.config], TinyOracle(), llm_generator=self.generator(options={'seed': 999}))
        case = data['cases'][0]
        self.assertEqual(case['requested']['seed'], 123)
        self.assertEqual(case['provenance']['options']['seed'], 999)
        self.assertEqual(case['expected']['withdrawal'], 'not_entailed')

    def test_bad_json_shape_and_bad_turtle_are_rejected(self):
        self.generated = {'generated_onto': 'not valid Turtle', 'target_property': 'https://test/p'}
        with self.assertRaises(Exception):
            benchmark.generate_onto(self.config, llm_generator=self.generator())
        self.generated = {'generated_onto': 'irrelevant'}
        with self.assertRaises(benchmark.jsonschema.ValidationError):
            benchmark.generate_onto(self.config, llm_generator=self.generator())

    def test_missing_declaration_or_unsupported_axioms_are_rejected(self):
        for suffix in ('', ' <https://test/o> owl:imports <https://test/external> .',
                       ' <https://test/C> <https://test/rel> [] .'):
            self.generated['generated_onto'] = (
                '@prefix owl: <http://www.w3.org/2002/07/owl#> . '
                '<https://test/p> a owl:ObjectProperty' +
                (', owl:TransitiveProperty .' if suffix else ' .') + suffix)
            with self.subTest(suffix=suffix), self.assertRaises(ValueError):
                benchmark.generate_onto(self.config, llm_generator=self.generator())

    def test_no_download_or_generation_if_model_is_missing(self):
        def missing(host, endpoint, payload=None, timeout=600):
            self.calls.append(endpoint)
            self.assertEqual(endpoint, '/api/tags')
            return {'models': []}
        generator = benchmark.make_ollama_generator('missing:1', transport=missing)
        with self.assertRaisesRegex(ValueError, 'not installed'):
            benchmark.generate_onto(self.config, llm_generator=generator)
        self.assertEqual(self.calls, ['/api/tags'])

    def test_output_truncation_is_rejected(self):
        def truncated(host, endpoint, payload=None, timeout=600):
            response = self.transport(host, endpoint, payload, timeout)
            if endpoint == '/api/chat':
                response['done_reason'] = 'length'
            return response
        generator = benchmark.make_ollama_generator('generator:1', transport=truncated)
        with self.assertRaisesRegex(ValueError, 'output token limit'):
            benchmark.generate_onto(self.config, llm_generator=generator)

    def test_context_budget_prevents_network_access(self):
        with self.assertRaisesRegex(ValueError, 'context budget'):
            benchmark.generate_onto(self.config, llm_generator=self.generator(options={'num_ctx': 10}))
        self.assertEqual(self.calls, [])

    def test_raw_response_is_preserved_even_if_validation_fails(self):
        artifacts = []
        self.generated = {'generated_onto': 'not Turtle', 'target_property': 'https://test/p'}
        generator = benchmark.make_ollama_generator('generator:1', transport=self.transport,
                                                    think=False, on_response=artifacts.append)
        with self.assertRaises(Exception):
            benchmark.generate_onto(self.config, llm_generator=generator)
        self.assertEqual(len(artifacts), 1)
        self.assertIs(artifacts[0]['request']['think'], False)
        self.assertEqual(json.loads(artifacts[0]['response']['message']['content']), self.generated)


class BatchTests(unittest.TestCase):
    def generator(self, *, prompt, system_prompt, config):
        obj = benchmark.generate_onto(benchmark.GenerationConfig(seed=config['seed']))
        return {k: obj[k] for k in ('generated_onto', 'target_property', 'provenance')}

    def test_separate_frozen_datasets_and_progressive_index(self):
        configs = [benchmark.GenerationConfig(generator='llm', domain=d, seed=i)
                   for i, d in enumerate(('food', 'school'))]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'batch'
            _, manifest = benchmark.generate_ontology_batch(configs, 'fake:1', root,
                oracle=TinyOracle(), llm_generator=self.generator, on_progress=None)
            self.assertEqual(manifest['status'], 'complete')
            self.assertEqual(len(manifest['items']), 2)
            for item in manifest['items']:
                folder = root / item['directory']
                self.assertTrue((folder / 'base.ttl').is_file())
                self.assertTrue((folder / 'generation.json').is_file())
                data, _, _, _ = benchmark.load_dataset(folder / 'dataset', oracle=TinyOracle())
                self.assertEqual(data['cases'][0]['requested']['domain'], item['domain'])
                self.assertEqual(item['path_length'], 3)
            with self.assertRaises(FileExistsError):
                benchmark.generate_ontology_batch(configs, 'fake:1', root,
                    oracle=TinyOracle(), llm_generator=self.generator, on_progress=None)

    def test_failed_batch_keeps_accepted_items_and_records_retry_feedback(self):
        prompts = []
        def generator(**kwargs):
            if kwargs['config']['domain'] == 'school':
                prompts.append(kwargs['system_prompt'])
                raise ValueError('invalid ontology output')
            return self.generator(**kwargs)
        configs = [benchmark.GenerationConfig(generator='llm', domain=d, seed=i,
                    max_generation_attempts=2) for i, d in enumerate(('food', 'school'))]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'batch'
            with self.assertRaisesRegex(RuntimeError, 'attempts exhausted'):
                benchmark.generate_ontology_batch(configs, 'fake:1', root,
                    oracle=TinyOracle(), llm_generator=generator, on_progress=None)
            manifest = json.loads((root / 'batch.json').read_text())
            self.assertEqual(manifest['status'], 'failed')
            self.assertEqual(len(manifest['items']), 1)
            self.assertIn('invalid ontology output', prompts[1])
            self.assertTrue((root / '01_food' / 'dataset' / 'dataset.json').is_file())


class OracleContractTests(unittest.TestCase):
    def test_refutation_and_cache(self):
        query = {'subject': 'https://test/a', 'property': 'https://test/p', 'object': 'https://test/b'}
        graph = benchmark.support_graph([], query)
        positive = subprocess.CompletedProcess([], 0, '{"consistent": true}', '')
        contradiction = subprocess.CompletedProcess([], 0, '{"consistent": false}', '')
        with patch.object(benchmark.subprocess, 'run', side_effect=[positive, contradiction]) as run:
            oracle = benchmark.PropertyOracle()
            self.assertTrue(oracle.entails(graph, query))
            self.assertTrue(oracle.entails(graph, query))
            self.assertEqual(run.call_count, 2)

    def test_reasoner_failure_propagates(self):
        with patch.object(benchmark.subprocess, 'run', side_effect=subprocess.TimeoutExpired('java', 1)):
            with self.assertRaises(subprocess.TimeoutExpired):
                benchmark.PropertyOracle(timeout_s=1).consistent(benchmark.rdflib.Graph())


if __name__ == '__main__':
    unittest.main()
