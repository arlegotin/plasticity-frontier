import hashlib
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np

from prepared_learning import natural_inputs as natural


class NaturalInputTests(unittest.TestCase):
    def test_duplicates_do_not_cross_pools(self):
        X = np.repeat(np.arange(10.)[:, None], [1,2,3,1,2,3,1,2,3,1], axis=0)
        pools = natural.split_duplicate_groups(X, 12)
        self.assertEqual([len(np.unique(pools[k])) for k in ('development','validation','protected')], [6,2,2])
        self.assertEqual(sum(map(len, pools.values())), len(X))
        for a, b in [('development','validation'),('development','protected'),('validation','protected')]:
            self.assertFalse(set(pools[a][:,0]) & set(pools[b][:,0]))
        for k, v in pools.items(): np.testing.assert_array_equal(v, natural.split_duplicate_groups(X, 12)[k])

    def test_transform_has_no_heldout_access(self):
        development = np.array([[0.,0.],[1.,3.],[2.,1.],[3.,2.]])
        first = natural.fit_development_transform(development, True)
        heldout = np.array([[100., 400.]])
        heldout *= 100
        self.assertEqual(first, natural.fit_development_transform(development, True))
        self.assertEqual(first['rank'], 2)
        with self.assertRaises(ValueError): natural.fit_development_transform(np.ones((5,2)), True)

    def test_asset_hash_license_and_columns(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            columns = natural.FEATURE_COLUMNS + ['quality']
            rows = ';'.join('"'+x+'"' for x in columns)+'\n'
            rows += (';'.join(str(x) for x in range(12))+'\n') * 4898
            table = tmp/'wine.csv'; table.write_text(rows)
            license_file = tmp/'license'; license_file.write_text('fixture license')
            manifest = {'dataset_url':natural.DATASET_URL,'license_url':natural.LICENSE_URL,
                        'source_url':'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv',
                        'operator_attested_official_origin':True,'license_id':'CC-BY-4.0',
                        'file_path':str(table),'file_sha256':hashlib.sha256(table.read_bytes()).hexdigest(),
                        'license_path':str(license_file),'license_sha256':hashlib.sha256(license_file.read_bytes()).hexdigest(),
                        'columns':natural.FEATURE_COLUMNS,'expected_rows':4898}
            np.testing.assert_array_equal(natural.load_natural_asset(manifest), np.tile(np.arange(11.), (4898, 1)))
            for change in ({'file_sha256':'0'*64}, {'license_sha256':'0'*64}, {'operator_attested_official_origin':False}, {'source_url':'https://example.com/wine.csv'}, {'license_id':'unknown'}, {'columns':natural.FEATURE_COLUMNS[:-1]+['quality']}):
                with self.assertRaises(ValueError): natural.load_natural_asset(dict(manifest, **change))

    def test_context_query_disjoint_and_population_label(self):
        pool = np.arange(30.).reshape(10,3)
        task = natural.sample_conditional_task(pool, 4, 5, .2, 7)
        self.assertFalse(set(task['context_indices']) & set(task['query_indices']))
        self.assertIn('conditional', task['inference'])
        self.assertIn('semi-synthetic', task['inference'])
        np.testing.assert_allclose(task['context_y'], pool[task['context_indices']] @ task['theta'] + task['context_noise'])

    def test_optional_asset_failure_keeps_core(self):
        result = natural.run_optional({'status':'run','asset':{}}, lambda: {'core':'complete'})
        self.assertEqual(result['core']['core'], 'complete')
        self.assertEqual(result['natural']['status'], 'not_run')
        self.assertTrue(result['natural']['reason'])
        bad = natural.optional_result({'status':'run','question':'x','amendment':'x','asset':{'dataset_url':natural.DATASET_URL,'license_url':natural.LICENSE_URL,'source_url':42}})
        self.assertEqual(bad['status'], 'not_run')

    def test_robustness_cli_records_omission_and_executes_core(self):
        import contextlib, io
        from prepared_learning.cli import main
        from prepared_learning.robustness import make_manifest
        from prepared_learning.specs import load_config
        from dataclasses import replace
        base = load_config(Path('configs/exploration/learned_prior.json'))
        base = replace(base, workload={**base.workload, 'population_seed':73})
        manifest = make_manifest(base, ['no_floor'])
        manifest['natural_inputs'] = {'status':'run', 'asset':{}}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'manifest.json'; path.write_text(json.dumps(manifest))
            output = Path(tmp)/'out'
            with contextlib.redirect_stdout(io.StringIO()):
                code = main(['robustness','--manifest',str(path),'--output',str(output)])
            self.assertEqual(code, 0)
            self.assertEqual(json.loads((output/'robustness.json').read_text())['status'], 'completed')
            self.assertEqual(json.loads((output/'natural_inputs.json').read_text())['status'], 'not_run')

    def test_operator_attestation_alone_never_runs_natural_science(self):
        from unittest.mock import patch
        rng = np.random.default_rng(9)
        fixture = rng.normal(size=(30,11))
        config = {'status':'run', 'question':'input law', 'amendment':'declared', 'asset':{'file_sha256':'a'*64,'license_sha256':'b'*64,'source_url':natural.DATASET_URL}, 'seed':1, 'task_seed':2, 'context_size':3, 'query_size':3, 'sigma2':.1}
        with patch.object(natural, 'load_natural_asset', return_value=fixture):
            result = natural.optional_result(config)
        self.assertEqual(result['status'], 'not_run')
        self.assertIn('receipt', result['reason'])
