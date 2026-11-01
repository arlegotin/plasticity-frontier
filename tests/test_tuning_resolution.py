"""Selection-policy fixtures; no campaign observations or evaluator outcomes."""
import math
import statistics
import unittest
from types import SimpleNamespace

from prepared_learning.records import SelectionResult
from prepared_learning.selection import select_method


class AnalyticCandidates:
    """Supply exact task-loss curves to the real adaptive selector."""
    def __init__(self, method_id='B02', loss=None, ridge=False):
        self.method_id = method_id
        self.loss = loss or (lambda gamma: [(1 + gamma)**2, (2 + gamma)**2])
        self.ridge = ridge

    def fit_meta(self, train, validation, config):
        rows = []
        settings = config['fitting_candidates']
        values = settings['ridge_multipliers'] if self.ridge else [None]
        for gamma in settings['gammas']:
            for value in values:
                losses = self.loss(value if self.ridge else gamma)
                name = f'{self.method_id}:{gamma}:{value}'
                rows.append({'candidate_id': name, 'method_id': self.method_id,
                    'status': 'completed', 'gamma': gamma, 'tuning_kind': 'penalty' if self.ridge else 'default',
                    'tuning_value': value, 'rank': None, 'ridge_scale': 1.,
                    'asset': {'fit_metadata': {'parent_hash': self.method_id}},
                    'asset_hash': f'{self.method_id}:{gamma}', 'validation_task_losses': losses,
                    'validation_loss': statistics.fmean(losses), 'elapsed_ns': 1})
        selected = min(rows, key=lambda row: row['validation_loss'])
        return SelectionResult(selected['candidate_id'], tuple(rows), None)


class GroupedRidgeCandidates:
    """Multiple fixed representations inside one method's penalty search."""
    method_id = 'B09:ridge'

    def __init__(self, control=False, limits=False):
        self.control = control
        self.limits = limits

    def fit_meta(self, train, validation, config):
        rows = []
        groups = [(asset, rank, 'ordinary') for asset in ('A', 'B') for rank in (1, 2)]
        if self.control:
            groups.append(('A', 1, 'alternate'))
        for asset, rank, variant in groups:
            for gamma in config['fitting_candidates']['gammas']:
                penalties = list(config['fitting_candidates']['ridge_multipliers'])
                if self.limits:
                    penalties.append(None)  # Explicit zero-estimator limit.
                for penalty in penalties:
                    if penalty is None:
                        level = 0. if (asset, rank) == ('B', 2) else 1000.
                    elif self.limits and (asset, rank) == ('A', 1):
                        level = 1.+.001*penalty**2  # Zero/pseudoinverse minimum.
                    elif self.control or self.limits or (asset, rank) == ('A', 1):
                        level = 1.+.1*rank+(.5 if asset == 'B' else 0)+(.5 if variant == 'alternate' else 0)+.001*(penalty-1)**2
                    elif (asset, rank) == ('A', 2):
                        level = 20.+10*penalty  # Losing lower boundary.
                    elif (asset, rank) == ('B', 1):
                        level = 200.+100/(penalty+1)  # Losing upper boundary.
                    else:
                        level = 300.+10*(penalty-1)**2  # Losing interior, both gaps unresolved.
                    options = {'rank': rank, 'variant': variant}
                    options.update({'zero_estimator': True} if penalty is None else {'penalty': penalty})
                    rows.append({'candidate_id': f'{asset}:{rank}:{variant}:{gamma}:{penalty}',
                        'method_id': self.method_id, 'status': 'completed', 'gamma': gamma,
                        'tuning_kind': 'zero' if penalty is None else 'penalty', 'tuning_value': penalty,
                        'rank': rank, 'ridge_scale': 1., 'options': options,
                        'asset': {'fit_metadata': {'parent_hash': asset}}, 'asset_hash': f'{asset}:{gamma}',
                        'validation_task_losses': [level, level+4], 'validation_loss': level+2, 'elapsed_ns': 1})
        selected = min(rows, key=lambda row: row['validation_loss'])
        return SelectionResult(selected['candidate_id'], tuple(rows), None)


class TuningResolutionTests(unittest.TestCase):
    def run_selection(self, methods=None, validation_count=2, **settings):
        def task(key):
            return SimpleNamespace(source_key=key,
                A=SimpleNamespace(task_key=key+'a'), B=SimpleNamespace(task_key=key+'b'))
        train = [task('train')]
        validation = [task(f'val{i}') for i in range(validation_count)]
        config = {'budget': {'depth': 2, 'initialization_norm_cap': 1., 'trainable_parameter_cap': 2, 'learning_rate': 1.},
            'preparation': {'target_context_size': 1, 'input_dimension': 1, 'input_law': 'gaussian_isotropic',
                'loss_normalization': 'squared_error_over_2n', 'task_second_moment_trace': 1.},
            'fitting_candidates': {'gammas': [0., 1.], 'max_refinement_rounds': 5, **settings},
            'source_access_manifest': {'training_keys': ['train'], 'validation_keys': [t.source_key for t in validation]}}
        return select_method(methods or [AnalyticCandidates()], train, validation, config)

    def test_smooth_monotone_endpoint_resolves_after_inward_refinement(self):
        result = self.run_selection()
        selected = next(r for r in result.candidate_records if r['candidate_id'] == result.selected_id)
        self.assertEqual(selected['tuning_resolution'], 'resolved')
        self.assertEqual(selected['gamma'], 0.)
        self.assertEqual(selected['refinement_rounds'], 2)
        self.assertEqual({r['gamma'] for r in result.candidate_records}, {0., .25, .5, 1.})
        check = selected['tuning_checks'][0]
        self.assertEqual(check['decision'], 'objective_resolved')
        self.assertEqual(check['inward_probe'], 'interior_candidate_present')
        self.assertAlmostEqual(check['absolute_mean_gap'], .8125)
        self.assertAlmostEqual(check['objective_tolerance'], 1.5)
        self.assertGreater(check['absolute_mean_gap'], check['paired_difference_se'])
        self.assertEqual(selected['tuning_resolution_policy'], 'min_marginal_loss_se_v1')

    def test_all_families_including_losing_groups_receive_same_checks(self):
        methods = [AnalyticCandidates('B01', lambda g: [0., 0.])]
        methods += [AnalyticCandidates(name) for name in ('B02', 'B03', 'B04', 'B11:F', 'B11:Z')]
        result = self.run_selection(methods)
        self.assertTrue(result.selected_id.startswith('B01:'))
        final = [r for r in result.candidate_records if r['refinement_iteration'] == 2]
        self.assertEqual({r['method_id'] for r in final if r.get('tuning_checks')}, {m.method_id for m in methods})
        for method in methods:
            rows = [r for r in final if r['method_id'] == method.method_id]
            self.assertEqual({r['gamma'] for r in rows}, {0., .25, .5, 1.})
            self.assertTrue(all(r['tuning_resolution'] == 'resolved' for r in rows))

    def test_interior_minimum_and_ridge_use_the_same_objective_scale(self):
        for ridge in (False, True):
            with self.subTest(ridge=ridge):
                # A common task component changes marginal precision, not ranking.
                loss = lambda x: [(x-.5)**2+1, (x-.5)**2+5]
                method = AnalyticCandidates('B08:ridge' if ridge else 'B02', loss, ridge)
                result = self.run_selection([method], ridge_multipliers=[0., .25, .5, 1.])
                selected = next(r for r in result.candidate_records if r['candidate_id'] == result.selected_id)
                self.assertEqual(selected['tuning_resolution'], 'resolved')
                dimension = 'ridge_penalty' if ridge else 'shrinkage'
                checks = [c for r in result.candidate_records if r['refinement_iteration'] == 1
                    for c in r.get('tuning_checks', []) if c['dimension'] == dimension]
                self.assertEqual(len(checks), 6 if ridge else 2)
                for check in checks:
                    self.assertEqual(check['decision'], 'objective_resolved')
                    self.assertAlmostEqual(check['objective_tolerance'], 2.)
                    self.assertAlmostEqual(check['paired_difference_se'], 0.)

    def test_finite_ridge_boundary_extends_even_when_objective_gap_is_resolved(self):
        method = AnalyticCandidates('B08:ridge', lambda x: [(1+x)**2+1, (1+x)**2+5], ridge=True)
        result = self.run_selection([method], ridge_multipliers=[.5, 1.], max_refinement_rounds=1)
        self.assertIn(.05, {r['tuning_value'] for r in result.candidate_records})
        first = next(r for r in result.candidate_records if r.get('ridge_boundary_check'))
        self.assertEqual(first['ridge_boundary_check']['decision'], 'extend_finite_boundary')
        check = next(c for c in first['tuning_checks'] if c['dimension'] == 'ridge_penalty')
        self.assertEqual(check['decision'], 'objective_resolved')
        self.assertEqual(result.candidate_records[-1]['tuning_resolution'], 'unresolved')

    def test_ridge_checks_losing_assets_and_ranks_and_unions_grid_requests(self):
        result = self.run_selection([GroupedRidgeCandidates()], gammas=[0., .5, 1.],
            ridge_multipliers=[.1, 1., 10.], max_refinement_rounds=1)
        self.assertEqual(result.candidate_records[-1]['tuning_resolution'], 'unresolved')
        first = [r for r in result.candidate_records if r['refinement_iteration'] == 0]
        checked = [r for r in first if 'ridge_boundary_check' in r]
        self.assertEqual({(r['asset_hash'], r['rank']) for r in checked},
            {(f'{asset}:{gamma}', rank) for asset in ('A', 'B') for gamma in (0., .5, 1.) for rank in (1, 2)})
        lower = next(r for r in checked if r['asset_hash'] == 'A:0.0' and r['rank'] == 2)
        check = next(c for c in lower['tuning_checks'] if c['dimension'] == 'ridge_penalty')
        self.assertAlmostEqual(check['absolute_mean_gap'], 9.)
        self.assertAlmostEqual(check['objective_tolerance'], 2.)
        self.assertEqual(lower['ridge_boundary_check']['extension_multipliers'], (.01,))
        final = [r for r in result.candidate_records if r['refinement_iteration'] == 1]
        for asset in ('A', 'B'):
            for rank in (1, 2):
                values = {r['tuning_value'] for r in final if r['asset_hash'] == f'{asset}:0.0' and r['rank'] == rank}
                self.assertTrue({.01, 100., math.sqrt(.1), math.sqrt(10.)} <= values)

    def test_fully_checked_ridge_assets_ranks_and_fixed_options_can_resolve(self):
        result = self.run_selection([GroupedRidgeCandidates(control=True)], gammas=[0., .5, 1.],
            ridge_multipliers=[.1, 1., 10.], max_refinement_rounds=0)
        self.assertEqual(result.candidate_records[-1]['tuning_resolution'], 'resolved')
        checked = [r for r in result.candidate_records if 'ridge_boundary_check' in r]
        self.assertEqual(len(checked), 15)  # Four asset/rank groups plus one fixed-option group, each at three gammas.
        by_id = {r['candidate_id']: r for r in result.candidate_records}
        for row in checked:
            checks = [c for c in row.get('tuning_checks', []) if c['dimension'] == 'ridge_penalty']
            self.assertEqual(len(checks), 2)
            for check in checks:
                self.assertEqual(check['decision'], 'objective_resolved')
                for side in ('left', 'right'):
                    peer = by_id[check[side+'_candidate_id']]
                    self.assertEqual((peer['asset_hash'], peer['rank'], peer['options']['variant']),
                        (row['asset_hash'], row['rank'], row['options']['variant']))

    def test_grouped_ridge_preserves_zero_penalty_and_zero_estimator_limits(self):
        result = self.run_selection([GroupedRidgeCandidates(limits=True)], gammas=[0., .5, 1.],
            ridge_multipliers=[0., .1, 1., 10.], max_refinement_rounds=0)
        self.assertEqual(result.candidate_records[-1]['tuning_resolution'], 'resolved')
        checked = [r for r in result.candidate_records if 'ridge_boundary_check' in r]
        self.assertEqual(len(checked), 12)
        for row in checked:
            self.assertEqual(row['ridge_boundary_check']['extension_multipliers'], ())
            if row['asset_hash'].startswith('A:') and row['rank'] == 1:
                self.assertEqual(row['tuning_kind'], 'penalty')
                self.assertEqual(row['tuning_value'], 0.)
                self.assertTrue(any(c['dimension'] == 'ridge_penalty' for c in row['tuning_checks']))
            if row['asset_hash'].startswith('B:') and row['rank'] == 2:
                self.assertEqual(row['tuning_kind'], 'zero')
                self.assertEqual(row['ridge_boundary_check']['decision'], 'zero_estimator_limit_selected')

    def test_public_policy_is_declared_and_cannot_be_overridden(self):
        from prepared_learning.records import BudgetSpec, PreparationSpec, RunConfig, SamplePlan
        from prepared_learning.selection import public_method_config
        config = RunConfig(1, {}, BudgetSpec(2, 1., 2, 1.),
            PreparationSpec(1, 1, 'gaussian_isotropic', 'squared_error_over_2n', 1.), {}, {}, (), {},
            SamplePlan('deterministic_frontier', None, None, None, None, None), {})
        public = public_method_config(config)
        self.assertEqual(public['tuning_resolution_policy']['version'], 'min_marginal_loss_se_v1')
        public['tuning_resolution_policy']['version'] = 'resolve_at_cap'
        with self.assertRaisesRegex(ValueError, 'unsupported tuning resolution policy'):
            select_method([], [], [], public)

    def test_real_b02_producer_retains_task_uncertainty_and_local_checks(self):
        import numpy as np
        from prepared_learning.methods import make_method
        from prepared_learning.records import Context, SourceTask
        def task(key, size):
            return SourceTask(Context(np.eye(2), np.array([size, 0.]), key+'a'),
                Context(np.eye(2), np.array([size, 0.]), key+'b'), key)
        config = {'budget': {'depth': 2, 'initialization_norm_cap': 1., 'trainable_parameter_cap': 6, 'learning_rate': 1.},
            'preparation': {'target_context_size': 2, 'input_dimension': 2, 'input_law': 'gaussian_isotropic',
                'loss_normalization': 'squared_error_over_2n', 'task_second_moment_trace': 1.},
            'fitting_candidates': {'gammas': [0., 1.], 'ranks': [1], 'max_refinement_rounds': 5},
            'information_policy': {'noise_access': 'known', 'permitted_noise_variance': 0.},
            'source_access_manifest': {'training_keys': ['train'], 'validation_keys': ['a', 'b']}}
        result = select_method([make_method('B02', {})], [task('train', 1.)], [task('a', 1.), task('b', 2.)], config)
        self.assertTrue(all(r['tuning_resolution'] == 'resolved' for r in result.candidate_records))
        self.assertIn(.5, {r['gamma'] for r in result.candidate_records})
        by_id = {r['candidate_id']: r for r in result.candidate_records}
        for row in by_id.values():
            self.assertAlmostEqual(row['validation_loss_se'], statistics.stdev(row['validation_task_losses'])/math.sqrt(2))
            for check in row.get('tuning_checks', []):
                left, right = (by_id[check[side+'_candidate_id']] for side in ('left', 'right'))
                self.assertAlmostEqual(check['objective_tolerance'], min(left['validation_loss_se'], right['validation_loss_se']))
                self.assertEqual(left['refinement_iteration'], row['refinement_iteration'])
                self.assertEqual(right['refinement_iteration'], row['refinement_iteration'])

    def test_missing_nonfinite_or_zero_uncertainty_cannot_hide_positive_gaps(self):
        for losses, count in ((lambda g: [(1+g)**2], 1),
                              (lambda g: [(1+g)**2], 2),
                              (lambda g: [(1+g)**2]*2, 2),
                              (lambda g: [(1+g)**2, math.nan], 2)):
            with self.subTest(count=count, losses=losses):
                result = self.run_selection([AnalyticCandidates(loss=losses)], validation_count=count, max_refinement_rounds=2)
                self.assertTrue(all(r['tuning_resolution'] != 'resolved' for r in result.candidate_records))

    def test_missing_task_losses_remain_uncertainty_unavailable(self):
        class MissingLosses(AnalyticCandidates):
            def fit_meta(self, *args):
                result = super().fit_meta(*args)
                rows = [dict(row) for row in result.candidate_records]
                for row in rows:
                    row.pop('validation_task_losses')
                return SelectionResult(result.selected_id, tuple(rows), result.asset)
        result = self.run_selection([MissingLosses()])
        self.assertTrue(all(r['tuning_resolution'] == 'uncertainty_unavailable' for r in result.candidate_records))
        self.assertTrue(all(r['validation_loss_se'] is None for r in result.candidate_records))

    def test_initial_interior_grid_supplies_probe_but_does_not_exempt_endpoint(self):
        for neighbor, expected in ((.25, 'resolved'), (.5, 'unresolved')):
            with self.subTest(neighbor=neighbor):
                result = self.run_selection(gammas=[0., neighbor, 1.], max_refinement_rounds=0)
                selected = next(r for r in result.candidate_records if r['candidate_id'] == result.selected_id)
                self.assertEqual(selected['tuning_resolution'], expected)
                check = selected['tuning_checks'][0]
                probe = next(r for r in result.candidate_records if r['candidate_id'] == check['inward_probe_candidate_id'])
                self.assertEqual(probe['gamma'], neighbor)

    def test_unrepresentable_mesh_progress_does_not_authorize_resolution(self):
        method = AnalyticCandidates(loss=lambda g: [1.+(g>0), 2.+(g>0)])
        result = self.run_selection([method], gammas=[0., math.nextafter(0., 1.), 1.], max_refinement_rounds=1)
        selected = next(r for r in result.candidate_records if r['candidate_id'] == result.selected_id)
        self.assertEqual(selected['tuning_resolution'], 'unresolved')
        check = selected['tuning_checks'][0]
        self.assertEqual(check['decision'], 'refine_objective_gap')
        self.assertFalse(check['representable_progress'])

    def test_common_task_component_can_change_stopping_but_not_paired_ranking(self):
        plain = self.run_selection([AnalyticCandidates(loss=lambda g: [g*g, g*g])], max_refinement_rounds=2)
        shifted = self.run_selection([AnalyticCandidates(loss=lambda g: [g*g+1, g*g+5])], max_refinement_rounds=2)
        self.assertEqual(plain.candidate_records[-1]['tuning_resolution'], 'unresolved')
        self.assertEqual(shifted.candidate_records[-1]['tuning_resolution'], 'resolved')
        checks = []
        for result in (plain, shifted):
            checks.append(next(c for r in result.candidate_records if r['refinement_iteration'] == 1
                for c in r.get('tuning_checks', []) if c['dimension'] == 'shrinkage'))
        self.assertEqual(checks[0]['absolute_mean_gap'], checks[1]['absolute_mean_gap'])
        self.assertEqual(checks[0]['paired_difference_se'], checks[1]['paired_difference_se'])
        self.assertNotEqual(checks[0]['objective_tolerance'], checks[1]['objective_tolerance'])


if __name__ == '__main__':
    unittest.main()
