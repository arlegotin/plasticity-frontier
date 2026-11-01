"""Public, configuration-bound rational fixture evidence; no floating certificates."""
from decimal import Decimal
from fractions import Fraction
import hashlib
import json
from pathlib import Path

from checks.exact_reference import load_exact_cases, frac, primary_rational_qp, verify_rational_kkt
from .records import content_hash


def export_exact(config_path, branch):
    config_path = Path(config_path)
    config_bytes = config_path.read_bytes()
    raw = json.loads(config_bytes, parse_float=Decimal)
    if branch not in ('F', 'Z'):
        raise ValueError('exact branch must be F or Z')
    expected_fields = {'schema_version', 'spectrum', 'n', 'sigma2', 'depth',
                       'initialization_norm_cap', 'trainable_parameter_cap',
                       'exact_fixture_path', 'exact_fixture_id'}
    if set(raw)!=expected_fields or type(raw['schema_version']) is not int or raw['schema_version']!=1:
        raise ValueError('exact export requires the declared single-frontier configuration')
    if (not isinstance(raw['spectrum'],list)
            or any(type(value) not in (int,Decimal) for value in
                   [raw['sigma2'],raw['initialization_norm_cap'],*raw['spectrum']])):
        raise ValueError('exact configuration requires numeric spectrum, noise and budget')
    fixture_path = Path(raw['exact_fixture_path'])
    if fixture_path.is_absolute() or '..' in fixture_path.parts:
        raise ValueError('exact fixture requires a repository-relative input path')
    normative = Path(__file__).resolve().parents[2]/'tests/fixtures/exact_cases.json'
    fixture_bytes = fixture_path.read_bytes()
    if fixture_bytes!=normative.read_bytes():
        raise ValueError('exact fixture differs from the committed normative fixture')
    fixture = load_exact_cases(fixture_path)['primary_richer_k_3_5']
    matches = (raw['exact_fixture_id']=='primary_richer_k_3_5'
               and type(raw['depth']) is int and raw['depth']==2
               and type(raw['trainable_parameter_cap']) is int and raw['trainable_parameter_cap']==21
               and type(raw['n']) is int and raw['n']==fixture['n']
               and Fraction(raw['initialization_norm_cap'])==frac(fixture['budget'])
               and Fraction(raw['sigma2'])==frac(fixture['sigma2'])
               and [Fraction(x) for x in raw['spectrum']]==list(map(frac, fixture['spectrum'])))
    if not matches:
        raise ValueError('configuration differs from the exact primary fixture')
    rational = primary_rational_qp(fixture, branch)
    active = [3]
    verification = verify_rational_kkt(rational, tuple(active))
    if not verification['verified']:
        raise ValueError('exact rational fixture verification failed')
    contract = {
        'architecture': {'class': 'all-entry deep linear, zero-output initialization',
                         'depth': 2, 'branch': branch, 'width': 3, 'learning_rate': [1, 1],
                         'biases': False, 'tied_weights': False, 'frozen_features': False},
        'population': {'input_law': 'independent Gaussian N(0,I_d)', 'dimension': 6,
                       'n': fixture['n'], 'spectrum': fixture['spectrum'], 'mean': [0, 1],
                       'sigma2': fixture['sigma2'], 'task_second_moment_trace': [1, 1],
                       'noise': 'independent mean zero', 'information_access': 'oracle'},
        'baseline': {'scope': 'branch-constrained rational QP optimum at declared caps',
                     'branch': branch, 'fixture_id': raw['exact_fixture_id']},
        'metric': {'risk': 'expected squared coefficient error [coefficient^2]',
                   'adaptation_loss': 'squared_error_over_2n'},
        'resource_contract': {'initialization_norm_cap': fixture['budget'],
                              'trainable_parameter_cap': 21, 'depth': 2,
                              'count_rule': 'every matrix entry, including initialized zeros'},
    }
    return {'schema_version': 1, 'status': 'pass', 'certificate_level': 'exact_rational',
            'branch': branch, 'fixture_id': raw['exact_fixture_id'],
            'configuration': json.loads(config_bytes),
            'configuration_sha256': hashlib.sha256(config_bytes).hexdigest(),
            'fixture_sha256': hashlib.sha256(fixture_bytes).hexdigest(),
            'rational_qp': rational, 'active_indices': active,
            'verification': verification, 'contract': contract,
            'qualification': 'Exact finite fixture KKT verification; not a global theorem or empirical campaign.'}


def verify_export(raw, entry, source_root=None):
    """Rebuild from inventoried executed input, rather than trusting self-description."""
    command = entry.get('command', [])
    if (len(command)!=5 or command[0]!='export-exact' or command[1]!='--config'
            or command[3]!='--branch' or command[4]!=raw.get('branch')):
        raise ValueError('rigorous exact evidence requires its recorded export-exact command')
    root=Path(source_root) if source_root is not None else Path(__file__).resolve().parents[2]
    config = (root/command[2]).resolve()
    assets = {Path(value).resolve() for value in entry.get('assets', [])}
    if config not in assets:
        raise ValueError('exact configuration must be an inventory-bound asset')
    configuration = json.loads(config.read_bytes())
    fixture = (root/configuration['exact_fixture_path']).resolve()
    if fixture not in assets:
        raise ValueError('exact fixture must be an inventory-bound asset')
    for path in (config, fixture):
        if hashlib.sha256(path.read_bytes()).hexdigest()!=entry['inventory'].get(str(path)):
            raise ValueError('exact input inventory changed')
    expected = export_exact(config, command[4])
    # Verify the received integer pairs themselves. Python numeric equality and
    # tolerant float replay are insufficient for an exact rational certificate.
    verification=verify_rational_kkt(raw['rational_qp'],tuple(raw['active_indices']))
    if not verification['verified'] or content_hash(raw)!=content_hash(expected):
        raise ValueError('exact rational evidence or contract differs from executed configuration')
    return verification
