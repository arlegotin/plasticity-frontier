"""Check compact public evidence without installing numerical dependencies."""
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'docs/results/release-v0.1.0'


def rows(name):
    with (DATA / name).open(newline='') as stream:
        return list(csv.DictReader(stream))


def close(actual, expected):
    if not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-14):
        raise ValueError(f'Numerical mismatch: {actual} != {expected}')


def verify():
    manifest = json.loads((DATA / 'provenance.json').read_text())
    for name, record in manifest['files'].items():
        payload = (DATA / name).read_bytes()
        if len(payload) != record['bytes'] or hashlib.sha256(payload).hexdigest() != record['sha256']:
            raise ValueError(f'Changed public evidence: {name}')

    primary = rows('primary_31_pairs.csv')
    if len(primary) != 31 or [int(r['preparation_id']) for r in primary] != list(range(31)):
        raise ValueError('Expected all 31 ordered independent preparations')
    if len({r['source_key'] for r in primary}) != 31:
        raise ValueError('Duplicate source preparation')
    baseline = [float(r['baseline_loss']) for r in primary]
    joint = [float(r['joint_loss']) for r in primary]
    differences = [a-b for a, b in zip(baseline, joint)]
    for difference, row in zip(differences, primary):
        close(difference, float(row['improvement_baseline_minus_joint']))
    summary = manifest['primary_summary']
    mean = statistics.mean(differences)
    se = statistics.stdev(differences) / math.sqrt(31)
    close(statistics.mean(baseline), summary['baseline_mean'])
    close(statistics.mean(joint), summary['joint_mean'])
    close(mean, summary['paired_improvement'])
    close(se, summary['standard_error'])
    # The archived run's Student critical value is recorded, not re-estimated.
    half = summary['student_critical_value'] * se
    close(mean-half, summary['approximate_95_interval'][0])
    close(mean+half, summary['approximate_95_interval'][1])
    close(100*mean/statistics.mean(baseline), summary['descriptive_reduction_percent'])

    tasks = rows('benchmark_2048_task_losses.csv')
    methods = rows('benchmark_method_summary.csv')
    if len(tasks) != 2048 or len({r['task_key'] for r in tasks}) != 2048 or len(methods) != 14:
        raise ValueError('Expected 2,048 unique shared tasks and 14 methods')
    for method in methods:
        name = method['method_id']
        close(statistics.mean(float(r[name]) for r in tasks), float(method['empirical_mean_of_2048_losses']))
    for name in ('B01', 'B02', 'B03', 'B04', 'B11:F', 'B11:Z'):
        if any(float(r[name]) != float(r['B05:'+name]) for r in tasks):
            raise ValueError(f'Direct-control loss mismatch: {name}')

    receipt_bytes = gzip.decompress((DATA / 'clean-replay.json.gz').read_bytes())
    if hashlib.sha256(receipt_bytes).hexdigest() != manifest['replay_uncompressed_sha256']:
        raise ValueError('Replay receipt decompression mismatch')
    receipt = json.loads(receipt_bytes)
    gate = json.loads((DATA / 'gate_decision.json').read_text())
    if (receipt['status'] != 'matched' or not receipt['clean_environment'] or receipt['fixture']
            or receipt['source']['git_commit'] != manifest['executed_source']
            or len(receipt['executions']) != 16 or gate['code'] != 'SUPPORTED_IN_SCOPE'):
        raise ValueError('Archived replay/gate scope mismatch')
    print('Verified: 31 paired preparations; 2,048 shared tasks; 14 methods; six equality controls.')
    print(f'Mean loss: {statistics.mean(baseline):.6f} -> {statistics.mean(joint):.6f}; reduction {100*mean/statistics.mean(baseline):.2f}%.')
    print('Archived receipt: matched same-machine clean replay, 16 inputs. Full raw store is not included.')
    return manifest


if __name__ == '__main__':
    verify()
