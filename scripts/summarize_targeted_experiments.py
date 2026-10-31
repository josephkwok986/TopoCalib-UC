"""Whole-part paired bootstrap comparisons and adjacency-deletion summaries."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import statistics
import numpy as np


def load_runs(paths):
    rows = [json.loads(Path(p).read_text()) for p in paths]
    if not rows:
        raise ValueError('No results supplied')
    if any(r.get('split_group') != 'test' for r in rows):
        raise ValueError('Summaries require test inference results')
    if len({(tuple(r['classes']), tuple(r['dataset']), str(r['budget'])) for r in rows}) != 1:
        raise ValueError('Class mapping, dataset and budget must match')
    return rows


def miou(matrix):
    union = matrix.sum(0) + matrix.sum(1) - np.diag(matrix)
    valid = union > 0
    return float(np.mean(np.diag(matrix)[valid] / union[valid])) if valid.any() else 0.0


def matrices(row, part_ids):
    classes = {int(c): i for i, c in enumerate(row['classes'])}
    parts = {p['part_id']: p for p in row['parts']}
    if len(parts) != len(row['parts']) or set(parts) != set(part_ids):
        raise ValueError('Part memberships differ or contain duplicates')
    out = []
    for pid in part_ids:
        p = parts[pid]
        if len(p['target']) != len(p['pred']) or len(p['target']) != p['num_faces']:
            raise ValueError('Face counts differ')
        cm = np.zeros((len(classes), len(classes)), dtype=np.int64)
        for y, pred in zip(p['target'], p['pred']):
            cm[classes[y], classes[pred]] += 1
        out.append(cm)
    return np.asarray(out)


def paired(left, right, resamples=10000, seed=0, randomization=False):
    if resamples < 1:
        raise ValueError('resamples must be positive')
    l = {r['seed']: r for r in left}; r = {x['seed']: x for x in right}
    if len(l) != len(left) or len(r) != len(right) or set(l) != {0, 1, 2} or set(r) != set(l):
        raise ValueError('Comparisons require one result per matched seed 0,1,2')
    all_rows = left + right
    if len({(tuple(x['classes']), tuple(x['dataset']), str(x['budget'])) for x in all_rows}) != 1:
        raise ValueError('Dataset, classes and budget must match')
    ids = sorted(p['part_id'] for p in left[0]['parts'])
    if not ids:
        raise ValueError('Empty test split')
    reference = {p['part_id']: p['target'] for p in left[0]['parts']}
    for row in all_rows:
        if {p['part_id']: p['target'] for p in row['parts']} != reference:
            raise ValueError('Ground truth or face ordering differs')
    lm = [matrices(l[s], ids) for s in sorted(l)]
    rm = [matrices(r[s], ids) for s in sorted(r)]
    differences = [100 * (miou(a.sum(0)) - miou(b.sum(0))) for a, b in zip(lm, rm)]
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(resamples):
        idx = rng.integers(0, len(ids), len(ids))
        draws.append(np.mean([100*(miou(a[idx].sum(0))-miou(b[idx].sum(0))) for a,b in zip(lm,rm)]))
    result = {'difference_direction': 'left minus right', 'paired_difference_pp': statistics.mean(differences),
            'seed_differences_pp': differences, 'ci95_pp': np.percentile(draws, [2.5,97.5]).tolist(),
            'left_miou_percent': describe([100*miou(a.sum(0)) for a in lm]),
            'right_miou_percent': describe([100*miou(a.sum(0)) for a in rm]),
            'resamples': resamples, 'bootstrap_seed': seed,
            'uncertainty': 'whole test parts conditional on the three training runs'}
    result['positive_seeds'] = sum(d > 0 for d in differences)
    if randomization:
        extreme = 0
        observed = abs(statistics.mean(differences))
        for _ in range(resamples):
            swap = rng.integers(0, 2, len(ids)).astype(bool)[:, None, None]
            deltas = []
            for a, b in zip(lm, rm):
                x = np.where(swap, b, a).sum(0)
                y = np.where(swap, a, b).sum(0)
                deltas.append(100 * (miou(x) - miou(y)))
            extreme += abs(statistics.mean(deltas)) >= observed - 1e-12
        result['randomization_p'] = (extreme + 1) / (resamples + 1)
        result['randomization_resamples'] = resamples
    return result


def describe(values):
    return {'mean': statistics.mean(values), 'sample_std': statistics.stdev(values) if len(values)>1 else None}


def topology(rows):
    groups = {}
    for row in rows:
        key = (row['seed'], row['adjacency_deletion'])
        realization = row['perturbation_seed']
        bucket = groups.setdefault(key, {})
        if realization in bucket:
            raise ValueError('Duplicate perturbation realization')
        bucket[realization] = row
    expected = {(s, p) for s in [0,1,2] for p in [0,.1,.3,.5]}
    if set(groups) != expected:
        raise ValueError('Unexpected training seeds or deletion probabilities')
    output = []
    for p in [0, .1, .3, .5]:
        calibrated, changes, gains = [], [], []
        for seed in [0,1,2]:
            current = groups.get((seed,p), {})
            clean = groups.get((seed,0), {})
            if set(current) != ({0} if p == 0 else {0,1,2}) or set(clean) != {0}:
                raise ValueError('Incomplete deletion grid or training seeds')
            baseline = clean[0]
            ids = sorted(part['part_id'] for part in baseline['parts'])
            targets = {part['part_id']: part['target'] for part in baseline['parts']}
            base_predictions = {part['part_id']: part['base_pred'] for part in baseline['parts']}
            for row in current.values():
                if row['checkpoint'] != baseline['checkpoint']:
                    raise ValueError('Clean and perturbed conditions must share a checkpoint')
                if {part['part_id']: part['target'] for part in row['parts']} != targets:
                    raise ValueError('Test faces differ between perturbations')
                if {part['part_id']: part['base_pred'] for part in row['parts']} != base_predictions:
                    raise ValueError('Initial predictions changed between perturbations')
            initial = dict(baseline, parts=[dict(part, pred=part['base_pred']) for part in baseline['parts']])
            score = statistics.mean(miou(matrices(row, ids).sum(0)) for row in current.values())
            calibrated.append(100*score)
            changes.append(100*(score-miou(matrices(baseline,ids).sum(0))))
            gains.append(100*(score-miou(matrices(initial,ids).sum(0))))
        output.append({'deletion_probability':p,'miou_percent':describe(calibrated),
                       'change_from_clean_pp':describe(changes),'inference_calibration_gain_pp':describe(gains)})
    return output


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--left', nargs='+', type=Path, required=True)
    p.add_argument('--right', nargs='+', type=Path)
    p.add_argument('--topology', action='store_true')
    p.add_argument('--randomization', action='store_true')
    p.add_argument('--resamples', type=int, default=10000)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if not a.topology and not a.right:
        p.error('--right is required for paired comparison')
    result = topology(load_runs(a.left)) if a.topology else paired(load_runs(a.left),load_runs(a.right),a.resamples,a.seed,a.randomization)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2)+'\n')


if __name__ == '__main__':
    main()
