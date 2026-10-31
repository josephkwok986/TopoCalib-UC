"""Select one control strength from three-seed, 1000-part validation results."""
import argparse
import json
from pathlib import Path
import statistics


def select(paths):
    groups = {}
    identity = set()
    for path in paths:
        row = json.loads(Path(path).read_text())
        config = row.get('config', {})
        budget = str(row.get('budget', config.get('labeled_part_budget', '')))
        if budget != '1000':
            raise ValueError('Strength selection requires budget 1000')
        if 'history_tail' in row:
            mode = config['calibration_mode']
            if mode not in {'smoothing', 'all_class'}:
                raise ValueError('Not a structural control training result')
            strength = config['smoothing_alpha' if mode == 'smoothing' else 'lambda_all']
            score = row['metrics']['val']['miou']
            seed = config['seed']
            identity.add((mode, row['cache_dir']))
        else:
            if row.get('split_group') != 'val' or row.get('frequency_eta') is None:
                raise ValueError('Frequency selection accepts only validation inference results')
            strength, seed, score = row['frequency_eta'], row['seed'], row['metrics']['miou']
            identity.add(('frequency', row['cache_dir']))
        if seed in groups.setdefault(float(strength), {}):
            raise ValueError('Duplicate strength/seed result')
        groups[float(strength)][seed] = float(score)
    if len(identity) != 1 or not groups:
        raise ValueError('Provide exactly one dataset and control')
    for values in groups.values():
        if set(values) != {0, 1, 2}:
            raise ValueError('Every strength requires matched seeds 0,1,2')
    mode = next(iter(identity))[0]
    required = {'smoothing': {0, .25, .5, .75}, 'all_class': {0, .5, 1, 2}, 'frequency': {0, .25, .5, 1, 2}}[mode]
    if set(groups) != required:
        raise ValueError(f'Incomplete strength grid: expected {sorted(required)}')
    means = {strength: statistics.mean(values.values()) for strength, values in groups.items()}
    best = max(sorted(means), key=means.get)
    return {'control': mode, 'cache_dir': next(iter(identity))[1], 'selected_strength': best, 'validation_mean_miou': means,
            'tie_rule': 'smallest strength', 'inputs': [str(p) for p in paths]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, nargs='+', required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    result = select(a.input)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
