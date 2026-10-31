"""Plan or execute the targeted controls using the standard training/inference entries."""
from __future__ import annotations
import argparse
import itertools
import json
from pathlib import Path
import subprocess
import sys
from topocalib_uc.batching import positive_part_count


def make_jobs(args):
    config = json.loads(args.config.read_text())
    if args.selected_strength is not None:
        selection = json.loads(args.selected_strength.read_text())
        expected = {'frequency': 'frequency', 'train': ('smoothing' if 'smoothing' in config.get('arguments', []) else 'all_class')}.get(config['kind'])
        if selection['control'] != expected or 'strengths' not in config:
            raise ValueError('Selected strength does not match this control')
        if selection.get('cache_dir') != str(args.cache_dir):
            raise ValueError('Selected strength belongs to a different dataset cache')
        config['strengths'] = [selection['selected_strength']]
    root = args.output_root
    jobs = []
    for budget, seed in itertools.product(config['budgets'], config['seeds']):
        values = dict(dataset=args.dataset, budget=budget, seed=seed)
        common = ['--cache-dir', str(args.cache_dir), '--ssrl-cache-dir', str(args.ssrl_cache_dir),
                  '--split-manifest', str(args.split_manifest), '--batch-part-count', str(getattr(args, 'batch_part_count', 16))]
        kind = config['kind']
        if kind == 'train':
            for strength in config.get('strengths', [None]):
                name = f'{args.dataset}_budget{budget}_seed{seed}'
                if strength is not None:
                    name += f'_strength{strength:g}'
                output = root / (name + '.json')
                flags = ['--labeled-part-budget', str(budget), '--seed', str(seed), '--variant', 'B5',
                         '--labeled-parts-manifest', args.labeled_parts_template.format(**values),
                         '--output', str(output)] + config['arguments'] + args.train_arg
                if config.get('validation_only', False):
                    flags += ['--validation-only']
                if strength is not None:
                    flags += [config['strength_argument'], str(strength)]
                jobs.append([sys.executable, '-m', 'topocalib_uc.train.run_frozen_proto', *common, *flags])
                if args.export_predictions:
                    jobs.append([sys.executable, '-m', 'topocalib_uc.inference.run', *common,
                                 '--checkpoint', str(output.with_suffix('.pt')), '--split-group', args.export_predictions,
                                 '--save-logits', '--output', str(output.with_name(output.stem + '_predictions.json'))])
        else:
            if not args.checkpoint_template:
                raise ValueError('Inference requires --checkpoint-template')
            checkpoint = args.checkpoint_template.format(**values)
            if config['kind'] == 'frequency':
                settings = [(['--frequency-eta', str(v)], f'eta{v:g}') for v in config['strengths']]
            elif config['kind'] == 'topology':
                settings = [(['--adjacency-deletion', str(p), '--perturbation-seed', str(r)], f'delete{p:g}_realization{r}')
                            for p in config['deletion_probabilities']
                            for r in ([0] if p == 0 else config['perturbation_seeds'])]
            else:
                raise ValueError(f'Unknown experiment kind: {kind}')
            for flags, suffix in settings:
                output = root / f'{args.dataset}_budget{budget}_seed{seed}_{suffix}.json'
                jobs.append([sys.executable, '-m', 'topocalib_uc.inference.run', *common,
                             '--checkpoint', checkpoint, '--split-group', args.split_group,
                             '--output', str(output), '--save-logits', *flags])
    return jobs


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--batch-part-count', '--batch-size', type=positive_part_count, default=16)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--dataset', choices=['fusion360', 'mfcadpp'], required=True)
    p.add_argument('--cache-dir', type=Path, required=True)
    p.add_argument('--ssrl-cache-dir', type=Path, required=True)
    p.add_argument('--split-manifest', type=Path, required=True)
    p.add_argument('--labeled-parts-template', default='', help='Path template containing {budget} and {seed}')
    p.add_argument('--checkpoint-template', default='')
    p.add_argument('--split-group', choices=['val', 'test'], default='test')
    p.add_argument('--output-root', type=Path, required=True)
    p.add_argument('--train-arg', action='append', default=[], help='Additional training token; use --train-arg=--epochs --train-arg=200')
    p.add_argument('--selected-strength', type=Path, help='JSON produced by select_control_strength')
    p.add_argument('--export-predictions', choices=['val', 'test'])
    p.add_argument('--execute', action='store_true', help='Execute the plan; otherwise only write commands.json')
    args = p.parse_args()
    config = json.loads(args.config.read_text())
    if config['kind'] == 'train' and not args.labeled_parts_template:
        p.error('Training requires --labeled-parts-template')
    if config.get('validation_only') and args.export_predictions == 'test':
        p.error('Validation-only configurations cannot export test predictions')
    if config['kind'] == 'frequency' and args.config.stem.endswith('_validation') and args.split_group != 'val':
        p.error('Frequency strength search requires --split-group val')
    jobs = make_jobs(args)
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / 'commands.json').write_text(json.dumps(jobs, indent=2) + '\n')
    if args.execute:
        for job in jobs:
            subprocess.run(job, check=True)
    print(json.dumps({'jobs': len(jobs), 'executed': args.execute, 'output_root': str(args.output_root)}))


if __name__ == '__main__':
    main()
