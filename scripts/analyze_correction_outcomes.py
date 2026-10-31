"""Summarize initial/final transitions and structural-evidence direction groups."""
import argparse
import json
from pathlib import Path
from scripts.analyze_fixed_subsets import load_run, validate_aligned_runs
from scripts.summarize_targeted_experiments import describe, miou
import numpy as np


def rate(numerator, denominator):
    return 100*numerator/denominator if denominator else None


def summary(rows):
    if any(value is None for value in rows):
        return {'mean':None,'sample_std':None}
    return describe(rows)


def analyze(paths):
    runs=[load_run(path) for path in paths]
    validate_aligned_runs(runs)
    if len(runs)!=3 or {r['seed'] for r in runs}!={0,1,2}:
        raise ValueError('Provide three matched B5 seed files')
    if any(r['variant']!='B5' or str(r['budget'])!='1000' for r in runs):
        raise ValueError('Correction analysis requires B5 at budget 1000')
    outputs=[]
    for run in sorted(runs,key=lambda r:r['seed']):
        faces=list(run['faces'].values())
        before=[f['pred_before']==f['target'] for f in faces]
        after=[f['pred_after']==f['target'] for f in faces]
        n=len(faces)
        wrong=sum(not v for v in before)
        classes={int(c):i for i,c in enumerate(run['classes'])}
        matrices=[]
        for field in ['pred_before','pred_after']:
            cm=np.zeros((len(classes),len(classes)),dtype=np.int64)
            for f in faces:cm[classes[f['target']],classes[f[field]]]+=1
            matrices.append(cm)
        transitions={'wrong_to_correct_percent':rate(sum(not b and a for b,a in zip(before,after)),n),
                     'correct_to_wrong_percent':rate(sum(b and not a for b,a in zip(before,after)),n),
                     'wrong_to_wrong_percent':rate(sum(not b and not a for b,a in zip(before,after)),n),
                     'top2_error_coverage_percent':rate(sum(not b and f['target']==f['candidate_b'] for f,b in zip(faces,before)),wrong),
                     'delta_miou_pp':100*(miou(matrices[1])-miou(matrices[0]))}
        groups={key:[] for key in ['both_top1','both_top2','conflicting','at_least_one_zero']}
        for f,b,a in zip(faces,before,after):
            if f['gate']<=0:continue
            s,u=f['local_evidence'],f['part_prior']
            key='both_top1' if s>0 and u>0 else 'both_top2' if s<0 and u<0 else 'conflicting' if s*u<0 else 'at_least_one_zero'
            groups[key].append((b,a))
        active=sum(map(len,groups.values()))
        evidence={}
        for key,items in groups.items():
            evidence[key]={'share_percent':rate(len(items),active),
                           'recovery_percent':rate(sum(not b and a for b,a in items),sum(not b for b,a in items)),
                           'damage_percent':rate(sum(b and not a for b,a in items),sum(b for b,a in items))}
        outputs.append({'seed':run['seed'],'transitions':transitions,'evidence_groups':evidence})
    return {'runs':outputs,'transitions':{k:summary([r['transitions'][k] for r in outputs]) for k in outputs[0]['transitions']},
            'evidence_groups':{g:{k:summary([r['evidence_groups'][g][k] for r in outputs]) for k in outputs[0]['evidence_groups'][g]} for g in outputs[0]['evidence_groups']}}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-jsonl',type=Path,nargs=3,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();result=analyze(a.input_jsonl)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2)+'\n')


if __name__ == '__main__':main()
