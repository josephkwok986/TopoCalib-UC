"""Reporting denominators, validation-only selection, and multiplicity checks."""
import json
from pathlib import Path
import pytest
from scripts.adjust_component_pvalues import holm
from scripts.analyze_correction_outcomes import analyze
from scripts.select_control_strength import select
from scripts.summarize_targeted_experiments import paired


def test_holm_monotonicity_in_original_order():
    assert holm([.03,.001,.04,.02]) == pytest.approx([.06,.004,.06,.06])


def test_strength_selection_uses_validation_and_complete_grid(tmp_path):
    paths=[]
    for strength in [0,.25,.5,1,2]:
        for seed in [0,1,2]:
            path=tmp_path/f'{strength}_{seed}.json'
            path.write_text(json.dumps({'budget':'1000','seed':seed,'cache_dir':'fixture',
                                        'split_group':'val','frequency_eta':strength,
                                        'metrics':{'miou':.5 if strength in [.25,.5] else .4}}))
            paths.append(path)
    result=select(paths)
    assert result['selected_strength']==.25
    row=json.loads(paths[0].read_text());row['split_group']='test'
    paths[0].write_text(json.dumps(row))
    with pytest.raises(ValueError,match='validation'):
        select(paths)


def test_correction_denominators_and_undefined_rates(tmp_path):
    paths=[]
    # One recovery, one damage, one persistent error, one unchanged correct.
    values=[(0,1,0,0,-1.,-1.),(1,1,0,0,-1.,-1.),
            (0,1,1,1,1.,1.),(0,0,0,1,1.,1.)]
    for seed in [0,1,2]:
        path=tmp_path/f'{seed}.jsonl';rows=[]
        for i,(target,before,after,second,local,part) in enumerate(values):
            rows.append({'dataset':'fixture','method':'topocalib_uc','variant':'B5','budget':'1000','seed':seed,
                         'part_id':'a','face_id':i,'classes':[0,1],'target':target,'pred_before':before,
                         'pred_after':after,'candidate_b':second,'local_evidence':local,'part_prior':part,
                         'gate':1.,'neighbor_face_ids':[]})
        path.write_text('\n'.join(json.dumps(row) for row in rows)+'\n');paths.append(path)
    result=analyze(paths)
    assert result['transitions']['wrong_to_correct_percent']['mean']==25
    assert result['transitions']['correct_to_wrong_percent']['mean']==25
    assert result['transitions']['wrong_to_wrong_percent']['mean']==25
    assert result['transitions']['top2_error_coverage_percent']['mean']==50
    assert result['evidence_groups']['both_top2']['recovery_percent']['mean']==100
    assert result['evidence_groups']['both_top2']['damage_percent']['mean']==100
    assert result['evidence_groups']['conflicting']['recovery_percent']['mean'] is None


def test_randomization_plus_one_for_zero_difference():
    rows=[{'seed':s,'dataset':['fixture'],'budget':'1000','classes':[0,1],
           'parts':[{'part_id':'a','num_faces':2,'pred':[0,1],'target':[0,1]}]} for s in [0,1,2]]
    result=paired(rows,rows,resamples=10,seed=1,randomization=True)
    assert result['randomization_p']==1.
    assert result['paired_difference_pp']==0.
