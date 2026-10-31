"""Mathematical invariants for structural controls and perturbations."""
import torch
from topocalib_uc.calibration.controls import structural_evidence, apply_control, frequency_adjustment
from topocalib_uc.calibration.margin import apply_margin_calibration
from topocalib_uc.candidate_pair.pairs import inference_candidate_pairs
from topocalib_uc.evidence.readout import FaceBatchMeta, local_candidate_pair_evidence, part_level_candidate_pair_prior
from topocalib_uc.inference.perturbation import delete_adjacencies


def test_evidence_and_matched_margin():
    torch.manual_seed(11)
    logits = torch.randn(5,3,requires_grad=True)
    probs = logits.softmax(1)
    a,b,m = inference_candidate_pairs(probs)
    meta = FaceBatchMeta(torch.tensor([0,0,0,1,1]),torch.tensor([[0,1],[1,2],[3,4]]),torch.tensor([0,1,3]))
    evidence = structural_evidence(probs,meta,lambda_part=.5,top_k=2)
    rows = torch.arange(5)
    expected = local_candidate_pair_evidence(probs,a,b,meta)+.5*part_level_candidate_pair_prior(probs,a,b,meta,top_k=2)
    torch.testing.assert_close(evidence[rows,a]-evidence[rows,b],expected)
    assert not evidence.requires_grad
    kw = dict(use_local_evidence=True,use_ambiguity_gate=True,use_part_prior=True,beta=.5,lambda_cal=1.,lambda_part=.5,top_k=2)
    pair = apply_margin_calibration(logits,probs,a,b,m,meta,**kw)
    full = apply_control(logits,probs,a,b,m,meta,mode='all_class',lambda_all=1.,**kw)
    torch.testing.assert_close(pair[rows,a]-pair[rows,b],full[rows,a]-full[rows,b])
    smooth = apply_control(logits,probs,a,b,m,meta,mode='smoothing',smoothing_alpha=.5,**kw)
    torch.testing.assert_close(smooth.exp().sum(1),torch.ones(5))
    torch.nn.functional.cross_entropy(smooth,torch.tensor([0,1,2,1,0])).backward()
    assert torch.isfinite(logits.grad).all()


def test_isolated_face_retains_distribution():
    logits=torch.tensor([[1.,2.,3.]],requires_grad=True)
    probs=logits.softmax(1);a,b,m=inference_candidate_pairs(probs)
    meta=FaceBatchMeta(torch.zeros(1,dtype=torch.long),torch.empty(0,2,dtype=torch.long),torch.empty(0,dtype=torch.long))
    result=apply_control(logits,probs,a,b,m,meta,mode='smoothing',lambda_part=.5,top_k=5)
    torch.testing.assert_close(result.exp(),probs)


def test_deletion_groups_directions_and_keeps_part_membership():
    meta=FaceBatchMeta(torch.zeros(4,dtype=torch.long),torch.tensor([[0,1],[1,0],[1,2],[2,1],[2,3],[3,2]]),torch.zeros(6,dtype=torch.long))
    changed=delete_adjacencies(meta,.5,7,'part')
    pairs=set(map(tuple,changed.edges.tolist()))
    assert all((b,a) in pairs for a,b in pairs)
    assert changed.part_index is meta.part_index
    assert delete_adjacencies(meta,0,7,'part') is meta
    assert delete_adjacencies(meta,1,7,'part').edges.numel()==0
    torch.testing.assert_close(changed.edges,delete_adjacencies(meta,.5,7,'part').edges)


def test_frequency_adjustment_sign_and_zero_strength():
    logits=torch.zeros(2,3)
    counts=[1,2,4]
    adjusted=frequency_adjustment(logits,counts,.5)
    assert adjusted[0,0]>adjusted[0,1]>adjusted[0,2]
    torch.testing.assert_close(frequency_adjustment(logits,counts,0),logits)


def test_paired_bootstrap_preserves_whole_part_alignment():
    import copy
    from scripts.summarize_targeted_experiments import paired
    import pytest
    left=[];right=[]
    for seed in range(3):
        row={'seed':seed,'dataset':['fixture'],'budget':'1000','classes':[0,1],
             'parts':[{'part_id':'a','num_faces':2,'target':[0,1],'pred':[0,1]},
                      {'part_id':'b','num_faces':2,'target':[0,1],'pred':[0,1]}]}
        other=copy.deepcopy(row);other['parts'][1]['pred']=[1,0]
        left.append(row);right.append(other)
    result=paired(left,right,resamples=20,seed=1)
    assert abs(result['paired_difference_pp']-200/3)<1e-10
    assert result['ci95_pp'][0]>=0
    right[0]['parts'][0]['target']=[1,0]
    with pytest.raises(ValueError,match='Ground truth'):
        paired(left,right,resamples=20)
