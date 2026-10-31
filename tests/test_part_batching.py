"""Part batching, global support exclusion, and prediction consistency."""
from pathlib import Path

import pytest
import torch

from topocalib_uc.batching import collate_parts, part_batches
from topocalib_uc.inference.run import _infer_proto, _infer_linear
from topocalib_uc.prototypes.matching import ClassIndex, leave_one_out_logits, prototype_logits
from topocalib_uc.prototypes.strategies import PrototypeStrategyConfig
from topocalib_uc.prototypes.support_bank import PrototypeBank
from topocalib_uc.tokenization.adapter import SurfaceTokenAdapter
from topocalib_uc.train.partgraph_dataset import PartRecord
from topocalib_uc.train.run_frozen_linear import FrozenLinearModel
from topocalib_uc.train.variants import variant_config


def records_and_embeddings():
    generator = torch.Generator().manual_seed(50)
    records, embeddings = [], {}
    for i in range(19):
        n = 3 + i % 4
        part_id = f'p{i}'
        edges = torch.tensor([[j, j+1] for j in range(n-1)])
        records.append(PartRecord(
            index=i, part_id=part_id, dataset='fixture', split='test', path=Path(part_id),
            num_faces=n, y=torch.arange(n) % 3, surface_type=torch.arange(n) % 5,
            edges=edges, edge_type=torch.arange(n-1) % 4,
            face_features_raw=torch.randn(n, 8, generator=generator), source_paths={}, meta={}))
        embeddings[part_id] = records[-1].face_features_raw
    return records, embeddings


def test_part_batches_keep_tail_and_shuffle_reproducibly():
    items = list(range(35))
    batches = list(part_batches(items, 16))
    assert [len(b) for b in batches] == [16,16,3]
    assert [x for b in batches for x in b] == items
    shuffled = list(part_batches(items, 16, shuffle_seed=2))
    assert shuffled == list(part_batches(items, 16, shuffle_seed=2))
    assert shuffled != list(part_batches(items, 16, shuffle_seed=3))
    for invalid in [0, -1, 1.5, True]:
        with pytest.raises(ValueError):
            list(part_batches(items, invalid))


def test_collation_keeps_edges_inside_each_part():
    records, embeddings = records_and_embeddings()
    batch = collate_parts(records, embeddings)
    src, dst = batch['meta'].edges.T
    assert torch.equal(batch['meta'].part_index[src], batch['meta'].part_index[dst])
    for record, selection in zip(records, batch['slices']):
        torch.testing.assert_close(batch['z'][selection], embeddings[record.part_id])


@pytest.mark.parametrize('method', ['mean','medoid'])
def test_global_bank_excludes_snapshot_face_even_when_query_changes(method):
    tokens = torch.tensor([[1.,0.,0.],[0.,1.,0.],[0.,0.,1.],[1.,1.,0.],[0.,1.,1.]])
    labels = torch.tensor([0,0,1,1,2])
    bank = PrototypeBank.build(tokens, labels, 3, PrototypeStrategyConfig(method=method))
    indices = torch.tensor([0,4])
    query = (tokens[indices] + .2).requires_grad_()
    logits = bank.training_logits(query, indices, tau=.2)
    from topocalib_uc.prototypes.strategies import build_single_class_prototype
    for row, index in enumerate(indices):
        cls = labels[index]
        keep = (labels == cls) & (torch.arange(len(labels)) != index)
        expected = bank.prototypes[cls] if not keep.any() else build_single_class_prototype(tokens[keep], bank.config)[0]
        score = prototype_logits(query[row:row+1], expected[None,:], .2)[0,0]
        torch.testing.assert_close(logits[row,cls], score)
    # Class 1 is absent from the query batch but retains its global prototype.
    assert bank.counts.tolist() == [2.,2.,1.]
    torch.testing.assert_close(logits[:,1], prototype_logits(query, bank.prototypes, .2)[:,1])
    logits.sum().backward()
    assert torch.isfinite(query.grad).all()
    assert not bank.prototypes.requires_grad
    assert not bank.tokens.requires_grad


def test_mean_bank_matches_full_support_leave_one_out_values():
    torch.manual_seed(0)
    tokens = torch.randn(8,5)
    labels = torch.tensor([0,0,1,1,1,2,2,3])
    bank = PrototypeBank.build(tokens,labels,4,PrototypeStrategyConfig())
    actual = bank.training_logits(tokens,torch.arange(8),tau=.07)
    expected = leave_one_out_logits(tokens,labels,num_classes=4,tau=.07)
    torch.testing.assert_close(actual,expected)


@pytest.mark.parametrize('mode,deletion,frequency,diagnostics', [
    ('pairwise',0.,False,True),('pairwise',.3,False,True),
    ('smoothing',0.,False,False),('all_class',0.,False,False),
    ('pairwise',0.,True,False)])
def test_proto_predictions_and_diagnostics_are_batch_independent(mode,deletion,frequency,diagnostics):
    torch.manual_seed(9)
    records, embeddings = records_and_embeddings()
    model = SurfaceTokenAdapter(8,6,5)
    checkpoint = {'embedding_dim':8,'hidden_dim':6,'num_surface_types':5,'tau':.7,
                  'model_state_dict':model.state_dict(),'prototypes':torch.randn(3,6),
                  'variant':variant_config('B0' if frequency else 'B5').as_dict(),
                  'method':'topocalib_uc','config':{'seed':0,'labeled_part_budget':1000},
                  'labeled_class_counts':[10,20,30],
                  'calibration':{'mode':mode,'beta':1.,'lambda_part':.5}}
    with torch.no_grad():
        reference, ref_diagnostics = _infer_proto(checkpoint,records,embeddings,ClassIndex([0,1,2]),torch.device('cpu'),
            save_logits=True,collect_records=diagnostics,batch_part_count=1,
            adjacency_deletion=deletion,perturbation_seed=2,frequency_eta=.5 if frequency else None)
        for size in [7,16,32]:
            actual, actual_diagnostics = _infer_proto(checkpoint,records,embeddings,ClassIndex([0,1,2]),torch.device('cpu'),
                save_logits=True,collect_records=diagnostics,batch_part_count=size,
                adjacency_deletion=deletion,perturbation_seed=2,frequency_eta=.5 if frequency else None)
            assert len(actual)==19
            for left,right in zip(reference,actual):
                assert left['part_id']==right['part_id']
                assert left['pred']==right['pred']
                assert left['base_pred']==right['base_pred']
                torch.testing.assert_close(torch.tensor(left['logits']),torch.tensor(right['logits']),atol=3e-6,rtol=1e-5)
            assert len(ref_diagnostics)==len(actual_diagnostics)
            for left,right in zip(ref_diagnostics,actual_diagnostics):
                for key in ['face_key','neighbor_face_ids','target','candidate_a','candidate_b']:
                    assert left[key]==right[key]
                for key in ['gate','local_evidence','part_prior','calibration_delta']:
                    assert left[key]==pytest.approx(right[key],abs=2e-6)


def test_linear_predictions_are_batch_independent():
    records, embeddings = records_and_embeddings()
    model = FrozenLinearModel(8,6,5,3)
    checkpoint={'embedding_dim':8,'hidden_dim':6,'num_surface_types':5,'model_state_dict':model.state_dict()}
    with torch.no_grad():
        rows = [_infer_linear(checkpoint,records,embeddings,ClassIndex([0,1,2]),torch.device('cpu'),
                             save_logits=True,batch_part_count=size) for size in [1,16]]
    for left,right in zip(*rows):
        assert left['pred']==right['pred']
        torch.testing.assert_close(torch.tensor(left['logits']),torch.tensor(right['logits']))
