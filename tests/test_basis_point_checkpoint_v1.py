import pytest
from openea_benchmark.adaptive.basis_point_checkpoint import load_basis_checkpoint, save_basis_checkpoint
from openea_benchmark.attachment.basis_convergence import EAIntervalEV, ElectronicEABasisPoint

def point(): return ElectronicEABasisPoint(basis_name='aug-cc-pv5z',cardinal_number=5,augmentation_level=1,method_signature='CCSD(T)|TEST',ea=EAIntervalEV(1.8,1.81,1.82),is_production_ea=False)

def test_checkpoint_roundtrip(tmp_path):
    path=tmp_path/'p.json'; save_basis_checkpoint(path,signature='sig',source='LIVE',points=(point(),)); x=load_basis_checkpoint(path,expected_signature='sig'); assert x.source=='LIVE'; assert x.points[0].ea.central_ev==1.81

def test_checkpoint_signature_mismatch_fails_closed(tmp_path):
    path=tmp_path/'p.json'; save_basis_checkpoint(path,signature='a',source='LIVE',points=(point(),))
    with pytest.raises(ValueError): load_basis_checkpoint(path,expected_signature='b')

def test_checkpoint_never_promotes_production(tmp_path):
    path=tmp_path/'p.json'; save_basis_checkpoint(path,signature='sig',source='LIVE',points=(point(),)); assert load_basis_checkpoint(path,expected_signature='sig').points[0].is_production_ea is False
