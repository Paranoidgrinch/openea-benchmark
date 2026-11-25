"""Blockwise UCCSD integrals (uccsd_eris.py) equal PySCF's in-core ones block by block."""
import numpy as np
import pytest
from pyscf import cc, gto, scf

from openea_benchmark.uccsd_eris import blockwise_peak_mb, kept_mb, make_eris_blockwise

BLOCKS = ("oooo ovoo ovov oovv ovvo ovvv vvvv OOOO OVOO OVOV OOVV OVVO OVVV VVVV "
          "ooOO ovOO ovOV ooVV ovVO ovVV vvVV OVoo OOvv OVvo OVvv").split()


@pytest.mark.parametrize("frozen", [0, 2])
def test_blockwise_eris_equal_pyscf_incore_and_give_the_same_energy(frozen):
    from pyscf.cc import uccsd
    mol = gto.M(atom="C 0 0 0; N 0 0 1.1718", basis="aug-cc-pvdz", spin=1, verbose=0)
    mf = scf.ROHF(mol).run(conv_tol=1e-11)
    umf = mf.to_uhf()
    umf.mo_energy, umf.mo_coeff = umf.canonicalize(umf.mo_coeff, umf.mo_occ)
    mycc = cc.UCCSD(umf, frozen=frozen)
    ref = uccsd._make_eris_incore(mycc)
    new = make_eris_blockwise(mycc)
    for name in BLOCKS:
        a, b = np.asarray(getattr(ref, name)), np.asarray(getattr(new, name))
        assert a.shape == b.shape, name
        assert np.abs(a - b).max() < 1e-10, name   # BLAS summation order only
    assert np.allclose(ref.focka, new.focka) and np.allclose(ref.fockb, new.fockb)
    mycc.conv_tol = 1e-10
    e_ref = mycc.kernel(eris=ref)[0] + mycc.ccsd_t(eris=ref)
    mycc2 = cc.UCCSD(umf, frozen=frozen)
    mycc2.conv_tol = 1e-10
    e_new = mycc2.kernel(eris=new)[0] + mycc2.ccsd_t(eris=new)
    assert e_new == pytest.approx(e_ref, abs=1e-10)
    nocca, noccb = mycc.nocc
    nmoa, nmob = mycc.nmo
    kept = sum(np.asarray(getattr(new, n)).nbytes for n in BLOCKS) / 1e6
    assert kept == pytest.approx(kept_mb(nocca, noccb, nmoa - nocca, nmob - noccb), rel=1e-9)
    assert blockwise_peak_mb(mol.nao, nocca, noccb, nmoa - nocca, nmob - noccb) > kept
