"""UCCSD integrals go out-of-core when PySCF's in-core estimate undercounts (server OOM 2026-10-12)."""
from pyscf import gto, scf
from pyscf.cc import uccsd

from openea_benchmark.adaptive.stage3_execution import _ccsd_eris_within_memory, uccsd_incore_memory_mb


def test_c2_a5z_incore_estimate_exceeds_48gb():
    assert uccsd_incore_memory_mb(nmo=252, nao=254) > 48000 * 2  # PySCF's own estimate was ~42 GB


def test_outcore_uccsd_t_matches_incore():
    mol = gto.M(atom="O 0 0 0; H 0 0 0.97", basis="cc-pvdz", spin=1, verbose=0)
    mf = scf.ROHF(mol).run().to_uhf()
    energies = []
    for max_memory in (100000, 1):
        mycc = uccsd.UCCSD(mf)
        mycc.max_memory = max_memory
        eris = _ccsd_eris_within_memory(mycc)
        assert hasattr(eris, "feri") == (max_memory == 1)  # out-of-core integrals live in an HDF5 file
        if max_memory == 1:  # AO-direct ladder: no vvvv block written to disk
            assert mycc.direct and "vvvv" not in eris.feri
        mycc.kernel(eris=eris)
        energies.append(mycc.e_tot + mycc.ccsd_t(eris=eris))
    assert abs(energies[0] - energies[1]) < 1e-9
