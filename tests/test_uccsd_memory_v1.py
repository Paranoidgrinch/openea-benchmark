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


def _request(atoms, basis, spin_2s, scf_reference="ROHF"):
    from openea_benchmark.adaptive.stage3_execution import Stage3ExecutionRequest
    return Stage3ExecutionRequest(
        request_id="r", job_id="j", system="".join(atoms), atoms=atoms, charge=0, spin_2s=spin_2s,
        component_id="c", r_angstrom=1.0, basis=basis, methods=("CCSD", "CCSD(T)"), requested_reference=scf_reference,
        scf_reference=scf_reference, source_link_status="LINKED", source_root_id="root",
        source_checkpoint_path="/nonexistent.chk", source_origin_guess=None, grid_index=0, initialization_index=0,
        dft_center_r_angstrom=1.0, dft_center_energy_hartree=0.0, requires_independent_state_identity_validation=False)


def test_point_budget_follows_basis_size_up_to_the_cap(monkeypatch):
    from openea_benchmark.adaptive import stage3_execution as s3
    settings = s3.Stage3ExecutionSettings(max_memory_mb=48000)
    small = s3.point_memory_budget_mb(_request(("O", "H"), "cc-pvdz", 1), settings)
    assert s3.POINT_MEMORY_FLOOR_MB <= small < s3.POINT_MEMORY_FLOOR_MB + 100  # OH/cc-pVDZ: interpreter floor
    big = _request(("C", "C"), "aug-cc-pv5z", 2)
    assert s3.point_memory_budget_mb(big, settings) == s3.DEFAULT_POINT_MEMORY_CAP_MB
    monkeypatch.setenv(s3.POINT_MEMORY_CAP_ENV, "9000")
    assert s3.point_memory_budget_mb(big, settings) == 9000
    assert s3.point_memory_budget_mb(big, s3.Stage3ExecutionSettings(max_memory_mb=5000)) == 5000
    mid = s3.point_memory_budget_mb(_request(("C", "C"), "aug-cc-pvtz", 2), settings)
    assert s3.POINT_MEMORY_FLOOR_MB < mid < 9000


def test_point_budget_exceeds_the_cap_only_with_gate_headroom(monkeypatch):
    from openea_benchmark import point_pool
    from openea_benchmark.adaptive import stage3_execution as s3
    settings = s3.Stage3ExecutionSettings(max_memory_mb=48000)
    big = _request(("C", "C"), "aug-cc-pv5z", 2)  # needs more than 48 GB
    monkeypatch.setattr(point_pool, "memory_headroom_mb", lambda: 300000.0)
    assert s3.point_memory_budget_mb(big, settings) == 48000  # the stage limit, not the 16 GB cap
    monkeypatch.setattr(point_pool, "memory_headroom_mb", lambda: 100000.0)  # 60 GB > half of it
    assert s3.point_memory_budget_mb(big, settings) == s3.DEFAULT_POINT_MEMORY_CAP_MB
    monkeypatch.setattr(point_pool, "memory_headroom_mb", lambda: None)  # no memory gate
    assert s3.point_memory_budget_mb(big, settings) == s3.DEFAULT_POINT_MEMORY_CAP_MB
    monkeypatch.setattr(point_pool, "memory_headroom_mb", lambda: 300000.0)
    monkeypatch.setenv(s3.POINT_MEMORY_ADAPTIVE_ENV, "0")
    assert s3.point_memory_budget_mb(big, settings) == s3.DEFAULT_POINT_MEMORY_CAP_MB
