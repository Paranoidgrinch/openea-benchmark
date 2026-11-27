import pytest
"""Patch 49: post-CCSD(T) layer (fake CCpy points)."""
from types import SimpleNamespace

from openea_benchmark import post_ccsd_t_stage
from openea_benchmark.post_ccsd_t_stage import (
    PostCCSDTStagePolicy,
    run_post_ccsd_t_stage,
)
from openea_benchmark.vibrational_stage import assemble_adiabatic_ea

from test_cbs_stage_v1 import ea_record, request

H = 1.0 / 27.211386245988
# neutral-minus-anion energy shifts (eV) per method and basis, relative to CCSD(T)
SHIFT = {("aug-cc-pvdz", "CCSDT"): -0.004, ("aug-cc-pvtz", "CCSDT"): -0.006,
         ("aug-cc-pvdz", "CCSDTQ"): -0.004 + 0.010, ("cc-pvdz", "CCSDT"): -0.003,
         ("cc-pvdz", "CCSDTQ"): -0.003 + 0.008}


def fake_point(spec, basis, method, policy, path):
    e = -75.4 if spec.role == "neutral" else -75.48
    if spec.role == "neutral":
        e += SHIFT.get((basis, method), 0.0) * H
    return {"role": spec.role, "basis": basis, "method": method, "status": "COMPLETED", "total_hartree": e}


@pytest.fixture(autouse=True)
def _no_fno_by_default(monkeypatch):
    monkeypatch.setattr(post_ccsd_t_stage, "FNO_BEYOND_TRIPLES", False)


def run(tmp_path, monkeypatch, limit, scratch_free_mb=1.0e9):
    monkeypatch.setattr(post_ccsd_t_stage, "cc_storage_mb",
                        lambda spec, basis, method: (50000.0 if basis == "aug-cc-pvdz" else 10000.0)
                        if method == "CCSDTQ" else 1000.0)
    monkeypatch.setattr(post_ccsd_t_stage, "cc_scratch_need_mb",
                        lambda spec, basis, method: 100000.0 if method == "CCSDTQ" else 0.0)
    monkeypatch.setattr(post_ccsd_t_stage, "scratch_free_mb", lambda: scratch_free_mb)
    monkeypatch.setattr(post_ccsd_t_stage, "cc_limit_group",
                        lambda spec, basis, method: post_ccsd_t_stage.OUT_OF_CORE_LIMIT_GROUP
                        if method == "CCSDTQ" else None)
    reqs = [request("N", 0, 1), request("A", -1, 0)]
    recs = {f"{j}-{i}": {"job_id": j, "request_id": f"{j}-{i}", "status": "COMPLETED", "r_angstrom": r,
                         "high_level_checkpoint_path": f"/hl/{j}{r}"} for j in "NA" for i, r in enumerate((0.9, 1.0))}
    return run_post_ccsd_t_stage(electronic_ea=ea_record(2.0), requests=reqs, point_records=recs,
                                 refinement_dir=tmp_path / "none", output_dir=tmp_path / "pc",
                                 policy=PostCCSDTStagePolicy(q_memory_limit_mb=limit), point_fn=fake_point)


def test_triples_tz_central_and_quadruples_in_first_affordable_basis(tmp_path, monkeypatch):
    res = run(tmp_path, monkeypatch, limit=60000)
    assert res["status"] == "CLEARED" and res["quadruples_basis"] == "aug-cc-pvdz"
    assert abs(res["delta_t_ev"] + 0.006) < 1e-9 and abs(res["delta_t_bound_ev"] - 0.002) < 1e-9
    assert abs(res["delta_q_ev"] - 0.010) < 1e-9 and abs(res["delta_post_ev"] - 0.004) < 1e-9
    assert res["quadruples_diffuse_allowance_ev"] == 0.0
    assert abs(res["delta_q_bound_ev"] - 0.005) < 1e-9
    res = run(tmp_path, monkeypatch, limit=20000)  # aug-cc-pVDZ too large: cc-pVDZ
    assert res["quadruples_basis"] == "cc-pvdz" and abs(res["delta_q_ev"] - 0.008) < 1e-9
    # no diffuse functions: the bound gains the measured aug-cc-pVDZ - cc-pVDZ difference
    allowance = post_ccsd_t_stage.Q_NO_DIFFUSE_ALLOWANCE_EV
    assert res["quadruples_diffuse_allowance_ev"] == allowance
    assert abs(res["delta_q_bound_ev"] - (0.004 + allowance)) < 1e-9
    assert abs(res["bound_ev"] - (0.002 + 0.004 + allowance)) < 1e-9


def test_diffuse_basis_names():
    assert post_ccsd_t_stage.has_diffuse("aug-cc-pvdz") and post_ccsd_t_stage.has_diffuse("AUG-cc-pVDZ-PP")
    assert post_ccsd_t_stage.has_diffuse("d-aug-cc-pvdz")
    assert not post_ccsd_t_stage.has_diffuse("cc-pvdz") and not post_ccsd_t_stage.has_diffuse("cc-pvdz-pp")


def test_unaffordable_quadruples_apply_triples_with_a_quadruples_allowance(tmp_path, monkeypatch):
    res = run(tmp_path, monkeypatch, limit=5000)
    assert res["status"] == "CLEARED_WITH_QUADRUPLES_ALLOWANCE" and not res["quadruples_computed"]
    assert abs(res["delta_post_ev"] + 0.006) < 1e-9  # Delta_T applied
    allowance = post_ccsd_t_stage.Q_ALLOWANCE_MIN_EV  # |Delta_T| = 6 meV is below the floor
    assert abs(res["quadruples_allowance_ev"] - allowance) < 1e-12
    assert abs(res["bound_ev"] - (0.002 + allowance)) < 1e-9
    cbs = {"binding_at_cbs": "BOUND", "status": "CLEARED", "ea_cbs_ev": 1.0, "ea_cbs_lower_ev": 0.99,
           "ea_cbs_upper_ev": 1.01}
    nuc = {"status": "CLEARED", "delta_zpe_ev": 0.0, "delta_zpe_half_width_ev": 0.0, "anion_vibrational_binding": "BOUND"}
    out = assemble_adiabatic_ea(cbs=cbs, nuclear=nuc, post_ccsd_t=res)
    assert out["includes_post_ccsd_t"] and abs(out["ea0_ev"] - 0.994) < 1e-9
    assert abs(out["ea0_upper_ev"] - (1.01 - 0.006 + 0.002 + allowance)) < 1e-9
    assert out["post_ccsd_t_quadruples_allowance_ev"] == res["quadruples_allowance_ev"]


def test_quadruples_need_ram_and_scratch(tmp_path, monkeypatch):
    # RAM fits, but the DIIS file needs more than half of the free scratch.
    res = run(tmp_path, monkeypatch, limit=60000, scratch_free_mb=150000.0)
    assert res["status"] == "CLEARED_WITH_QUADRUPLES_ALLOWANCE"
    # A larger machine: the environment raises the RAM limit, exact CCSDTQ runs.
    monkeypatch.setenv(post_ccsd_t_stage.Q_MEMORY_LIMIT_ENV, "60000")
    res = run(tmp_path, monkeypatch, limit=5000)
    assert res["status"] == "CLEARED" and res["quadruples_computed"]


def test_point_files_are_distinct_per_method():
    from openea_benchmark.post_ccsd_t_stage import METHOD_TOKEN, METHODS
    assert len({METHOD_TOKEN[m] for m in METHODS}) == len(METHODS)


def test_reservations_follow_the_estimates(tmp_path, monkeypatch):
    seen = {}

    def fake_parallel(fn, jobs, reserve_mb=None, limit_groups=None, hold_reservations=False):
        seen["reserves"] = list(reserve_mb)
        return [fn(*job) for job in jobs]

    monkeypatch.setattr(post_ccsd_t_stage, "run_parallel", fake_parallel)
    run(tmp_path, monkeypatch, limit=60000)
    assert 50000.0 in seen["reserves"] and 1000.0 in seen["reserves"]


def test_large_tasks_run_one_at_a_time(tmp_path, monkeypatch):
    import threading
    import time
    from openea_benchmark import point_pool
    monkeypatch.setattr(point_pool, "_SLOT_DIR", str(tmp_path))
    monkeypatch.setattr(point_pool, "_N_SLOTS", 4)
    monkeypatch.setattr(point_pool, "_MEM_LIMIT_FRAC", 0.0)
    active, peak = [0], [0]

    def task():
        with point_pool.core_slot(reserve_mb=150000):
            active[0] += 1
            peak[0] = max(peak[0], active[0])
            time.sleep(0.2)
            active[0] -= 1

    threads = [threading.Thread(target=task) for _ in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert peak[0] == 1


def test_pending_large_reservation_is_counted_by_others(tmp_path, monkeypatch):
    import fcntl
    import os
    from openea_benchmark import point_pool
    monkeypatch.setattr(point_pool, "_SLOT_DIR", str(tmp_path))
    assert point_pool._pending_big_reservation_mb() == 0.0
    fh = open(os.path.join(tmp_path, "bigmem.lock"), "a+")
    fcntl.flock(fh, fcntl.LOCK_EX)
    fh.write("host/ns 1 150000 PENDING\n")
    fh.flush()
    assert point_pool._pending_big_reservation_mb() == 150000.0
    fh.seek(0); fh.truncate(); fh.write("host/ns 1 150000 ADMITTED\n"); fh.flush()
    assert point_pool._pending_big_reservation_mb() == 0.0  # admitted: counted via its slot instead
    fcntl.flock(fh, fcntl.LOCK_UN)
    fh.close()


def test_cached_points_reserve_no_memory(tmp_path, monkeypatch):
    """A completed point only reads its file; it must not queue for the large-task slot."""
    import json
    seen = []
    base_point = fake_point

    def fake_parallel(fn, jobs, reserve_mb=None, limit_groups=None, hold_reservations=False):
        seen.append(list(reserve_mb))
        return [fn(*job) for job in jobs]

    def writing_point(spec, basis, method, policy, path):
        rec = dict(base_point(spec, basis, method, policy, path),
                   signature=post_ccsd_t_stage._point_signature(spec, basis, method, policy))
        from pathlib import Path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(rec))
        return rec

    monkeypatch.setattr(post_ccsd_t_stage, "run_parallel", fake_parallel)
    import test_post_ccsd_t_stage_v1 as me
    monkeypatch.setattr(me, "fake_point", writing_point)
    run(tmp_path, monkeypatch, limit=60000)
    assert max(seen[0]) == 50000.0
    run(tmp_path, monkeypatch, limit=60000)  # second pass: every point cached
    assert seen[1] and all(x == 0.0 for x in seen[1])


def test_reference_orbitals_are_handed_to_ccpy_in_occupation_order():
    # CH- 3Sigma- in aug-cc-pVTZ: a ROHF virtual lay between the doubly and the singly occupied
    # orbitals and CCpy failed with "Integrals don't match mean field energy" (2026-10-26).
    import numpy as np
    from pyscf import gto, scf
    from ccpy.drivers.driver import Driver
    from openea_benchmark.post_ccsd_t_stage import occupation_ordered

    mol = gto.M(atom="C 0 0 0; H 0 0 1.13", basis="cc-pvdz", charge=-1, spin=2, symmetry="C1", verbose=0)
    mf = scf.ROHF(mol)
    mf.kernel()
    energy = mf.e_tot
    swap = np.arange(mol.nao_nr())
    swap[[3, 5]] = swap[[5, 3]]              # a virtual in front of the singly occupied orbitals
    mf.mo_coeff, mf.mo_occ, mf.mo_energy = mf.mo_coeff[:, swap], mf.mo_occ[swap], mf.mo_energy[swap]
    assert abs(mf.energy_tot() - energy) < 1e-8
    with pytest.raises(RuntimeError, match="Integrals don't match"):
        Driver.from_pyscf(mf, nfrozen=1)
    ordered = occupation_ordered(mf)
    assert list(ordered.mo_occ[:6]) == [2, 2, 2, 1, 1, 0]
    driver = Driver.from_pyscf(ordered, nfrozen=1)
    assert abs(driver.system.reference_energy - energy) < 1e-6


def test_no_affordable_triples_gives_a_post_cc_allowance(tmp_path, monkeypatch):
    res = run(tmp_path, monkeypatch, limit=500)  # even CCSDT (1000 MB) does not fit
    assert res["status"] == "CLEARED_WITH_POST_CC_ALLOWANCE" and res["delta_post_ev"] == 0.0
    assert res["bound_ev"] == post_ccsd_t_stage.POST_CC_ALLOWANCE_EV
    cbs = {"binding_at_cbs": "BOUND", "status": "CLEARED", "ea_cbs_ev": 1.0, "ea_cbs_lower_ev": 0.99,
           "ea_cbs_upper_ev": 1.01}
    nuc = {"status": "CLEARED", "delta_zpe_ev": 0.0, "delta_zpe_half_width_ev": 0.0, "anion_vibrational_binding": "BOUND"}
    out = assemble_adiabatic_ea(cbs=cbs, nuclear=nuc, post_ccsd_t=res)
    assert out["includes_post_ccsd_t"] and out["post_ccsd_t_triples_computed"] is False
    assert abs(out["ea0_upper_ev"] - (1.01 + post_ccsd_t_stage.POST_CC_ALLOWANCE_EV)) < 1e-12


def test_unaffordable_quadruples_use_fno_ccsdtq_increment(tmp_path, monkeypatch):
    monkeypatch.setattr(post_ccsd_t_stage, "FNO_BEYOND_TRIPLES", True)
    monkeypatch.setattr(post_ccsd_t_stage, "_fno_increment",
                        lambda states, policy, out: {"status": "COMPLETED", "delta_ev": -0.008, "half_width_ev": 0.001})
    res = run(tmp_path, monkeypatch, limit=5000)
    assert res["status"] == "CLEARED_FNO_CCSDTQ" and res["beyond_triples_computed"]
    assert abs(res["delta_post_ev"] - (-0.006 - 0.008)) < 1e-9
    assert abs(res["bound_ev"] - (0.002 + 0.001 + 0.5 * 0.008 + post_ccsd_t_stage.Q_NO_DIFFUSE_ALLOWANCE_EV)) < 1e-9
    cbs = {"binding_at_cbs": "BOUND", "status": "CLEARED", "ea_cbs_ev": 1.0, "ea_cbs_lower_ev": 0.99,
           "ea_cbs_upper_ev": 1.01}
    nuc = {"status": "CLEARED", "delta_zpe_ev": 0.0, "delta_zpe_half_width_ev": 0.0, "anion_vibrational_binding": "BOUND"}
    out = assemble_adiabatic_ea(cbs=cbs, nuclear=nuc, post_ccsd_t=res)
    assert out["includes_post_ccsd_t"] and abs(out["ea0_ev"] - 0.986) < 1e-9


def test_failed_fno_falls_back_to_the_allowance(tmp_path, monkeypatch):
    monkeypatch.setattr(post_ccsd_t_stage, "FNO_BEYOND_TRIPLES", True)
    monkeypatch.setattr(post_ccsd_t_stage, "_fno_increment",
                        lambda states, policy, out: {"status": "FAILED", "reason": "x"})
    res = run(tmp_path, monkeypatch, limit=5000)
    assert res["status"] == "CLEARED_WITH_QUADRUPLES_ALLOWANCE" and res["fno_ccsdtq"]["status"] == "FAILED"


def test_points_stay_in_core_with_a_smaller_diis_before_going_out_of_core(monkeypatch):
    monkeypatch.delenv("OPENEA_DIIS_OUT_OF_CORE", raising=False)
    monkeypatch.setenv(post_ccsd_t_stage.IN_CORE_LIMIT_ENV, "120000")
    mode = post_ccsd_t_stage.cc_mode
    assert mode(2000.0) == (False, 6)        # VO anion aDZ CCSDT: 16 copies = 32 GB
    assert mode(7000.0) == (False, 6)        # 112 GB
    assert mode(10000.0) == (False, 3)       # 160 GB with 6 vectors, 100 GB with 3
    assert mode(13000.0) == (True, 6)        # 130 GB even with 3 vectors: DIIS on scratch
    monkeypatch.setenv("OPENEA_DIIS_OUT_OF_CORE", "1")
    assert mode(1.0) == (True, 6)
    monkeypatch.setenv("OPENEA_DIIS_OUT_OF_CORE", "0")
    assert mode(1.0e6) == (False, 6)


def test_storage_estimate_follows_the_mode(monkeypatch):
    monkeypatch.delenv("OPENEA_DIIS_OUT_OF_CORE", raising=False)
    monkeypatch.setenv(post_ccsd_t_stage.IN_CORE_LIMIT_ENV, "120000")
    for one, copies, scratch in ((2000.0, 16, 0.0), (10000.0, 10, 0.0), (13000.0, 6, 12 * 13000.0)):
        monkeypatch.setattr(post_ccsd_t_stage, "_amplitude_mb", lambda spec, basis, method, one=one: one)
        assert post_ccsd_t_stage.cc_storage_mb(None, "b", "CCSDT") == copies * one
        assert post_ccsd_t_stage.cc_scratch_need_mb(None, "b", "CCSDT") == scratch
        group = post_ccsd_t_stage.cc_limit_group(None, "b", "CCSDT")
        assert (group is not None) == (scratch > 0.0)


def test_out_of_core_points_carry_the_limit_group(tmp_path, monkeypatch):
    seen = {}

    def fake_parallel(fn, jobs, reserve_mb=None, limit_groups=None, hold_reservations=False):
        seen["groups"] = list(limit_groups)
        seen["methods"] = [job[2] for job in jobs]
        return [fn(*job) for job in jobs]

    monkeypatch.setattr(post_ccsd_t_stage, "run_parallel", fake_parallel)
    run(tmp_path, monkeypatch, limit=60000)
    for method, group in zip(seen["methods"], seen["groups"]):
        assert (group == post_ccsd_t_stage.OUT_OF_CORE_LIMIT_GROUP) == (method == "CCSDTQ")


def test_stalled_reference_scf_is_rescued_by_second_order_scf():
    class FakeSOSCF:
        def __init__(self, parent):
            self.parent, self.converged = parent, False
        def kernel(self, mo, occ):
            self.converged = self.parent.newton_converges
            self.mo_coeff, self.mo_occ, self.mo_energy, self.e_tot = "C", "occ", "eps", -1.0

    class FakeSCF:
        max_cycle = 200
        def __init__(self, diis, newton):
            self.diis_converges, self.newton_converges, self.converged = diis, newton, False
            self.mo_coeff = self.mo_occ = None
        def kernel(self, dm0=None):
            self.converged = self.diis_converges
        def newton(self):
            return FakeSOSCF(self)

    rescue = post_ccsd_t_stage._kernel_with_soscf_rescue
    mf = FakeSCF(True, False)
    assert rescue(mf, None, 1e-10) is False and mf.converged
    mf = FakeSCF(False, True)
    assert rescue(mf, None, 1e-10) is True and mf.converged and mf.e_tot == -1.0 and mf.mo_coeff == "C"
    mf = FakeSCF(False, False)
    assert rescue(mf, None, 1e-10) is False and not mf.converged


def _ch_point(tmp_path, monkeypatch, spin_2s):
    """One CCSD(T) point of CH (2Pi radical or CH+ closed shell) in cc-pVDZ through _run_point."""
    from pyscf import gto, scf
    from openea_benchmark.adaptive.core_valence_runner import CoreValenceStateSpec
    from openea_benchmark.post_ccsd_t_stage import _run_point, occupation_ordered

    charge = 0 if spin_2s else 1
    mol = gto.M(atom="C 0 0 0; H 0 0 1.12", basis="cc-pvdz", charge=charge, spin=spin_2s, symmetry="C1", verbose=0)
    mf = (scf.ROHF(mol) if spin_2s else scf.RHF(mol))
    mf.conv_tol = 1e-10
    mf.kernel()
    monkeypatch.setattr(post_ccsd_t_stage, "reference_c1", lambda spec, basis, policy: occupation_ordered(mf))
    spec = CoreValenceStateSpec(role="neutral", system="CH", atoms=("C", "H"), charge=charge, spin_2s=spin_2s,
                                state_id="X", r_angstrom=1.12, source_root_id="X", source_checkpoint_path="none",
                                state_identity_validated=True)
    rec = _run_point(spec, "cc-pvdz", "CCSD(T)", PostCCSDTStagePolicy(), str(tmp_path / f"p{spin_2s}.json"))
    return mf, rec


def test_open_shell_ccsd_t_baseline_is_the_semicanonical_one_of_the_cbs_layer(tmp_path, monkeypatch):
    # CBS (Stage 3) and the post-CC baseline must contain the same (T): CCSDT - CCSD(T) then
    # removes exactly it.  CCpy's ROHF (T) differs from the semicanonical one by ~6 microhartree here.
    from pyscf import cc
    mf, rec = _ch_point(tmp_path, monkeypatch, 1)
    assert rec["status"] == "COMPLETED" and rec["triples_baseline"] == "SEMICANONICAL_UHF_FROM_ROHF"
    umf = mf.to_uhf()
    umf.mo_energy, umf.mo_coeff = umf.canonicalize(umf.mo_coeff, umf.mo_occ)
    ref = cc.CCSD(umf)
    ref.frozen, ref.conv_tol, ref.conv_tol_normt = 1, 1e-10, 1e-8
    ref.kernel()
    stage3_total = mf.e_tot + ref.e_corr + ref.ccsd_t()
    assert rec["total_hartree"] == pytest.approx(stage3_total, abs=1e-7)
    assert abs(rec["total_hartree"] - (mf.e_tot + rec["ccpy_ccsd_t_correlation_hartree"])) > 1e-6
    assert abs(rec["ccsd_invariance_difference_hartree"]) < post_ccsd_t_stage.CCSD_INVARIANCE_TOLERANCE_HARTREE


def test_closed_shell_ccsd_t_points_keep_their_cache_signature(tmp_path, monkeypatch):
    from openea_benchmark.adaptive.core_valence_runner import CoreValenceStateSpec
    from openea_benchmark.post_ccsd_t_stage import _point_signature
    _, rec = _ch_point(tmp_path, monkeypatch, 0)
    assert rec["status"] == "COMPLETED" and "triples_baseline" not in rec
    base = dict(role="neutral", system="CH", atoms=("C", "H"), charge=0, state_id="X", r_angstrom=1.12,
                source_root_id="X", source_checkpoint_path="none", state_identity_validated=True)
    closed, open_ = CoreValenceStateSpec(spin_2s=0, **base), CoreValenceStateSpec(spin_2s=1, **base)
    policy = PostCCSDTStagePolicy()
    # the open-shell CCSD(T) signature changed (old CCpy-(T) points are recomputed), CCSDT did not
    assert "triples_baseline" not in str(_point_signature(closed, "cc-pvdz", "CCSD(T)", policy))
    import hashlib, json
    from dataclasses import asdict
    legacy = lambda spec, method: hashlib.sha256(json.dumps(
        {"state": asdict(spec), "basis": "cc-pvdz", "method": method, "policy": asdict(policy)},
        sort_keys=True, default=str).encode()).hexdigest()
    assert _point_signature(closed, "cc-pvdz", "CCSD(T)", policy) == legacy(closed, "CCSD(T)")
    assert _point_signature(open_, "cc-pvdz", "CCSDT", policy) == legacy(open_, "CCSDT")
    assert _point_signature(open_, "cc-pvdz", "CCSD(T)", policy) != legacy(open_, "CCSD(T)")


def test_symmetric_ccp_ccsdt_point_equals_the_dense_point(tmp_path, monkeypatch):
    """CCSDT through _run_point: CC(P) over the symmetry-allowed triples on the symmetry-block
    reference equals CCpy's dense ccsdt on the C1 reference (same determinant); the point
    stays in core and records its triples count."""
    import dataclasses
    from pyscf import gto, scf
    from openea_benchmark.adaptive.core_valence_runner import CoreValenceStateSpec
    from openea_benchmark.post_ccsd_t_stage import _run_point, occupation_ordered

    def ref(symmetry):
        mol = gto.M(atom="C 0 0 0; N 0 0 1.1718", basis="cc-pvdz", spin=1, symmetry=symmetry, verbose=0)
        mf = scf.ROHF(mol)
        mf.conv_tol = 1e-11
        mf.kernel()
        return occupation_ordered(mf)
    sym, c1 = ref("C2v"), ref("C1")
    assert sym.e_tot == pytest.approx(c1.e_tot, abs=1e-8)
    monkeypatch.setattr(post_ccsd_t_stage, "reference_c1", lambda spec, basis, policy: c1)
    monkeypatch.setattr(post_ccsd_t_stage, "reference_symmetric", lambda spec, basis, policy: sym)
    spec = CoreValenceStateSpec(role="neutral", system="CN", atoms=("C", "N"), charge=0, spin_2s=1,
                                state_id="X", r_angstrom=1.1718, source_root_id="X", source_checkpoint_path="none",
                                state_identity_validated=True)
    spec = dataclasses.replace(spec, point_group="C2v") if hasattr(spec, "point_group") else spec
    assert post_ccsd_t_stage.symmetric_ccp_applies(spec, "CCSDT")
    packed = _run_point(spec, "cc-pvdz", "CCSDT", PostCCSDTStagePolicy(), str(tmp_path / "ccp.json"))
    monkeypatch.setattr(post_ccsd_t_stage, "CCSDT_ALGORITHM", "dense")
    dense = _run_point(spec, "cc-pvdz", "CCSDT", PostCCSDTStagePolicy(), str(tmp_path / "dense.json"))
    assert packed["status"] == dense["status"] == "COMPLETED", (packed, dense)
    assert packed["triples_algorithm"] == "CCP_SYMMETRY_ALLOWED_TRIPLES" and "triples_algorithm" not in dense
    assert packed["total_hartree"] == pytest.approx(dense["total_hartree"], abs=1e-8)
    assert packed["diis_out_of_core"] is False
    monkeypatch.setattr(post_ccsd_t_stage, "CCSDT_ALGORITHM", "symmetric_ccp")
    assert post_ccsd_t_stage.cc_limit_group(spec, "cc-pvdz", "CCSDT") is None


def test_unconverged_cc_point_fails_closed(tmp_path, monkeypatch):
    from openea_benchmark.post_ccsd_t_stage import PostCCSDTStagePolicy as P
    mf, rec = _ch_point_policy(tmp_path, monkeypatch, P(cc_max_iterations=2))
    assert rec["status"] == "FAILED" and "without convergence" in rec["error"]


def _ch_point_policy(tmp_path, monkeypatch, policy):
    from pyscf import gto, scf
    from openea_benchmark.adaptive.core_valence_runner import CoreValenceStateSpec
    from openea_benchmark.post_ccsd_t_stage import _run_point, occupation_ordered
    mol = gto.M(atom="C 0 0 0; H 0 0 1.12", basis="cc-pvdz", spin=1, symmetry="C1", verbose=0)
    mf = scf.ROHF(mol)
    mf.conv_tol = 1e-10
    mf.kernel()
    monkeypatch.setattr(post_ccsd_t_stage, "reference_c1", lambda spec, basis, policy: occupation_ordered(mf))
    spec = CoreValenceStateSpec(role="neutral", system="CH", atoms=("C", "H"), charge=0, spin_2s=1,
                                state_id="X", r_angstrom=1.12, source_root_id="X", source_checkpoint_path="none",
                                state_identity_validated=True)
    return mf, _run_point(spec, "cc-pvdz", "CCSDT", policy, str(tmp_path / "unconv.json"))


def test_symcc_ccsdtq_point_equals_the_dense_point(tmp_path, monkeypatch):
    """CCSDTQ through _run_point on symmetry-blocked tensors (symcc) equals CCpy's dense ccsdtq."""
    pytest.importorskip("symcc.cc")
    import dataclasses
    from pyscf import gto, scf
    from openea_benchmark.adaptive.core_valence_runner import CoreValenceStateSpec
    from openea_benchmark.post_ccsd_t_stage import _run_point, occupation_ordered

    def ref(symmetry):
        mol = gto.M(atom="O 0 0 0; H 0 0 0.97", basis="6-31g", spin=1, symmetry=symmetry, verbose=0)
        mf = scf.ROHF(mol)
        mf.conv_tol = 1e-11
        mf.kernel()
        return occupation_ordered(mf)
    sym, c1 = ref("C2v"), ref("C1")
    monkeypatch.setattr(post_ccsd_t_stage, "reference_c1", lambda spec, basis, policy: c1)
    monkeypatch.setattr(post_ccsd_t_stage, "reference_symmetric", lambda spec, basis, policy: sym)
    spec = CoreValenceStateSpec(role="neutral", system="OH", atoms=("O", "H"), charge=0, spin_2s=1,
                                state_id="X", r_angstrom=0.97, source_root_id="X", source_checkpoint_path="none",
                                state_identity_validated=True)
    spec = dataclasses.replace(spec, point_group="C2v")
    assert post_ccsd_t_stage.symcc_applies(spec, "CCSDTQ")
    blocked = _run_point(spec, "6-31g", "CCSDTQ", PostCCSDTStagePolicy(), str(tmp_path / "symcc.json"))
    monkeypatch.setattr(post_ccsd_t_stage, "CCSDTQ_ALGORITHM", "dense")
    dense = _run_point(spec, "6-31g", "CCSDTQ", PostCCSDTStagePolicy(), str(tmp_path / "dense.json"))
    assert blocked["status"] == dense["status"] == "COMPLETED", (blocked, dense)
    assert blocked["quadruples_algorithm"] == "SYMCC_SYMMETRY_BLOCKED"
    assert blocked["total_hartree"] == pytest.approx(dense["total_hartree"], abs=1e-8)
