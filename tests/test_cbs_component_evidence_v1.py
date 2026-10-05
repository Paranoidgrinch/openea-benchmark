import pickle
from types import SimpleNamespace
from openea_benchmark.adaptive.stage3_execution import PointExecutionStatus, Stage3PointResult
from openea_benchmark.attachment.cbs_component_evidence import (
    ComponentEvidenceStatus, extract_reference_geometry, load_reference_geometries,
    make_cbs_single_point_request, point_from_stage3_result, summarize_component_evidence,
)

def ev(v): return SimpleNamespace(value=v)

def loop(role, r, basis="d-aug-cc-pv5z"):
    rid = f"{role}-min"
    return SimpleNamespace(
        results=(SimpleNamespace(request_id=rid,status=ev("COMPLETED"),basis=basis,r_angstrom=r),),
        final_pec=SimpleNamespace(minimum_scout=SimpleNamespace(
            candidates=(SimpleNamespace(canonical_request_id=rid),)
        )),
    )

def result(role,basis,r,complete=True):
    spin = 1 if role=="neutral" else 0
    return Stage3PointResult(
        request_id=f"{role}-{basis}", job_id="j",
        status=PointExecutionStatus.COMPLETED if complete else PointExecutionStatus.CCSD_NOT_CONVERGED,
        system="OH", charge=0 if spin else -1, spin_2s=spin, component_id="c",
        r_angstrom=r, basis=basis, source_root_id="r", source_checkpoint_path="/tmp/x",
        scf_reference="ROHF" if spin else "RHF", cc_reference="X", pyscf_version="2.14",
        scf_converged=True, scf_energy_hartree=-75.4, s2=0.75 if spin else 0.0,
        multiplicity=2.0 if spin else 1.0, internal_stable=True, external_stable=None,
        external_stability_available=False, semicanonicalization="X", cc_class="CCSD",
        ccsd_converged=complete, ccsd_correlation_hartree=-0.3 if complete else None,
        ccsd_total_hartree=-75.7 if complete else None,
        triples_correction_hartree=-0.01 if complete else None,
        ccsd_t_total_hartree=-75.71 if complete else None,
        t1_diagnostic=None,t1_diagnostic_definition=None,error_type=None,error_message=None,
    )

def test_reference_geometry(tmp_path):
    g=extract_reference_geometry(loop("neutral",0.97),role="neutral",source_loop_path=tmp_path/"x")
    assert g.r_angstrom==0.97

def test_ambiguous_reference_fails(tmp_path):
    obj=SimpleNamespace(
        results=(SimpleNamespace(request_id="a",status=ev("COMPLETED"),basis="d-aug-cc-pv5z",r_angstrom=.96),
                 SimpleNamespace(request_id="b",status=ev("COMPLETED"),basis="d-aug-cc-pv5z",r_angstrom=.98)),
        final_pec=SimpleNamespace(minimum_scout=SimpleNamespace(
            candidates=(SimpleNamespace(canonical_request_id="a"),SimpleNamespace(canonical_request_id="b"))
        )))
    try: extract_reference_geometry(obj,role="anion",source_loop_path=tmp_path/"x")
    except ValueError as e: assert "ambiguous" in str(e)
    else: raise AssertionError

def test_load_reference_geometries(tmp_path):
    stage=tmp_path/"run"/"basis_stage_checkpoints"/"d-aug-cc-pv5z"; stage.mkdir(parents=True)
    for role,r in (("neutral",.97),("anion",.96)):
        with (stage/f"{role}_loop.pkl").open("wb") as h: pickle.dump(loop(role,r),h)
    n,a=load_reference_geometries(run_dir=tmp_path/"run")
    assert (n.r_angstrom,a.r_angstrom)==(.97,.96)

def test_request_fixed_geometry(tmp_path):
    q=make_cbs_single_point_request(role="neutral",basis="aug-cc-pvqz",cardinal_number=4,
                                    r_angstrom=.97,source_checkpoint_path=tmp_path/"x")
    assert q.r_angstrom==.97 and q.scf_reference=="ROHF"

def test_complete_result_reusable():
    p=point_from_stage3_result(result("neutral","aug-cc-pvqz",.97),role="neutral",cardinal_number=4)
    assert p.reusable and p.triples_correction_hartree==-0.01

def test_summary_ready():
    points=[]
    for basis,x in (("aug-cc-pvqz",4),("aug-cc-pv5z",5)):
        points += [
            point_from_stage3_result(result("neutral",basis,.97),role="neutral",cardinal_number=x),
            point_from_stage3_result(result("anion",basis,.96),role="anion",cardinal_number=x),
        ]
    e=summarize_component_evidence(reference_basis="d-aug-cc-pv5z",reference_geometries=(),
                                   points=points,required_bases=("aug-cc-pvqz","aug-cc-pv5z"))
    assert e.status is ComponentEvidenceStatus.READY and not e.missing_points

def test_missing_point_blocks():
    p=point_from_stage3_result(result("neutral","aug-cc-pvqz",.97),role="neutral",cardinal_number=4)
    e=summarize_component_evidence(reference_basis="d-aug-cc-pv5z",reference_geometries=(),
                                   points=(p,),required_bases=("aug-cc-pvqz","aug-cc-pv5z"))
    assert e.status is ComponentEvidenceStatus.PARTIAL
