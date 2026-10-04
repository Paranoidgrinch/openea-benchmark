from __future__ import annotations
from dataclasses import dataclass
import json, os
from pathlib import Path
from openea_benchmark.attachment.basis_convergence import EAIntervalEV, ElectronicEABasisPoint

@dataclass(frozen=True)
class BasisPointCheckpoint:
    signature:str; source:str; points:tuple[ElectronicEABasisPoint,...]

def save_basis_checkpoint(path:Path, *, signature:str, source:str, points:tuple[ElectronicEABasisPoint,...]):
    if not signature.strip(): raise ValueError('checkpoint signature must be non-empty')
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    payload={'signature':signature,'source':source,'points':[{
        'basis_name':p.basis_name,'cardinal_number':p.cardinal_number,'augmentation_level':p.augmentation_level,
        'method_signature':p.method_signature,'decision_status':p.decision_status,'evidence_quality':p.evidence_quality,
        'is_production_ea':p.is_production_ea,'ea':{'lower_ev':p.ea.lower_ev,'central_ev':p.ea.central_ev,'upper_ev':p.ea.upper_ev}
    } for p in points]}
    tmp=path.with_suffix(path.suffix+'.tmp'); tmp.write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n',encoding='utf-8'); os.replace(tmp,path)

def load_basis_checkpoint(path:Path, *, expected_signature:str):
    path=Path(path)
    if not path.exists(): return None
    payload=json.loads(path.read_text(encoding='utf-8'))
    if payload.get('signature')!=expected_signature: raise ValueError('basis checkpoint signature does not match this workflow configuration')
    pts=[]
    for raw in payload.get('points',[]):
        ea=raw['ea']; pts.append(ElectronicEABasisPoint(
            basis_name=raw['basis_name'], cardinal_number=int(raw['cardinal_number']), augmentation_level=int(raw['augmentation_level']),
            method_signature=raw['method_signature'], ea=EAIntervalEV(float(ea['lower_ev']),float(ea['central_ev']),float(ea['upper_ev'])),
            decision_status=raw.get('decision_status','BOUND'), evidence_quality=raw.get('evidence_quality','CONVERGENCE_ESTIMATED'),
            is_production_ea=bool(raw.get('is_production_ea',False))))
    return BasisPointCheckpoint(payload['signature'],payload.get('source','UNKNOWN'),tuple(pts))
