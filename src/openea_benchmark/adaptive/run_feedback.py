from __future__ import annotations
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json, os, sys
from pathlib import Path
from typing import Any

class ProgressStatus:
    RUNNING='RUNNING'; COMPLETED='COMPLETED'; BLOCKED='BLOCKED'; FAILED='FAILED'

@dataclass
class WorkflowProgress:
    run_id:str; workflow:str; status:str; current_step:str
    completed_steps:list[str]=field(default_factory=list)
    next_steps:list[str]=field(default_factory=list)
    details:dict[str,Any]=field(default_factory=dict)
    updated_at_utc:str=''

class ProgressReporter:
    def __init__(self, *, run_id:str, workflow:str, status_file:Path):
        if not run_id.strip() or not workflow.strip(): raise ValueError('run_id/workflow must be non-empty')
        self.status_file=Path(status_file); self.status_file.parent.mkdir(parents=True, exist_ok=True)
        self.state=WorkflowProgress(run_id,workflow,ProgressStatus.RUNNING,'INITIALIZING')
        self._publish()
    def update(self, *, current_step:str, completed_steps=None, next_steps=None, details=None, status:str=ProgressStatus.RUNNING):
        self.state.status=status; self.state.current_step=current_step
        if completed_steps is not None: self.state.completed_steps=list(completed_steps)
        if next_steps is not None: self.state.next_steps=list(next_steps)
        if details is not None: self.state.details=dict(details)
        self._publish()
    def completed(self, **kw): self.update(status=ProgressStatus.COMPLETED, **kw)
    def blocked(self, **kw): self.update(status=ProgressStatus.BLOCKED, **kw)
    def failed(self, **kw): self.update(status=ProgressStatus.FAILED, **kw)
    def _publish(self):
        self.state.updated_at_utc=datetime.now(timezone.utc).isoformat()
        tmp=self.status_file.with_suffix(self.status_file.suffix+'.tmp')
        tmp.write_text(json.dumps(asdict(self.state),indent=2,sort_keys=True)+'\n',encoding='utf-8'); os.replace(tmp,self.status_file)
        completed=', '.join(self.state.completed_steps) or '-'; nxt=' -> '.join(self.state.next_steps) or '-'
        print(f'[OPENEA][{self.state.status}] step={self.state.current_step} | completed={completed} | next={nxt}',file=sys.stderr,flush=True)
        if self.state.details:
            print('[OPENEA][DETAIL] '+json.dumps(self.state.details,sort_keys=True,separators=(',',':')),file=sys.stderr,flush=True)
