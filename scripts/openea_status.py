#!/usr/bin/env python3
import argparse, json
from pathlib import Path
p=argparse.ArgumentParser(); p.add_argument('run_id',nargs='?'); p.add_argument('--run-root',default='runs'); a=p.parse_args()
run_dir=Path(a.run_root)/(a.run_id or 'latest'); status=run_dir/'status.json'
if not status.exists(): raise SystemExit(f'no status file: {status}')
x=json.loads(status.read_text())
print(f"run       : {x['run_id']}"); print(f"workflow  : {x['workflow']}"); print(f"status    : {x['status']}"); print(f"current   : {x['current_step']}"); print(f"updated   : {x['updated_at_utc']}")
print('completed :'); [print(f'  - {v}') for v in x.get('completed_steps',[])]
print('next      :'); [print(f'  - {v}') for v in x.get('next_steps',[])]
if x.get('details'): print('details   :'); print(json.dumps(x['details'],indent=2,sort_keys=True))
