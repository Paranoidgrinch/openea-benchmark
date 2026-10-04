#!/usr/bin/env python3
import argparse, json
from datetime import datetime, timezone
from pathlib import Path

def fmt_seconds(seconds):
    seconds=max(0,int(seconds)); days,rem=divmod(seconds,86400); hours,rem=divmod(rem,3600); minutes,sec=divmod(rem,60)
    parts=[]
    if days: parts.append(f'{days}d')
    if days or hours: parts.append(f'{hours}h')
    if days or hours or minutes: parts.append(f'{minutes}m')
    parts.append(f'{sec}s')
    return ' '.join(parts)

p=argparse.ArgumentParser(); p.add_argument('run_id',nargs='?'); p.add_argument('--run-root',default='runs'); a=p.parse_args()
run_dir=Path(a.run_root)/(a.run_id or 'latest'); status=run_dir/'status.json'
if not status.exists(): raise SystemExit(f'no status file: {status}')
x=json.loads(status.read_text()); now=datetime.now(timezone.utc)
print(f"run       : {x['run_id']}"); print(f"workflow  : {x['workflow']}"); print(f"status    : {x['status']}"); print(f"current   : {x['current_step']}"); print(f"updated   : {x['updated_at_utc']}")
if x.get('started_at_utc'):
    print(f"elapsed   : {fmt_seconds((now-datetime.fromisoformat(x['started_at_utc'])).total_seconds())}")
print(f"step age  : {fmt_seconds((now-datetime.fromisoformat(x['updated_at_utc'])).total_seconds())}")
print('completed :'); [print(f'  - {v}') for v in x.get('completed_steps',[])]
print('next      :'); [print(f'  - {v}') for v in x.get('next_steps',[])]
if x.get('details'): print('details   :'); print(json.dumps(x['details'],indent=2,sort_keys=True))
