import json
from openea_benchmark.adaptive.run_feedback import ProgressReporter, ProgressStatus

def test_progress_reporter_writes_current_and_next_steps(tmp_path,capsys):
    path=tmp_path/'status.json'; r=ProgressReporter(run_id='r1',workflow='TEST',status_file=path)
    r.update(current_step='STEP_B',completed_steps=['STEP_A'],next_steps=['STEP_C','STEP_D'],details={'basis':'aug-cc-pv5z'})
    x=json.loads(path.read_text()); assert x['status']==ProgressStatus.RUNNING; assert x['current_step']=='STEP_B'; assert x['next_steps']==['STEP_C','STEP_D']; assert '[OPENEA][RUNNING]' in capsys.readouterr().err

def test_completion_and_blocked_are_persistent(tmp_path):
    path=tmp_path/'status.json'; r=ProgressReporter(run_id='r1',workflow='TEST',status_file=path); r.completed(current_step='DONE',completed_steps=['A'],next_steps=['B'])
    assert json.loads(path.read_text())['status']==ProgressStatus.COMPLETED
    r.blocked(current_step='BACKEND',next_steps=['INSTALL','RERUN'],details={'reason':'missing'})
    x=json.loads(path.read_text()); assert x['status']==ProgressStatus.BLOCKED; assert x['details']['reason']=='missing'
