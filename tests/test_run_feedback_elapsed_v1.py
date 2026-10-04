import json
from openea_benchmark.adaptive.run_feedback import ProgressReporter

def test_progress_records_start_time_once(tmp_path):
    path=tmp_path/'status.json'; reporter=ProgressReporter(run_id='r',workflow='w',status_file=path); first=json.loads(path.read_text()); reporter.update(current_step='LONG_STEP'); second=json.loads(path.read_text()); assert first['started_at_utc']; assert second['started_at_utc']==first['started_at_utc']; assert second['updated_at_utc']>=first['updated_at_utc']
