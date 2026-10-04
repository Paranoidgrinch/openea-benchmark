\
from pathlib import Path
import json
import subprocess
import sys

import pytest

from openea_benchmark.adaptive.stage_checkpoint import StageCheckpointStore


def test_stage_checkpoint_roundtrip(tmp_path):
    store=StageCheckpointStore(tmp_path/'stages',signature='sig')
    value={'stage':'neutral','values':[1,2,3]}
    store.save('neutral_loop',value)
    assert store.has('neutral_loop')
    assert store.load('neutral_loop')==value


def test_stage_checkpoint_signature_mismatch_fails_closed(tmp_path):
    StageCheckpointStore(tmp_path/'stages',signature='sig-a').save('neutral_loop',{'x':1})
    with pytest.raises(ValueError):
        StageCheckpointStore(tmp_path/'stages',signature='sig-b')


def test_stage_checkpoint_rejects_unsafe_key(tmp_path):
    store=StageCheckpointStore(tmp_path/'stages',signature='sig')
    with pytest.raises(ValueError):
        store.save('../escape',{'x':1})
