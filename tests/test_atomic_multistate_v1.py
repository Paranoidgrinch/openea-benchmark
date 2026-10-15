"""Atomic multistate layer (Patch 61): active spaces, roots, inputs, trigger."""
from pathlib import Path

from openea_benchmark.atomic_multistate import (
    AtomicMultistatePolicy, active_space, block_input, rassi_input, roots_per_spin,
)
from openea_benchmark.electronic_ea_stage import needs_atomic_multistate


def test_active_space_spans_competing_shells():
    la = active_space("La", -1)
    assert la["shells"] == ["4f", "5d", "6s", "6p"] and la["active_electrons"] == 4 and la["inactive"] == 27
    ag = active_space("Ag", 0)
    assert ag["shells"] == ["4d", "5s", "5p"] and ag["active_electrons"] == 11


def test_roots_follow_the_determinant_configuration():
    assert roots_per_spin("La", 0, [1, 3], 15, {1: "d1", 3: "s1d2"}) == {1: 5, 3: 7}  # 2D, 4F
    assert roots_per_spin("La", -1, [2, 4], 15, {2: "d2", 4: "s1d3"}) == {2: 7, 4: 7}  # 3F, 5F
    assert roots_per_spin("I", 0, [1], 15, {1: "p1"}) == {1: 3}  # 2P
    assert roots_per_spin("Ag", -1, [0], 15, {0: "closed"}) == {0: 1}  # 1S


def test_trigger_only_for_d_and_f_atoms():
    assert needs_atomic_multistate("La") and needs_atomic_multistate("Ti") and needs_atomic_multistate("Ag")
    assert not needs_atomic_multistate("O") and not needs_atomic_multistate("I")


def test_block_and_rassi_inputs():
    pol = AtomicMultistatePolicy()
    sp = active_space("La", 0)
    text = block_input("La", 0, sp, 1, 5, pol, Path("/x/jobmix"))
    assert "XMultistate" in text and "CIRoot\n5 5 1" in text and ">>> COPY $Project.JobMix /x/jobmix" in text
    r = rassi_input("La", {1: 5, 3: 7}, {1: Path("/a"), 3: Path("/b")}, pol)
    assert "Nr of JobIph\n2 5 7" in r and ">>> COPY /b JOB002" in r and "EJob" in r
