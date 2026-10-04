\
# OpenEA v1 — Nested Progress and Stage-Level Resume

The first d-aug OH/OH- run returned child exit code 2. In the OH validation
child this code specifically denotes an atomic-fragment failure, but the
parent diffuse runner previously discarded the child's stdout JSON and exposed
only the integer exit code.

This patch makes nested basis calculations observable and resumable.

## Nested progress

Each finite-basis child now reports its own stages:

    NEUTRAL_PEC
    ANION_PEC
    FRAGMENT_<id>   (one entry per atomic fragment)
    ATTACHMENT_DECISION
    ELECTRONIC_EA_BASIS_POINT_COMPLETE

The parent still reports the higher-level diffuse/cardinal workflow. Child
status is stored in a separate JSON file so the two progress layers do not
overwrite one another.

## Child failure evidence

Child stdout is persisted under the parent run directory. On a non-zero child
exit the complete child JSON is also copied into the live stderr/log stream.
For fragment failures the parent extracts and reports the failing fragment ID,
status, exception type, and message.

## Stage-level resume

Within one basis point, successful expensive stages are checkpointed:

- neutral Stage-3 PEC loop;
- anion Stage-3 PEC loop;
- each atomic fragment independently.

A rerun with the same workflow signature reuses those completed stages. Thus a
failure of the fourth atomic fragment no longer requires recomputing both
molecular PECs or the first three fragments.

Stage checkpoints use Python pickle because Stage3 loop results contain rich
Python/numpy dataclasses. This is explicitly a trusted-local-run format: OpenEA
must only load checkpoint files it created inside its own run directory. A
workflow-signature hash prevents accidental reuse under changed scientific
settings.
