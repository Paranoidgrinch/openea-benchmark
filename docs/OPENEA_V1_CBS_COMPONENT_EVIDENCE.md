# OpenEA v1 — Targeted CBS component evidence

The checkpoint audit found complete component evidence for the resolved
d-aug-cc-pV5Z PEC, but no persistent QZ/5Z component records from the older
cardinal runs.

Rather than rerunning complete PECs, this layer uses the already resolved
d-aug-cc-pV5Z neutral and anion minimum geometries as fixed reference
geometries and computes only four molecular single points:

- neutral / aug-cc-pVQZ
- anion / aug-cc-pVQZ
- neutral / aug-cc-pV5Z
- anion / aug-cc-pV5Z

Each point records HF, CCSD correlation, CCSD total, (T), and CCSD(T) total.
No fragments, no PEC refinement, and no CBS extrapolation are performed here.

Completed points are checkpointed independently; reruns with the same run ID
reuse them.

Workflow:
cardinal CLEARED -> diffuse CLEARED -> component audit -> targeted QZ/5Z
component evidence -> component-resolved CBS.
