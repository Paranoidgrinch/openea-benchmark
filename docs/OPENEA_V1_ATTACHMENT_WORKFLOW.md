# OpenEA v1 — Stage-3 Electronic-EA Workflow

This layer orchestrates the first complete high-level electronic-EA evidence
path from independent neutral and anion Stage-3 PEC loops through:
Stage-3 attachment bridges, equilibrium-energy intervals, neutral/anion
pairing, interval-aware anion dissociation stability, and the EA interval
decision.

For an anion equilibrium interval [L_A,U_A] and lowest supplied dissociation
threshold D, the dissociation-margin interval is [D-U_A, D-L_A].

- Entire margin above the explicit safety margin -> BOUND against supplied
  channels.
- Entire margin non-positive -> UNBOUND against dissociation.
- Otherwise -> UNRESOLVED.
- Any missing supplied relevant channel energy -> UNRESOLVED.

Even a BOUND result is explicitly only electronic EA evidence. This workflow
does not include ZPE/nuclear motion, CBS incompleteness, higher-order
correlation, core-valence, scalar relativity, SOC, DBOC, or nonadiabatic
terms where relevant. It cannot claim a production adiabatic EA.
