"""PySCF UCCSD in-core integrals built block by block (same ERIs object, ~6x less peak RAM).

``pyscf.cc.uccsd._make_eris_incore`` transforms the full (pq|rs) for alpha-alpha,
beta-beta and alpha-beta as three nmo^4 arrays (``ao2mo.restore(1, ...)``), slices
copies of every block out of them, and only then packs the virtual pairs.  What it
keeps is small against that peak (vvvv blocks in triangular pair storage, ~0.75 v^4
in all), but the peak is ~6 nmo^4 x 8 bytes: an open-shell aug-cc-pV5Z point
(nmo ~250) needs ~130 GB and therefore always ran AO-direct with its o/v blocks on
scratch.  Here every block is transformed directly into the form PySCF keeps
(``ao2mo.general`` on the SCF's packed AO integrals, ``compact=True`` where PySCF
packs), so the peak is the kept blocks plus one transformation buffer.  The arrays
are the same (checked element by element in tests), hence the same energies.
"""
from __future__ import annotations

import numpy as np


def kept_mb(nocca: int, noccb: int, nvira: int, nvirb: int) -> float:
    """Size of the integral blocks PySCF's in-core UCCSD keeps (MB)."""
    pa, pb = nvira * (nvira + 1) // 2, nvirb * (nvirb + 1) // 2
    oa, ob, va, vb = nocca, noccb, nvira, nvirb
    aa = oa ** 4 + oa ** 3 * va + 3 * oa ** 2 * va ** 2 + oa * va * pa + pa * pa
    bb = ob ** 4 + ob ** 3 * vb + 3 * ob ** 2 * vb ** 2 + ob * vb * pb + pb * pb
    ab = oa ** 2 * ob ** 2 + oa * va * ob ** 2 + oa * va * ob * vb + oa ** 2 * vb ** 2 + oa * va * vb * ob \
        + oa * va * pb + pa * pb
    ba = ob * vb * oa ** 2 + ob ** 2 * va ** 2 + ob * vb * va * oa + ob * vb * pa
    return 8.0 * (aa + bb + ab + ba) / 1.0e6


def blockwise_peak_mb(nao: int, nocca: int, noccb: int, nvira: int, nvirb: int) -> float:
    """Kept blocks + the SCF's packed AO integrals + the largest half-transformed buffer."""
    nao_pair = nao * (nao + 1) // 2
    nvir = max(nvira, nvirb)
    buffer = nao_pair * (nvir * (nvir + 1) // 2)
    return kept_mb(nocca, noccb, nvira, nvirb) + 8.0 * (nao_pair * (nao_pair + 1) // 2 + buffer) / 1.0e6


def make_eris_blockwise(mycc, mo_coeff=None):
    """``uccsd._ChemistsERIs`` equal to ``_make_eris_incore`` (needs ``mycc._scf._eri``)."""
    from pyscf import ao2mo
    from pyscf.cc import uccsd
    eri = mycc._scf._eri
    if eri is None:
        raise ValueError("blockwise UCCSD integrals need the SCF's in-core AO integrals")
    eris = uccsd._ChemistsERIs()
    eris._common_init_(mycc, mo_coeff)
    nocca, noccb = mycc.nocc
    moa, mob = eris.mo_coeff
    oa, va = moa[:, :nocca], moa[:, nocca:]
    ob, vb = mob[:, :noccb], mob[:, noccb:]

    def full(c1, c2, c3, c4):
        n = [c.shape[1] for c in (c1, c2, c3, c4)]
        return ao2mo.general(eri, (c1, c2, c3, c4), compact=False).reshape(n)

    def packed(c1, c2, c3):       # (c1 c2 | c3 c3) with the last pair packed (tril)
        n3 = c3.shape[1]
        return ao2mo.general(eri, (c1, c2, c3, c3), compact=True).reshape(
            c1.shape[1], c2.shape[1], n3 * (n3 + 1) // 2)

    eris.oooo, eris.ovoo, eris.ovov = full(oa, oa, oa, oa), full(oa, va, oa, oa), full(oa, va, oa, va)
    eris.oovv, eris.ovvo = full(oa, oa, va, va), full(oa, va, va, oa)
    eris.ovvv = packed(oa, va, va)
    eris.vvvv = ao2mo.general(eri, (va, va, va, va), compact=True)
    eris.OOOO, eris.OVOO, eris.OVOV = full(ob, ob, ob, ob), full(ob, vb, ob, ob), full(ob, vb, ob, vb)
    eris.OOVV, eris.OVVO = full(ob, ob, vb, vb), full(ob, vb, vb, ob)
    eris.OVVV = packed(ob, vb, vb)
    eris.VVVV = ao2mo.general(eri, (vb, vb, vb, vb), compact=True)
    eris.ooOO, eris.ovOO, eris.ovOV = full(oa, oa, ob, ob), full(oa, va, ob, ob), full(oa, va, ob, vb)
    eris.ooVV, eris.ovVO = full(oa, oa, vb, vb), full(oa, va, vb, ob)
    eris.ovVV = packed(oa, va, vb)
    eris.vvVV = ao2mo.general(eri, (va, va, vb, vb), compact=True)
    eris.OVoo, eris.OOvv, eris.OVvo = full(ob, vb, oa, oa), full(ob, ob, va, va), full(ob, vb, va, oa)
    eris.OVvv = packed(ob, vb, va)
    return eris
