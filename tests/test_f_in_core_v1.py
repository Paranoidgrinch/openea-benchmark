import numpy as np
from scipy.integrate import quad

from openea_benchmark import f_in_core


def test_variants_and_measured_levels():
    u = f_in_core.variants("U")
    assert {rec["f_in_core"] for rec in u.values()} == {0, 1, 2, 3}
    assert u["ECP81MWB"]["core_electrons"] == 81 and u["ECP81MWB"]["cpp"]["alpha"] > 0
    assert {"ECP81MWB-AVDZ", "ECP81MWB-AVTZ", "ECP81MWB-AVQZ"} <= set(u["ECP81MWB"]["basis_sets"])
    assert f_in_core.reference_level("U", 3)["energy_cm1"] == 0.0
    assert f_in_core.comparable_f_counts("Fm") == [12]


def test_cpp_one_electron_integral_matches_radial_quadrature():
    from pyscf import gto
    a1, a2, alpha, delta, ncut = 0.8, 0.3, 1.5, 0.74, 2
    mol = gto.M(atom="Ne 0 0 0", basis={"Ne": [[0, [a1, 1.0]], [0, [a2, 1.0]]]}, verbose=0)
    h1, G, e = f_in_core.cpp_matrices(mol, {0: dict(alpha=alpha, delta=delta, ncut=ncut)}, 4)
    norm = lambda a: (2 * a / np.pi) ** 0.75
    f = lambda r: (4 * np.pi * r ** 2 * norm(a1) * norm(a2) * np.exp(-(a1 + a2) * r ** 2)
                   * (1 - np.exp(-delta * r ** 2)) ** (2 * ncut) / r ** 4)
    assert abs(h1[0, 1] - (-0.5 * alpha * quad(f, 0, np.inf, epsabs=1e-14, limit=200)[0])) < 1e-10
    assert e == 0.0 and np.allclose(G, 0.0, atol=1e-12)  # s functions, one center: no field


def test_cpp_nuclear_constant_of_a_point_charge():
    from pyscf import gto
    alpha, delta, ncut, R = 2.0, 0.5, 2, 3.0
    mol = gto.M(atom=f"Ne 0 0 0; H 0 0 {R}", unit="Bohr", basis="sto-3g", spin=1, verbose=0)
    _, _, e = f_in_core.cpp_matrices(mol, {0: dict(alpha=alpha, delta=delta, ncut=ncut)}, 3)
    field = (1 - np.exp(-delta * R ** 2)) ** ncut / R ** 2
    assert abs(e - (-0.5 * alpha * field ** 2)) < 1e-12
