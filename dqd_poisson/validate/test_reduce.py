"""Validate the 1D z-Schrodinger solver against the harmonic-oscillator
analytic ground-state energy hbar*omega/2 (sanity check for reduce.py,
independent of the full 3D electrostatics solve)."""

import numpy as np

from ..reduce import solve_1d_schrodinger_z, HBAR
from ..materials import M_T


def test_harmonic_oscillator_ground_state():
    omega = 5e13  # rad/s, chosen so localization length ~ few nm (well-like)
    L = 60e-9
    Nz = 400
    z = np.linspace(-L / 2, L / 2, Nz)
    u = 0.5 * M_T * omega ** 2 * z ** 2

    energies, psis = solve_1d_schrodinger_z(z, u, m_t=M_T, n_states=3)

    E0_exact = 0.5 * HBAR * omega
    E1_exact = 1.5 * HBAR * omega
    assert abs(energies[0] - E0_exact) / E0_exact < 1e-3
    assert abs(energies[1] - E1_exact) / E1_exact < 1e-3

    dz = np.gradient(z)
    norm0 = np.sum(psis[0] ** 2 * dz)
    assert abs(norm0 - 1.0) < 1e-6
    # ground state should be even (no node), first excited odd (one node)
    assert np.sign(psis[0][10]) == np.sign(psis[0][-10])


if __name__ == "__main__":
    test_harmonic_oscillator_ground_state()
    print("reduce.py 1D Schrodinger solver: harmonic-oscillator check passed")
