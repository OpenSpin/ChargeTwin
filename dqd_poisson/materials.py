"""Material constants for the Si/SiGe DQD heterostructure (spec §3.3)."""

from dataclasses import dataclass

EPS0 = 8.8541878128e-12  # F/m
E_CHARGE = 1.602176634e-19  # C
M_E = 9.1093837015e-31  # kg

EPS_R_SIO2 = 3.9
EPS_R_SI = 12.0
EPS_R_SIGE = 13.2

U0_SIGE_BARRIER = 0.150  # eV, conduction-band offset of SiGe barrier above Si well
M_T = 0.19 * M_E  # transverse effective mass, in-plane


@dataclass(frozen=True)
class Material:
    name: str
    eps_r: float
    u_band_eV: float  # conduction-band offset relative to Si well (0 = well)


SIGE = Material("SiGe", EPS_R_SIGE, U0_SIGE_BARRIER)
SI_WELL = Material("Si_well", EPS_R_SI, 0.0)
SI_CAP = Material("Si_cap", EPS_R_SI, 0.0)
SIO2 = Material("SiO2", EPS_R_SIO2, U0_SIGE_BARRIER)  # insulator; U-> handled as hard wall in Schrodinger stage
METAL = Material("metal", None, None)  # Dirichlet BC region, not a dielectric cell
