"""Generator for the 4-tier XRD crystallographic benchmark suite."""

from pathlib import Path

import numpy as np
from pymatgen.analysis.diffraction.xrd import XRDCalculator
from pymatgen.core import Lattice, Structure
from pymatgen.io.cif import CifWriter


def build_benchmark_fixtures(target_dir: Path):
    """Generate CIF models and synthetic XRD patterns for benchmark tiers."""
    cif_dir = target_dir / "cifs"
    pattern_dir = target_dir / "patterns"
    cif_dir.mkdir(parents=True, exist_ok=True)
    pattern_dir.mkdir(parents=True, exist_ok=True)

    calc = XRDCalculator(wavelength=1.540596)

    # 1. Tier 1 structures
    # CeO2 (Fm-3m)
    ceo2 = Structure.from_spacegroup(
        sg="Fm-3m",
        lattice=Lattice.cubic(5.411),
        species=["Ce", "O"],
        coords=[[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]],
    )
    CifWriter(ceo2, symprec=0.01).write_file(str(cif_dir / "CeO2.cif"))

    # TiO2 Rutile (P4_2/mnm)
    rutile = Structure.from_spacegroup(
        sg="P4_2/mnm",
        lattice=Lattice.tetragonal(4.5937, 2.9587),
        species=["Ti", "O"],
        coords=[[0.0, 0.0, 0.0], [0.305, 0.305, 0.0]],
    )
    CifWriter(rutile, symprec=0.01).write_file(str(cif_dir / "TiO2_rutile.cif"))

    # 2. Tier 2 structures
    # TiO2 Anatase (I4_1/amd)
    anatase = Structure.from_spacegroup(
        sg="I4_1/amd",
        lattice=Lattice.tetragonal(3.7845, 9.5143),
        species=["Ti", "O"],
        coords=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.208]],
    )
    CifWriter(anatase, symprec=0.01).write_file(str(cif_dir / "TiO2_anatase.cif"))

    # ZnO Wurtzite (P6_3mc)
    zno = Structure.from_spacegroup(
        sg="P6_3mc",
        lattice=Lattice.hexagonal(3.253, 5.207),
        species=["Zn", "O"],
        coords=[[1 / 3, 2 / 3, 0.0], [1 / 3, 2 / 3, 0.382]],
    )
    CifWriter(zno, symprec=0.01).write_file(str(cif_dir / "ZnO.cif"))

    # 3. Tier 4 invalid structure (Fe on O site - severe BVS inversion)
    inverted_rutile = Structure.from_spacegroup(
        sg="P4_2/mnm",
        lattice=Lattice.tetragonal(4.5937, 2.9587),
        species=["O", "Fe"],
        coords=[[0.0, 0.0, 0.0], [0.305, 0.305, 0.0]],
    )
    CifWriter(inverted_rutile, symprec=0.01).write_file(
        str(cif_dir / "Inverted_Rutile_Invalid.cif")
    )

    # Helper to generate continuous profile
    two_theta_grid = np.linspace(10.0, 80.0, 3501)  # step 0.02 deg

    def simulate_profile(pattern_peaks_2th, pattern_peaks_i, fwhm_base=0.08, noise_level=5.0):
        intensity = np.zeros_like(two_theta_grid)
        # Background: flat 50 + slight slope
        bg = 50.0 + 0.2 * (two_theta_grid - 10.0)
        intensity += bg

        for t, h in zip(pattern_peaks_2th, pattern_peaks_i, strict=False):
            if 10.0 <= t <= 80.0:
                # Pseudo-Voigt peak (50% Gaussian, 50% Lorentzian)
                sig = fwhm_base / (2 * np.sqrt(2 * np.log(2)))
                gauss = np.exp(-0.5 * ((two_theta_grid - t) / sig) ** 2)
                lorentz = 1.0 / (1.0 + ((two_theta_grid - t) / (fwhm_base / 2)) ** 2)
                intensity += h * 10.0 * (0.5 * gauss + 0.5 * lorentz)

        # Poisson/Gaussian noise
        rng = np.random.default_rng(42)
        noise = rng.normal(0, noise_level, size=len(two_theta_grid))
        intensity = np.maximum(intensity + noise, 0.0)
        return intensity

    # Generate CeO2 pattern
    pat_ceo2 = calc.get_pattern(ceo2, two_theta_range=(10, 80))
    i_ceo2 = simulate_profile(pat_ceo2.x, pat_ceo2.y)
    np.savetxt(pattern_dir / "CeO2.xy", np.column_stack((two_theta_grid, i_ceo2)), fmt="%.4f %.2f")

    # Generate Rutile pattern
    pat_rut = calc.get_pattern(rutile, two_theta_range=(10, 80))
    i_rut = simulate_profile(pat_rut.x, pat_rut.y)
    np.savetxt(
        pattern_dir / "TiO2_rutile.xy", np.column_stack((two_theta_grid, i_rut)), fmt="%.4f %.2f"
    )

    # Generate 95:5 Rutile:Anatase mixture
    pat_ana = calc.get_pattern(anatase, two_theta_range=(10, 80))
    combined_peaks_2th = np.concatenate([pat_rut.x, pat_ana.x])
    combined_peaks_i = np.concatenate([0.95 * pat_rut.y, 0.05 * pat_ana.y])
    i_mix_95_5 = simulate_profile(combined_peaks_2th, combined_peaks_i)
    np.savetxt(
        pattern_dir / "Mixture_Rutile95_Anatase5.xy",
        np.column_stack((two_theta_grid, i_mix_95_5)),
        fmt="%.4f %.2f",
    )

    # Generate 50:50 Rutile:Anatase mixture
    combined_50_50_i = np.concatenate([0.50 * pat_rut.y, 0.50 * pat_ana.y])
    i_mix_50_50 = simulate_profile(combined_peaks_2th, combined_50_50_i)
    np.savetxt(
        pattern_dir / "Mixture_Rutile50_Anatase50.xy",
        np.column_stack((two_theta_grid, i_mix_50_50)),
        fmt="%.4f %.2f",
    )

    # Generate Nanocrystalline broadening pattern (FWHM = 0.40 deg)
    i_nano = simulate_profile(pat_rut.x, pat_rut.y, fwhm_base=0.40)
    np.savetxt(
        pattern_dir / "TiO2_nano_broadened.xy",
        np.column_stack((two_theta_grid, i_nano)),
        fmt="%.4f %.2f",
    )

    print(f"Benchmark fixtures generated in {target_dir}")


if __name__ == "__main__":
    out = Path(__file__).parent
    build_benchmark_fixtures(out)
