# Reference data

This directory contains inputs used to construct or validate packaged runtime
data, but which are not needed when running `paxphot`.

- `bosz2024_wave_r500.txt` is the wavelength grid used when the BOSZ EMPCA
  basis was constructed. Its wavelengths are in Angstroms.
- The local `m+0.00/` directory contains the source BOSZ spectra and is ignored
  by Git because it is not required at runtime.

The installable package includes the compact EMPCA result at
`src/paxphotometrica/data/bosz_logflux_empca_basis.npz`.
