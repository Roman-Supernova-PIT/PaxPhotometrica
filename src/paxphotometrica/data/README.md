# Bundled Reference Data

PaxPhotometrica ships a small set of reference inputs so its deterministic toy
simulation works immediately after installation:

- `passbands.txt`: six Roman WFI relative-throughput curves used by the imaging
  model.
- `prism_wavelengths.txt`: fixed prism wavelength-pixel centers in Angstroms.
- `ice_loglam_nodes.txt`: default log10-wavelength nodes for the ice spline.
- `bosz_logflux_empca_basis.npz`: a normalized log-flux EMPCA basis derived
  from selected BOSZ 2024 spectra.

The BOSZ High-Level Science Products are distributed by MAST under CC BY 4.0:

- R. C. Bohlin et al., 2017, AJ, 153, 234
- S. Meszaros et al., 2024, A&A, 688, A197
- DOI: 10.17909/T95G68
- https://archive.stsci.edu/hlsp/bosz

The EMPCA basis contains relative spectral shapes, not BOSZ absolute fluxes.
Its metadata records the source model filenames and construction details.

The passband and prism tables are bundled as the current reference inputs for
this advanced calibration prototype.
