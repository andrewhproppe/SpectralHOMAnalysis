# SpectralHOMAnalysis

Simulation of spectrally-resolved Hong-Ou-Mandel inferferometry experiment using a time-tagging camera (TimePix3). Spectrally-entangled photon pairs are generated via spontaneous parametric downconversion (SPDC).

As currently implemented, the user has the option to filter the entire joint spectral amplitude (JSA) as generated by the SPDC.
More relevant to the authors' experiments, the user can also specify amplitude and phase filters that correspond to e.g. a molecular sample with frequency-dependent absorption, α(ω), and refractive index, n(ω).
The detected joint spectral intensity (JSI) is then calculated with or without the molecular sample in one arm of the interferometer.

## Usage

```py
import numpy as np
import matplotlib.pyplot as plt

from SpectralHOM import SpectralHOM
from HOM_util make_lorentzian_spectrum

shom = SpectralHOM() # create a SpectralHOM object
shom.generate_wavelengths(λp=405.5, λ_start=775, λ_end=850, resolution=256*2, offset=0.0) # set pump wavelength and the start/end wavelengths of the SPDC spectrum
shom.generate_spdc() # generate the SPDC object

refr_idx, spectrum = make_lorentzian_spectrum(
    ω=shom.ωs,
    ω0=[360, 365],
    T2=[0.3, 0.5],
    od=[1000, 500]
) # Simulate a spectrum and refractive index

shom.assign_spectrum(spectrum, refr_idx) # and assign the spectra to the shom object

shom.calculate_JSI_scan(start=-300, nstep=151, units='μm') # calculates a JSI for a set of interferometer delays

shom.JSI_proj = SpectralHOM.project_JSI(shom.JSIs) # projects the JSIs onto one axis

plt.figure()
plt.imshow(shom.JSI_proj)
```
