import numpy as np
import matplotlib.pyplot as plt

from SpectralHOM import SpectralHOM
from HOM_util import n_and_k_from_abs_data, make_lorentzian_spectrum

shom = SpectralHOM()
shom.generate_wavelengths(λp=405.5, λ_start=775, λ_end=850, resolution=256*2, offset=0.0)
shom.generate_spdc()

# Make spectrum
""" Make spectral filter """
spectrum, refr_idx = n_and_k_from_abs_data(root='../Data/', fname="ZnNc_shifted.txt", λ_target=shom.λs)

refr_idx, spectrum = make_lorentzian_spectrum(
    ω=shom.ωs,
    ω0=[360, 365],
    T2=[0.3, 0.5],
    od=[1000, 500]
)

spectrum = spectrum / np.max(spectrum) * 0.5
spectrum = 1 - spectrum
refr_idx /= 400

shom.assign_spectrum(spectrum, refr_idx)
# shom.prefilter_JSA(amp=1, σ=1)
shom.calculate_JSI_scan(start=-300, nstep=151, units='μm')
shom.JSI_proj = SpectralHOM.project_JSI(shom.JSIs)

plt.figure()
plt.imshow(shom.JSI_proj)