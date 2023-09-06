import numpy as np
import matplotlib.pyplot as plt

from HOM_absorption.SpectralHOM import SpectralHOM
from HOM_absorption.HOM_util import n_and_k_from_abs_data, make_lorentzian_spectrum

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

# shom object with spectral filtering applied
shom.assign_spectrum(spectrum, refr_idx)
shom.JSI_FFT(δ=500, pad=1000, filter_width=100, offset=0, filter_pos='right')
shom.plot_FFT_filter()

# raise RuntimeError

# without spectral filtering
shomu = shom.create_copy()
shomu.assign_spectrum(None, None)
shomu.JSI_FFT(load_params=True)

# Subtract iFFTs
# ifft_diff = shom.FFT['ifft'] - shomu.FFT['ifft']
# plt.plot(ifft_diff.real)
# plt.plot(ifft_diff.imag)

# Subtract JSI and then do FFT
# plt.figure()
# plt.plot(shom.FFT['jsi_proj'])
# shom.FFT['jsi_proj'] -= shomu.FFT['jsi_proj']
# plt.plot(shom.FFT['jsi_proj'])

shom.JSI_FFT(
    jsi_proj=shom.FFT['jsi_proj'] - shomu.FFT['jsi_proj'],
    load_params=True
)
shom.plot_FFT_filter()
# shom.JSI_FFT(load_params=True)
# shom.plot_FFT_filter()