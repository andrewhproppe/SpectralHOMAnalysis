import time

from tqdm import tqdm
from fig_util import *
from SPDC import Type0SPDC, ppKTP
from HOM_util import (
    generate_signal_idler_wavelengths,
    wl_to_THz,
    get_diagonal_indices,
    gaussian,
    FFT_filter,
    c_μm_per_ps,
)
from copy import deepcopy
from numba import jit


@jit(nopython=True, parallel=True)
def HOM_JSI_ω(T, R, Φ, ωs, ωi, f1s, n1s, τ):
    return (
        T ** 2 * (np.abs(f1s) ** 2) * (np.abs(Φ) ** 2)
        + R ** 2 * (np.abs(np.flipud(f1s.T)) ** 2) * (np.abs(Φ) ** 2)
        - (2 * R * T)
        * (np.abs(f1s) * np.abs(np.flipud(f1s.T)))
        * (np.abs(Φ) * np.abs(Φ))
        * np.cos((ωs - ωi) * 2 * π * τ + n1s - np.flipud(n1s.T))
    )


@jit(nopython=True, parallel=True)
def HOM_JSI_ω(T, R, Φ, ωs, ωi, f1s, n1s, τ):
    return (
        T ** 2 * (np.abs(f1s) ** 2) * (np.abs(Φ) ** 2)
        + R ** 2 * (np.abs(np.flipud(f1s.T)) ** 2) * (np.abs(Φ) ** 2)
        - (2 * R * T)
        * (np.abs(f1s) * np.abs(np.flipud(f1s.T)))
        * (np.abs(Φ) * np.abs(Φ))
        * np.cos((ωs - ωi) * 2 * π * τ + n1s - np.flipud(n1s.T))
    )


@jit(nopython=True, parallel=True)
def homJsiStaticTerm(
    T: float,
    R: float,
    Φ: np.array,
    f1s: np.array,
) -> np.array:
    return T ** 2 * (np.abs(f1s) ** 2) * (np.abs(Φ) ** 2) + R ** 2 * (
        np.abs(np.flipud(f1s.T)) ** 2
    ) * (np.abs(Φ) ** 2)


@jit(nopython=True, parallel=True)
def homJsiCosArg(ωs: np.array, ωi: np.array, n1s: np.array, τ: float):
    return (ωs - ωi) * 2 * π * τ + n1s - np.flipud(n1s.T)


@jit(nopython=True, parallel=True)
def homJsiInterferenceTerm(T, R, Φ, f1s, cos_arg):
    return (
        -(2 * R * T)
        * (np.abs(f1s) * np.abs(np.flipud(f1s.T)))
        * (np.abs(Φ) * np.abs(Φ))
        * np.cos(cos_arg)
    )


@jit(nopython=True, parallel=True)
def HOM_JSI_λ(T, R, Φ, c, λs, λi, f1s, n1s, Δz):
    """Calculates HOM JSI in wavelength domain (need to fix to change delay to ps instead of μm)"""
    return (
        T ** 2 * (np.abs(f1s) ** 2) * (np.abs(Φ) ** 2)
        + R ** 2 * (np.abs(np.flipud(f1s.T)) ** 2) * (np.abs(Φ) ** 2)
        - 2
        * R
        * T
        * np.abs(f1s)
        * np.abs(np.flipud(f1s.T))
        * np.abs(Φ)
        * np.abs(Φ)
        * np.cos((1 / λs - 1 / λi) * 2 * π * c * Δz / c + n1s - np.flipud(n1s.T))
    )


class SpectralHOM:
    def __init__(self):
        self.λp = None
        self.λs = None
        self.λi = None
        self.spdc = None
        self.spectrum = None
        self.refr_idx = None
        self.JSIs = None
        self.JSI_proj = None
        self.T = 0.5
        self.R = 0.5

    def generate_wavelengths(self, λp, λ_start, λ_end=None, resolution=256, offset=0.0):
        self.λs, self.λi = generate_signal_idler_wavelengths(
            resolution, λp, λ_start, λ_end, offset
        )
        self.λp = λp
        self.resolution = resolution

    def generate_spdc(self, T=24, L=1e-3, pp=3.425, pump_σ=0.1):
        self.spdc = Type0SPDC(
            crystal=ppKTP(T=T, L=L, pp=pp),
            pump_σ=pump_σ,
        )
        self.spdc.phasematch(self.λp, self.λs, self.λi)
        self.ωs = wl_to_THz(self.λs)
        self.ωi = wl_to_THz(self.λi)
        self.ω_s = wl_to_THz(self.spdc.λ_s)
        self.ω_i = wl_to_THz(self.spdc.λ_i)
        self.diag_idxs = get_diagonal_indices(self.spdc.JSA)

    def prefilter_JSA(self, amp=1, σ=1):
        gauss = gaussian(self.ωs, amp, σ, wl_to_THz(self.λp * 2))
        self.spdc.JSA = self.spdc.JSA * gauss * gauss.T

    def assign_spectrum(self, spectrum=None, refr_idx=None):
        """
        - Assigns spectrum and refractive index to the HOM object, which are applied to the signal arm in calculate_JSI()
        :param spectrum: Normalized transmission spectrum
        :param refr_idx: Normalized refractive index spectrum
        :return:
        """
        if spectrum is None:
            spectrum = np.ones_like(self.ωs)
        if refr_idx is None:
            refr_idx = np.ones_like(self.ωs)

        self.spectrum = spectrum
        self.refr_idx = refr_idx
        self.f1s, _ = np.meshgrid(
            spectrum, spectrum
        )  # filter signal meshed, filter idler meshed
        self.n1s, _ = np.meshgrid(refr_idx, refr_idx)  # n signal meshed, n idler meshed

    def calculate_JSI(self, δ, units="μm", sample=True):
        """
        Calculates joint spectral intensity (JSI) from the SPDC JSA after passing through the HOM interferometer for a given delay τ
        :param δ: HOM interferometer delay in microns or picoseconds
        :param units: Units of τ, either "μm" or "ps"
        :return: HOM JSI
        """
        τ = δ / c_μm_per_ps if units == "μm" else δ
        JSI = HOM_JSI_ω(
            self.T, self.R, self.spdc.JSA, self.ω_s, self.ω_i, self.f1s, self.n1s, τ
        )
        self.JSI = JSI
        self.λd = np.concatenate((self.λs, self.λi), axis=0)
        return JSI

    def calculate_JSI_scan(self, nstep, start, stop=None, units="μm"):
        stop = -1 * start if stop is None else stop
        τs = np.linspace(start, stop, nstep)
        JSIs = []
        tic = time.time()
        for τ in tqdm(τs, desc="Calculating JSIs..."):
            JSIs.append(self.calculate_JSI(τ, units))
        print(f"JSI calculation took {time.time()-tic:.2f} seconds")
        self.JSIs = np.array(JSIs)

    @staticmethod
    def project_JSI(JSIs):
        """
        Project JSI onto specified axis
        :return: JSI projections
        """
        diag_arrs = []
        for jsi in tqdm(JSIs, desc="Calculating JSI projections..."):
            diag_vals = []
            for i in range(-jsi.shape[0], jsi.shape[0]):
                diag = np.diagonal(np.fliplr(jsi), offset=i)
                diag_vals.append(np.sum(diag))
            diag_arrs.append(np.array(diag_vals))
        return np.array(diag_arrs)

    def get_HOM_dip(self, start=None, stop=None, norm=True):
        if start is None:
            start, stop = 0, -1
        if self.JSI_proj is not None:
            self.HOM_dip = np.sum(self.JSI_proj[:, start:stop], axis=1)
            if norm:
                self.HOM_dip /= self.HOM_dip[0]

    def JSI_FFT(
        self,
        jsi_proj=None,
        δ=None,
        pad=None,
        offset=None,
        filter_width=None,
        filter_pos="right",
        units="μm",
        load_params=False,
    ):
        if load_params:
            δ = self.FFT_params["delay"]
            pad = self.FFT_params["pad"]
            offset = self.FFT_params["offset"]
            filter_width = self.FFT_params["filter_width"]
            filter_pos = self.FFT_params["filter_pos"]

        filter_width = (
            (self.resolution + pad) // 10 if filter_width is None else filter_width
        )

        if jsi_proj is None:
            jsi = self.calculate_JSI(δ, units)
            jsi_proj = self.project_JSI((jsi,))[0]

        arr_fft, arr_ifft, idx_arr, filter = FFT_filter(
            jsi_proj,
            pad=pad,
            filter_pos=filter_pos,
            filter_width=filter_width,
            centering_offset=offset,
        )

        # Store results and parameters in a dictionary to load into new HOM object
        self.FFT = {
            "jsi_proj": jsi_proj,
            "fft": arr_fft,
            "ifft": arr_ifft,
            "idx_arr": idx_arr,
            "filter": filter,
        }

        self.FFT_params = {
            "delay": δ,
            "pad": pad,
            "offset": offset,
            "filter_width": filter_width,
            "filter_pos": filter_pos,
        }

    def plot_FFT_filter(self):
        fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(3.0 * 3, 3.0), dpi=150)
        ax[0].plot(self.FFT["jsi_proj"])
        ax[0].set_title("JSI projection")
        ax[1].plot(self.FFT["idx_arr"], self.FFT["fft"].real)
        ax[1].plot(self.FFT["idx_arr"], self.FFT["fft"].imag)
        ax[1].plot(self.FFT["idx_arr"], self.FFT["filter"] * max(self.FFT["fft"]))
        ax[1].set_title("FFT + filter")
        ax[2].plot(self.FFT["ifft"].real, label="Real")
        ax[2].plot(self.FFT["ifft"].imag, label="Imag.")
        ax[2].set_title("iFFT")
        ax[1].ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        dress_fig()

    def create_copy(self):
        """
        Creates a copy of the SpectralHOM object. The main purpose of this is to copy the experimental conditions (wavelengths, SPDC, etc.) to a new HOM object
        that has no spectral filtering applied to the signal/idler arms
        :return: copy of SpectralHOM object
        """
        return deepcopy(self)


if __name__ == "__main__":
    print(0)