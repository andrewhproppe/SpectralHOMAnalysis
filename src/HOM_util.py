from fig_util import *
from conversions import wl_to_THz, deg2rad, wl2angfreq
from numba import jit
from scipy.signal import savgol_filter

π      = np.pi
c      = 299792458  # m/s
c_nm_per_ps = c*1e9*1e-12
c_μm_per_ps = c*1e6*1e-12
c_μm_per_fs = c_nm_per_ps*1e6
c_nm_per_fs = c*1e9*1e-15

@jit(nopython=True, parallel=True)
def HOM_JSI_λ(T, R, Φ, c, λs, λi, f1s, n1s, Δz):
    return T**2*(np.abs(f1s)**2)*(np.abs(Φ)**2) \
         + R**2*(np.abs(np.flipud(f1s.T))**2)*(np.abs(Φ)**2) \
         - 2*R*T*np.abs(f1s)*np.abs(np.flipud(f1s.T))*np.abs(Φ)*np.abs(Φ)*np.cos((1/λs - 1/λi)*2*π*c*Δz/c + n1s - np.flipud(n1s.T))


# @jit(nopython=True, parallel=True)
# def HOM_JSI_ω(T, R, Φ, c, ωs, ωi, f1s, n1s, Δz):
#     return T**2*(np.abs(f1s)**2)*(np.abs(Φ)**2) \
#          + R**2*(np.abs(np.flipud(f1s.T))**2)*(np.abs(Φ)**2) \
#          - 2*R*T*np.abs(f1s)*np.abs(np.flipud(f1s.T))*np.abs(Φ)*np.abs(Φ)*np.cos((ωs - ωi)*2*π*Δz/c + n1s - np.flipud(n1s.T))


@jit(nopython=True, parallel=True)
def HOM_JSI_ω(T, R, Φ, ωs, ωi, f1s, n1s, τ):
    return T**2*(np.abs(f1s)**2)*(np.abs(Φ)**2) \
         + R**2*(np.abs(np.flipud(f1s.T))**2)*(np.abs(Φ)**2) \
         - 2*R*T*np.abs(f1s)*np.abs(np.flipud(f1s.T))*np.abs(Φ)*np.abs(Φ)*np.cos((ωs - ωi)*2*π*τ + n1s - np.flipud(n1s.T))

def generate_signal_idler_wavelengths(n, λ_pump, λ_start, λ_end=None, offset=0.4):
    """
    Generate signal and idler wavelength arrays. The input values are converted into THz to ensure the subsequently generated
    JSA is evenly spaced in frequency rather than wavelength, so that there is no curvature along the diagonal.
    :param n: number of points (resolution)
    :param λ_pump: pump wavelength (nm)
    :param λ_start: wavelength range start (nm)
    :param λ_end: wavelength range end (nm)
    :param offset: ?
    :return:
    """
    if λ_end is None:
        λ_end  = λ_pump*λ_start/(λ_start-λ_pump) # SPDC upper
    # λ = np.linspace(λ_start, λ_end, n)
    ω_start = wl_to_THz(λ_start)
    ω_end   = wl_to_THz(λ_end)
    ω = np.linspace(ω_start, ω_end, n,
                    # dtype=np.float32
                    )
    λ = wl_to_THz(ω)
    λs = λ+offset
    λi = np.flip(λs)
    return λs, λi


def kth_diag_indices(a, k):
    rows, cols = np.diag_indices_from(a)
    if k < 0:
        return rows[-k:], cols[:k]
    elif k > 0:
        return rows[:-k], cols[k:]
    else:
        return rows, cols


def get_JSI_diagonal(JSI, λs, λi, offset=14, spread=5, interp=True, extrap=False):
    """
    Returns values and indices along diagonal of JSI. Indices are used to get subsections of the λs and λs arrays, so that the diagonal values can be
    interpolated or extrapolated along a common λ.
    """
    diag_vals = []
    diag_offsets = []
    λs_diag = []
    λi_diag = []
    len_rows = []
    len_cols = []
    for i in range(offset-spread, offset+spread):
        diag = np.diagonal(JSI, offset=i)
        rows, cols = kth_diag_indices(JSI, i)
        diag_vals.append(diag)
        λs_diag.append(λs[rows])
        λi_diag.append(λi[cols])
        diag_offsets.append(i)
        len_rows.append(len(rows))
        len_cols.append(len(cols))

    λinterp, diag_interp, λ_extrap, diag_extrap = None, None, None, None

    # Interpolate over shortest λ array
    if interp:
        λinterp = λs_diag[int(np.where(len_rows==np.min(len_rows))[0])]
        diag_interp = np.zeros((len(diag_vals), len(λinterp)))
    # Extrapolate over longest λ array
    if extrap:
        # λ_extrap = λs_diag[int(np.where(len_rows==np.max(len_rows))[0])]
        λ_extrap = λs
        diag_extrap = np.zeros((len(diag_vals), len(λ_extrap)))

    for i in range(0, len(diag_vals)):
        if interp:
            diag_interp[i, :] = np.interp(λinterp, λs_diag[i], diag_vals[i])
        if extrap:
            diag_extrap[i, :] = np.interp(λ_extrap, λs_diag[i], diag_vals[i])

    return λinterp, diag_interp, λ_extrap, diag_extrap


def PbS_n_and_k(λ, L= 0.5):
    data = np.genfromtxt('../Data/PbS_QD_n_and_k.txt', delimiter=',')
    s = find_nearest(1020, data[0, :])
    e = find_nearest(1350, data[0, :])
    data_new = data[:, s:e]
    λ_temp = np.linspace(λ[0], λ[-1], len(data_new[0, :]))
    interp_k = np.interp(λ, λ_temp, data_new[1, :])
    interp_n = np.interp(λ, λ_temp, data_new[2, :])

    interp_k *= L
    interp_n = (1 + L*(interp_n-1))
    return interp_k, interp_n


def load_absorption_data(fname, root='Data/', delimiter=''):
    data = np.genfromtxt(root + fname, delimiter=delimiter)
    λ = data[:, 0]
    α = data[:, 1]
    λ, α = zip(*sorted(zip(λ, α)))  # to correct any ordering mistakes from plot digitization
    λ = np.array(λ)
    α = np.array(α)
    return λ, α

def spectrum_to_refr_idx(spectrum, smooth=True, window_size=101, poly_order=3):
    refr_idx = np.diff(spectrum)
    refr_idx = np.append(refr_idx, refr_idx[-1])
    refr_idx = savgol_filter(refr_idx, window_size, poly_order) if smooth else refr_idx
    return refr_idx

def n_and_k_from_abs_data(fname, λ_target, root='Data/', delimiter='', smooth=True, window_size=101, poly_order=3):
    λ, α = load_absorption_data(fname, root=root, delimiter=delimiter)
    α = np.interp(λ_target, λ, α)
    α = α/α.max()
    n = spectrum_to_refr_idx(α, smooth=smooth, window_size=window_size, poly_order=poly_order)
    return α, n

def plot_n_and_k(fs, ns, λs, figsize=(3.5, 2.5), units='THz', mirrored=False, dpi=150):
    fig, ax1 = plt.subplots(dpi=150, figsize=figsize)

    if mirrored:
        fs_mir = np.flip(fs)
        fs = fs + fs_mir
        ns_mir = np.flip(ns)
        ns = ns + ns_mir

    color = 'tab:red'
    ax1.set_xlabel('$ω_{sig.}$ ('+units+')')
    ax1.set_ylabel('Spectral filter', color=color)
    ax1.plot(λs, fs, color=color, label = 'α(ω)')
    ax1.tick_params(axis='y', labelcolor=color)

    # Adding Twin Axes to plot using dataset_2
    ax2 = ax1.twinx()

    color = 'tab:green'
    ax2.set_ylabel('n(ω)', color=color)
    ax2.plot(λs, ns, color=color, label = 'n(ω)')
    ax2.tick_params(axis='y', labelcolor=color)

    dress_fig(tight=True)
    return fig


def plot_JSI(Φ, λs, λi, levels=50, units='THz', figsize=(3.5, 2.5), dpi=150):
    plt.figure(figsize=figsize, dpi=dpi)
    plt.contourf(λs, λi, Φ, levels, extend='both')
    dress_fig(xlabel='$ω_{s}$ (%s)' % units, ylabel='$ω_{i} $ (%s)' % units, tight=True)


def lorentzian(ω, ω0, T1):
    """
    from https://chem.libretexts.org/
    :param ω: frequency array
    :param ω0: central frequency
    :param T1: lifetime (in units of 1/ω)
    :return: Lorentzian lineshape with real and imaginary parts
    """
    # return (1/T1)/((1/T1)**2+(ω-ω0)**2) + 1j*(ω-ω0)/((1/T1)**2+(ω-ω0)**2)
    return 1/((1/T1) - 1j*(ω-ω0))


def complex_permittivity(ω, ω0, ωp, γ):
    return 1+ωp/(ω0**2-ω**2-1j*γ*ω)


def lorentzian_resonance_λ(λ, λ0, T, od):
    x = 2*np.pi*T*c_nm_per_fs*(λ - λ0)/(λ0**2)
    Λreal = od*x/(1+x**2)
    Λimag = od*1/(1+x**2)
    return Λreal, Λimag

def lorentzian_resonance_ω(ω, ω0, T, od):
    x = T*(ω-ω0)*2*np.pi
    Λreal = -od*x/(1+x**2)
    Λimag = od*1/(1+x**2)
    return Λreal, Λimag

def make_lorentzian_spectrum(ω, ω0, T2, od):
    total_real, total_imag = np.zeros_like(ω), np.zeros_like(ω)
    for i in range(len(ω0)):
        real, imag = lorentzian_resonance_ω(ω, ω0[i], T=T2[i], od=od[i])
        total_real += real
        total_imag += imag
    return total_real, total_imag

# def tophat_filter(arr, x1, x2):
#     tophat = np.ones_like(arr)
#     tophat[0:x1] = 0
#     tophat[x2:] = 0
#     return tophat

def tophat_filter(arr, centre_idx, width_idx):
    tophat = np.ones_like(arr)
    tophat[0:centre_idx-width_idx] = 0
    tophat[centre_idx+width_idx:] = 0
    return tophat

gaussian = lambda x, a, σ, μ: a*np.exp(-(x-μ)**2/(2*σ**2))

normal = lambda x, a, σ, μ: 1/np.sqrt(2*np.pi*σ**2)*np.exp(-(x-μ)**2/(2*σ**2))
# gauss_filter = gaussian(x=idx_arr, a=100, σ=100, μ=1024)

def find_max_diag(arr):
    maxdiag = 0
    for i in range(-10, 10):
        maxdiag_temp = np.max(np.diagonal(arr, offset=i))
        if maxdiag_temp > maxdiag:
            maxdiag = maxdiag_temp
            offset = i
    return offset


def FFT_filter(arr, pad, filter_pos, filter_width, filter_func=tophat_filter, centering_offset=0):
    arr       = np.pad(arr, pad)
    idx_arr   = np.linspace(0, len(arr), len(arr), dtype=int)
    arr_fft   = np.fft.fftshift(np.fft.fft(arr))
    if filter_pos == 'left':
        ctr_idx   = np.where(arr_fft==np.max(arr_fft[0:int(len(arr_fft)/2)-100]))[0][0]
    if filter_pos == 'right':
        ctr_idx   = np.where(arr_fft==np.max(arr_fft[int(len(arr_fft)/2)+100:]))[0][0]
    elif filter_pos == 'centre':
        ctr_idx   = int(len(arr_fft)/2) # to get middle section, which gives the spectrum
    filter    = filter_func(idx_arr, ctr_idx, filter_width)
    arr_fft_f = arr_fft*filter # apply filter
    max_idx   = np.where(np.abs(arr_fft_f)==np.max(np.abs(arr_fft_f)))[0][0]
    idx_diff  = round(len(arr_fft_f)/2 - max_idx)
    arr_fft_f_centered = np.roll(arr_fft_f, idx_diff - centering_offset)
    arr_ifft     = np.fft.ifft(np.fft.ifftshift(arr_fft_f_centered))
    arr_ifft     = arr_ifft[pad:-pad]

    return arr_fft, arr_ifft, idx_arr, filter


def get_diagonal_indices(Φ):
    """
    Tracks the maximum value across the JSA to return x and y indices. Using this instead of built-in np.diag functions because
    those functions wouldn't get the curvature of the JSA
    """
    diag_x = np.arange(0, Φ.shape[0], 1)
    diag_y = np.argmax(Φ, axis=0)
    return (diag_y, diag_x)


def ppKTP_Sellmeier(λ_signal, λ_idler, λ_pump=405.5, dl=0.1, T=24, L=1e-3):
    """ Parameters for Sellmeier equations of PPKTP crystal from DOI: 10.1134/S1054660X12080142 """
    λp = λ_pump
    λs = λ_signal
    λi = λ_idler

    A = 2.12725
    B = 1.18431
    C = 5.14852e-2
    D = 0.6603
    E = 100.00507
    F = 9.68956e-3

    a0 = 9.9587e-6
    a1 = 9.9228e-6
    a2 = -8.9603e-6
    a3 = 4.1010e-6
    b0 = -1.1882e-8
    b1 = 10.459e-8
    b2 = -9.8136e-8
    b3 = 3.1481e-8

    λs = λs * 1e-9 # to m
    λi = λi * 1e-9 # to m

    ns = np.sqrt(A + B/(1-C/(λs*1e6)**2) + D/(1-E/(λs*1e6)**2) - F*(λs*1e6)**2)
    n1 = a0 + a1/(λs*1e6) + a2/(λs*1e6)**2 + a3/(λs*1e6)**3
    n2 = b0 + b1/(λs*1e6) + b2/(λs*1e6)**2 + b3/(λs*1e6)**3
    dns = n1*(T-25) + n2*(T-25)
    ns = ns+dns

    ni = np.sqrt(A + B/(1-C/(λi*1e6)**2) + D/(1-E/(λi*1e6)**2) - F*(λi*1e6)**2)
    n1 = a0 + a1/(λi*1e6) + a2/(λi*1e6)**2 + a3/(λi*1e6)**3
    n2 = b0 + b1/(λi*1e6) + b2/(λi*1e6)**2 + b3/(λi*1e6)**3
    dni = n1*(T-25) + n2*(T-25)
    ni = ni+dni

    λp = λp*1e-9

    npump = np.sqrt(A + B/(1-C/(λp*1e6)**2) + D/(1-E/(λp*1e6)**2) - F*(λp*1e6)**2)
    n1 = a0 + a1/(λp*1e6) + a2/(λp*1e6)**2 + a3/(λp*1e6)**3
    n2 = b0 + b1/(λp*1e6) + b2/(λp*1e6)**2 + b3/(λp*1e6)**3
    dnpump = n1*(T-25) + n2*(T-25)
    npump = npump+dnpump

    th = 1/180*π
    Pol0 = (3.425e-6)/np.cos(th)
    a = 6.7e-6
    b = 11e-9

    Pol = Pol0*(1 + a*(T-25) + b*(T-25)**2)

    [ls, li] = np.meshgrid(λs, λi)
    [ns, ni] = np.meshgrid(ns, ni)

    dk = (npump/λp - ns/ls - ni/li - 1/Pol)

    """ Phase matching equation (eq. 1 in HOM spectral paper) """
    phi = (np.sin(L*dk)/(L*dk))

    dl = dl * 1e-9 # pump bandwidth?

    """ Pump envelope (eq. 2 in HOM spectral paper) """
    p = np.exp(-((λp-li*ls/(li+ls))/(np.sqrt(2)*dl))**2)

    JSA = p*phi

    return p, phi, JSA, ls, li


class ppKTP(object):
    """ Parameters for Sellmeier equations of PPKTP crystal from DOI: 10.1134/S1054660X12080142 """
    def __init__(
        self,
        T=24,
        L=1e-3,
        pp = (3.425),
        th=(1/180)*π,
        **kwargs
    ):
        self.T = T
        self.L = L
        self.pp = pp

        self._A = 2.12725
        self._B = 1.18431
        self._C = 5.14852e-2
        self._D = 0.6603
        self._E = 100.00507
        self._F = 9.68956e-3

        self._a0 = 9.9587e-6
        self._a1 = 9.9228e-6
        self._a2 = -8.9603e-6
        self._a3 = 4.1010e-6
        self._b0 = -1.1882e-8
        self._b1 = 10.459e-8
        self._b2 = -9.8136e-8
        self._b3 = 3.1481e-8

        self.th = th
        self.Pol0 = (self.pp*1e-6)/np.cos(self.th)
        self._a = 6.7e-6
        self._b = 11e-9
        self.Pol = self.Pol0 *(1+self._a*(T-25)+self._b*(T-25)**2)

    def ne(self, λ):
        return np.sqrt(self._A + self._B/(1-self._C/(λ)**2) + self._D/(1-self._E/(λ)**2) - self._F*(λ)**2)

    def n1(self, λ):
        return self._a0 + self._a1/(λ) + self._a2/(λ)**2 + self._a3/(λ)**3

    def n2(self, λ):
        return self._b0 + self._b1/(λ) + self._b2/(λ)**2 + self._b3/(λ)**3

    def Δne(self, λ):
        return self.n1(λ)*(self.T-25) + self.n2(λ)*(self.T-25)

    def n(self, λ):
        return self.ne(λ) + self.Δne(λ)

class Type0SPDC(object):
    def __init__(
        self,
        crystal,
        pump_σ
    ):
        self.JSA = None
        self.pump_env = None
        self.phi = None
        self.crystal = crystal
        self.pump_σ = pump_σ

    # def phasematch(self, λp, λs, λi):
    #     # Convert nm to μm
    #     λs = λs * 1e-3
    #     λi = λi * 1e-3
    #     λp = λp * 1e-3
    #
    #     ns = self.crystal.n(λs)
    #     ni = self.crystal.n(λi)
    #     n_p = self.crystal.n(λp)
    #
    #     λs = λs * 1e-6
    #     λi = λi * 1e-6
    #     λp = λp * 1e-6
    #
    #     [ls, li] = np.meshgrid(λs, λi)
    #     [ns, ni] = np.meshgrid(ns, ni)
    #
    #     dk = (n_p/λp - ns/ls - ni/li - 1/self.crystal.Pol)
    #
    #     """ Phase matching equation (eq. 1 in HOM spectral paper) """
    #     phi = (np.sin(self.crystal.L*dk) / (self.crystal.L*dk))
    #     # phi = (np.sin(self.crystal.L*dk) / (self.crystal.L*dk))
    #
    #     dl = self.pump_σ*1e-9  # pump bandwidth?
    #
    #     """ Pump envelope (eq. 2 in HOM spectral paper) """
    #     p = np.exp(-((λp-li*ls/(li+ls))/(np.sqrt(2)*dl))**2)
    #
    #     self.phi = phi
    #     self.pump_env = p
    #     self.JSA = p*phi

    def phasematch(self, λp, λs, λi):
        [λ_s, λ_i] = np.meshgrid(λs, λi)
        λ_p = 1/(1/λ_s + 1/λ_i)

        n_p = self.crystal.n(λ_p*1e-3)
        n_s = self.crystal.n(λ_s*1e-3)
        n_i = self.crystal.n(λ_i*1e-3)
        # dk = (1/c)*(n_p*ωp - n_s*ωs - n_i*ωi - self.crystal.Pol)
        dk = (n_p/λ_p - n_s/λ_s - n_i/λ_i - 1/self.crystal.Pol)
        # L = self.crystal.L
        # phi = np.exp(-1j*dk*L/2)*np.sinc(dk*L/2)

        """ Phase matching equation (eq. 1 in HOM spectral paper) """
        phi = (np.sin(self.crystal.L*dk) / (self.crystal.L*dk))

        dl = self.pump_σ

        """ Pump envelope (eq. 2 in HOM spectral paper) """
        p = np.exp(-((λp-λ_i*λ_s/(λ_i+λ_s))/(np.sqrt(2)*dl))**2)

        self.phi = phi
        self.pump_env = p
        self.JSA = p*phi
        self.λ_s = λ_s
        self.λ_i = λ_i


class TypeΙΙSPDC(object):
    def __init__(
        self,
        crystal,
        pump_σ
    ):
        self.JSA = None
        self.pump_env = None
        self.phi = None
        self.crystal = crystal
        self.pump_σ = pump_σ

    def phasematch(self, λp, λs, λi, crystal_angle):
        [λ_s, λ_i] = np.meshgrid(λs, λi)
        λ_p = 1/(1/λ_s + 1/λ_i)
        ωs = wl2angfreq(λ_s*1e-9)
        ωi = wl2angfreq(λ_i*1e-9)
        ωp = wl2angfreq(λ_p*1e-9)

        n_p = self.crystal.n(λ_p*1e-3, crystal_angle, pol='e')
        n_s = self.crystal.n(λ_s*1e-3, crystal_angle, pol='o')
        n_i = self.crystal.n(λ_i*1e-3, crystal_angle, pol='e')

        dk = (1/c)*(n_p*ωp - n_s*ωs - n_i*ωi)

        """ Phase matching equation (eq. 1 in HOM spectral paper) """
        # phi = (np.sin(self.crystal.L*dk) / (self.crystal.L*dk))

        L = self.crystal.L
        phi = np.exp(-1j*dk*L/2)*np.sinc(dk*L)

        dl = self.pump_σ  # pump bandwidth?

        """ Pump envelope (eq. 2 in HOM spectral paper) """
        p = np.exp(-((λp-λ_i*λ_s/(λ_i+λ_s))/(np.sqrt(2)*dl))**2)

        self.phi = phi
        self.pump_env = p
        self.JSA = p*phi


if __name__ == '__main__':

    # res = 256
    # λp = 405.5  # Pump wavelength
    # λs, λi = generate_signal_idler_wavelengths(res, λp, 770, offset=0.)
    #
    # # pump_envelope, phi, , _, _ = ppKTP_Sellmeier(λ_pump=λp, λ_signal=λs, λ_idler=λi, dl=0.5)
    #
    # spdc = Type0SPDC(
    #     crystal=ppKTP(T=24, L=1e-3, pp=3.425),
    #     pump_σ=0.1,
    # )
    #
    # spdc.phasematch(λp, λs, λi)
    #
    # plt.imshow(spdc.JSA)
    #
    # x = np.arange(0, res, 1)
    # plt.plot(x, x, lw=2, color='k')

    def load_absorption_data(fname, root='Data/', delimiter=''):
        data = np.genfromtxt(root+fname, delimiter=delimiter)
        λ = data[:, 0]
        α = data[:, 1]
        λ, α = zip(*sorted(zip(λ, α))) # to correct any ordering mistakes from plot digitization
        λ = np.array(λ)
        α = np.array(α)
        return λ, α

    def interpolate_spectrum(wl, abs, wl_new):
        abs_interp = np.interp(wl_new, wl, abs)
        return abs_interp

    wl, abs = load_absorption_data('ZnNc_shifted.txt')
    plt.plot(wl, abs)

    # wl = wl + 137
    # plt.plot(wl, abs)
    #
    # np.savetxt('ZnNc_shifted.txt', np.transpose([wl, abs]), delimiter='\t', fmt='%1.3f')
    # abs_interp = interpolate_spectrum(wl, abs, λs)


    # # Testing BBO
    # import ndispers as nd
    # λp = 405  # Pump wavelength
    # λs, λi = generate_signal_idler_wavelengths(res, λp, 800, offset=0.)
    # spdcII = TypeΙΙSPDC(
    #     crystal=nd.media.crystals.AlphaBBO(L=1e-3),
    #     pump_σ=30,
    # )
    # spdcII.phasematch(λp, λs, λi, crystal_angle=deg2rad(40.8))
    # JSA = spdcII.JSA.real
    # plt.imshow(JSA*JSA.T)