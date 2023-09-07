from numpy import pi
c      = 299792458  # m/s
π      = pi
hbar   = 6.582119569e-16 # eV s

def wl_to_eV(λ):
    return 1239.8/λ

def wl_to_meV(λ):
    return (1239.8/λ)*1000

def eV_to_wn(x):
    return x*8065.610420

def wn_to_eV(x):
    return x/8065.610420

def wn_to_s(x):
    return 1/(x*2.99792458e10)

def wn_to_ps(x):
    return (1/(x*2.99792458e10))*1e12

def ps_to_wn(x):
    return (1/(x*c/1e12*100))

def mm_to_ps(x):
    return x/c/1000*1e12

def eV_to_ps(x):
    y = eV_to_wn(x)
    return wn_to_ps(y)

# def ps_to_wv(x):
#     return (x/1e12)*2.99792458e10

def eV_to_angfreq(x):
    return x*1.5193e15

def angfreq_to_eV(x):
    return x/1.5193e15

def eV_to_THz(x):
    return x*241.8

def eV_to_Hz(x):
    return eV_to_THz(x)*1e+12

def eV_to_T2(x):
    """ Converts a Lorentzian FWHM in eV from PCFS to a T2 in ps"""
    return (2/eV_to_angfreq(x))*1e12
    # return 1/(eV_to_Hz(x)*pi)*1e12

def T2_to_eV(x):
    return angfreq_to_eV(1/((x/2)*1e-12))

def meV_to_T2(x):
    """ Converts a Lorentzian FWHM in meV from PCFS to a T2 in ps"""
    # return (1/eV_to_angfreq(x))*1e12
    return 1/(eV_to_Hz(x/1000)*pi)*1e12

def wl_to_THz(λ):
    return c*1e-3/λ # 1e9 nm/m, 1e-12 THz/Hz = 1e-3 nm THz

def wl_to_THz_angular(λ):
    return c*1e-3/λ*2*π # 1e9 nm/m, 1e-12 THz/Hz = 1e-3 nm THz

def deg2rad(deg):
    return (π/180)*deg

def wl2angfreq(λ):
    return 2*π*c/λ