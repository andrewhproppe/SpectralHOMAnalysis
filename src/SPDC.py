import numpy as np
from HOM_util import π, c, wl2angfreq

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

    th = 1/180*np.pi
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
        th=(1/180)*np.pi,
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