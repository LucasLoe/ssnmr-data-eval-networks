import numpy as np

class DataEvaluator:
    def __init__(self):
        pass

    @staticmethod
    def tail_fun(t, a, t2):
        return a * np.exp(-(t / t2))

    @staticmethod
    def al_fun(t, rdc):
        return 1 - np.exp(-(2.375 * t * rdc) ** 1.5) * np.cos(3.663 * rdc * t)
    
    @staticmethod
    def rlx_fun(t, t2, b):
        return np.exp(-(t / t2) ** b)
    
    @staticmethod
    def smooth_heaviside(diff_in_vals, amplitude, steepness):
        return 1 + amplitude * 0.5 * (1 + np.tanh((diff_in_vals) / steepness))

    def _ismq_fit_fun(self, t, par_dic):
        a1, a2, a3, a4 = par_dic['a1'], par_dic['a2'], par_dic['a3'], par_dic['a4']
        t21, t22, t23, t24 = par_dic['t21'], par_dic['t22'], par_dic['t23'], par_dic['t24']
        b1, b2, b3 = par_dic['b1'], par_dic['b2'], par_dic['b3']

        return a1 * self.rlx_fun(t, t21, b1) + a2 * self.rlx_fun(t, t22, b2) + a3 * self.rlx_fun(t, t23, b3) + a4 * self.rlx_fun(t, t24, b=1)

    def _idq_fit_fun(self, t, par_dic):
        a1, a2, a3, a4 = par_dic['a1'], par_dic['a2'], par_dic['a3'], par_dic['a4']
        t21, t22, t23, t24 = par_dic['t21'], par_dic['t22'], par_dic['t23'], par_dic['t24']
        b1, b2, b3 = par_dic['b1'], par_dic['b2'], par_dic['b3']
        rdc1, rdc2, rdc3 = par_dic['rdc1'], par_dic['rdc2'], par_dic['rdc3']

        c1 = self.rlx_fun(t, t21, b1) * self.al_fun(t, rdc1)
        c2 = self.rlx_fun(t, t22, b2) * self.al_fun(t, rdc2)
        c3 = self.rlx_fun(t, t23, b3) * self.al_fun(t, rdc3)

        return 0.5 * (a1 * c1 + a2 * c2 + a3 * c3)

    def _idq_comp_fun(self, t, par_dic):
        a1, a2, a3, a4 = par_dic['a1'], par_dic['a2'], par_dic['a3'], par_dic['a4']
        t21, t22, t23, t24 = par_dic['t21'], par_dic['t22'], par_dic['t23'], par_dic['t24']
        b1, b2, b3 = par_dic['b1'], par_dic['b2'], par_dic['b3']
        rdc1, rdc2, rdc3 = par_dic['rdc1'], par_dic['rdc2'], par_dic['rdc3']

        c1 = 0.5 * a1 * self.rlx_fun(t, t21, b1) * self.al_fun(t, rdc1)
        c2 = 0.5 * a2 * self.rlx_fun(t, t22, b2) * self.al_fun(t, rdc2)
        c3 = 0.5 * a3 * self.rlx_fun(t, t23, b3) * self.al_fun(t, rdc3)

        return [c1, c2, c3]
    
    def ismq_fit_fun(self):
        return self._ismq_fit_fun
    
    def idq_fit_fun(self):
        return self._idq_fit_fun
        
    def idq_comp_fun(self):
        return self._idq_comp_fun
    
    