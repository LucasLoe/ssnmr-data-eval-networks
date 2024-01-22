from lmfit import Parameters
from .PFGFunctionCollection import PFGFunctionCollection

class PFGFitConfig():

    def __init__(self) -> None:
        self.pfg_function_collection = PFGFunctionCollection()

    def init_lmfit_params_for_p0exp2(self, de1, de2, a_1, a_2):

        if de1 != de1:
            de1 = 2*10**-10
        if de2 != de2:
            de2 = 4*10**-11

        if (a_1 < 0.01) or (a_1 > 0.99):
            a_1 = 0.5
            a_2 = 0.45

        if (a_2 < 0.01) or (a_2 > 0.99):
            a_1 = 0.3
            a_2 = 0.3

        lmfit_fp = Parameters()
        lmfit_fp.add('a1', value=0.5 , min=0.01 , max=0.99 )
        lmfit_fp.add('a2', value=0.5 , min=0.01 , max=0.99 )
        lmfit_fp.add('D1', value=1e-9 , min=1*10**-12 , max=5*10**-9 )
        lmfit_fp.add('D2', value=1e-11 , min=1*10**-12 , max=5*10**-9 )
        lmfit_fp.add('offset', value=0, vary=False)

        return lmfit_fp
    

    def _lmfit_loss(self, param_dict, b_axis, signal):
        res = (signal-self.pfg_function_collection.p0exp2(b_axis, param_dict))
        return res
    

    def get_lmfit_loss_fun_handle(self):
        return self._lmfit_loss