import os
import random
import numpy as np
from mq_functions import DataEvaluator
from lmfit import Parameters

class Config:
    def __init__(self,file_str_array):
        
        self.data_eval = DataEvaluator()
        self.file_str_arr = file_str_array

 
    def _init_fit_params(self, a1_set, tv):
            fit_params = Parameters()

            fit_params.add('a1', value=a1_set, vary=False)
            fit_params.add('a2', value=random.uniform(0.1, 0.6), min=0.01, max=0.6)
            fit_params.add('a4', value=tv[0], min=0.9*tv[0], max=1.1*tv[0])
            fit_params.add('a3', expr='1-a1-a2-a4', min=0.01)

            fit_params.add('rdc1', value=random.uniform(0.04, 0.1), min=0.02, max=0.7)
            fit_params.add('rdc2', value=random.uniform(0.01, 0.04), min=0.005, max=0.04)
            fit_params.add('rdc3', value=random.uniform(0.001, 0.01), min=0.0005, max=0.007)

            fit_params.add('t21', value=random.uniform(3, 15), min=3, max=30)
            fit_params.add('t22', value=random.uniform(35, 80), min=25, max=180)
            fit_params.add('t23', value=random.uniform(80, 150), min=40, max=250)
            fit_params.add('t24', value=tv[1], min=0.9*tv[1], max=1.1*tv[1])

            fit_params.add('b1', value=random.uniform(1.2, 2.0), min=1.5, max=2.0)
            fit_params.add('b2', value=random.uniform(1.2, 2.0), min=1.5, max=2.0)
            fit_params.add('b3', value=random.uniform(1.0, 2.0), min=1.0, max=2.0)

            return fit_params

    def _loss_fun_split(self, parDict, t, ISMQ, IDQ, use_heaviside_constraint=True):
            tdq = t[0:len(IDQ)]
            res_isum = (ISMQ - self.data_eval._ismq_fit_fun(t, parDict))
            res_dq = (len(t)/len(tdq))*(IDQ - self.data_eval._idq_fit_fun(tdq, parDict))/max(IDQ)

            if use_heaviside_constraint:
                overlap_penalty = 1
                overlap_penalty *= self.data_eval.smooth_heaviside(
                    parDict['rdc2']-parDict['rdc1'], 3, 0.01)
                overlap_penalty *= self.data_eval.smooth_heaviside(
                    parDict['rdc3']-parDict['rdc2'], 3, 0.001)
                overlap_penalty *= self.data_eval.smooth_heaviside(
                    parDict['t21']-parDict['t22'], 3, 0.5)
                overlap_penalty *= self.data_eval.smooth_heaviside(
                    parDict['t22']-parDict['t23'], 3, 0.5)
                overlap_penalty *= self.data_eval.smooth_heaviside(
                    parDict['t23']-parDict['t24'], 3, 0.5)
                res_dq *= overlap_penalty
                res_isum *= overlap_penalty

            return res_dq, res_isum

    def _loss_fun(self, parDict, t, ISMQ, IDQ, use_heaviside_constraint=True):
            tdq = t[0:len(IDQ)]
            res_isum = (ISMQ - self.data_eval._ismq_fit_fun(t, parDict))
            res_dq = (IDQ - self.data_eval._idq_fit_fun(tdq, parDict))/max(IDQ)

            res_total = np.append(res_isum, res_dq)

            if use_heaviside_constraint:
                overlap_penalty = 1
                overlap_penalty *= self.data_eval.smooth_heaviside(parDict['rdc2']-parDict['rdc1'], 3, 0.01)
                overlap_penalty *= self.data_eval.smooth_heaviside(parDict['rdc3']-parDict['rdc2'], 3, 0.001)
                overlap_penalty *= self.data_eval.smooth_heaviside(parDict['t21']-parDict['t22'], 3, 0.5)
                overlap_penalty *= self.data_eval.smooth_heaviside(parDict['t22']-parDict['t23'], 3, 0.5)
                overlap_penalty *= self.data_eval.smooth_heaviside(parDict['t23']-parDict['t24'], 3, 0.5)
                res_total *= overlap_penalty

            return res_total
    
    def get_loss_fun(self):
        return self._loss_fun
    
    def get_loss_fun_split(self):
        return self._loss_fun_split
    
    def get_fit_param_initializer(self):
        return self._init_fit_params
