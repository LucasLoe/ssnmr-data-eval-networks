from lmfit import minimize
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from copy import deepcopy

from .PFGFunctionCollection import PFGFunctionCollection
from .PFGFitConfig import PFGFitConfig


class PFGRegionSummationRoutine():

    def __init__(self, data, ppm_axis, diff_axis, left_ppm_limit, right_ppm_limit, zero_grad_data) -> None:
        self.pfg_function_collection = PFGFunctionCollection()
        self.pfg_fit_config = PFGFitConfig()
        self.raw_data = data
        self.raw_ppm_axis = ppm_axis

        self.diff_axis = diff_axis

        self.split_limits_index = {
            'left_limit': np.where(self.raw_ppm_axis < left_ppm_limit)[0][0],
            'right_limit': np.where(self.raw_ppm_axis < right_ppm_limit)[0][0]
        }
        self.ppm_axis = self.raw_ppm_axis[self.split_limits_index['left_limit']:self.split_limits_index['right_limit']]

        self.data = deepcopy(data)[:, self.split_limits_index['left_limit']:self.split_limits_index['right_limit']]

        self.zero_grad_data = zero_grad_data[self.split_limits_index['left_limit']:self.split_limits_index['right_limit']]
        self.split_fit_result = []
        self.result_dataframe = pd.DataFrame()

        self.colorDict = {
            "Rich Black": "#001219",
            "Blue Sapphire": "#005f73",
            "Viridian Green": "#0a9396",
            "Middle Blue Green": "#94d2bd",
            "Medium Champagne": "#e9d8a6",
            "Gamboge": "#ee9b00",
            "Alloy Orange": "#ca6702",
            "Rust": "#bb3e03",
            "Rufous": "#ae2012",
            "Ruby Red": "#9b2226"
            }
    

    def _splitted_diff_experiments(self):
        
        splitted_diff_exp = np.zeros((len(self.diff_axis)))
        normalization_factor = np.sum(self.zero_grad_data)

        for ig, g in enumerate(self.diff_axis):
            try:
                splitted_diff_exp[ig] = np.sum(self.data[ig,:])  # slicing in gradspace                
            except Exception as exc:
                print('Exception encountered in create_splitted_diff_experiments')
                print(exc)

        splitted_diff_exp[:] /= normalization_factor  # normalisation

        return splitted_diff_exp
    

    def _estimate_startparams_for_exp_data(self, single_exp_data):
        p1 = int(0.1*len(self.diff_axis))
        p2 = int(0.5*len(self.diff_axis))
        p3 = int(0.8*len(self.diff_axis))

        d_est1 = np.log(single_exp_data[0]/np.mean(single_exp_data[p1:p1]))/(self.diff_axis[p1]-self.diff_axis[0])
        d_est2 = np.log(np.mean(single_exp_data[p2-1:p2+1])/np.mean(single_exp_data[p3-1:p3+1]))/(self.diff_axis[p3]-self.diff_axis[p2])

        a_off = single_exp_data[int(0.95*len(self.diff_axis))]
        a_2 = np.mean(single_exp_data[p2-1:p2+1]) / np.exp(-d_est2 / self.diff_axis[p2])
        a_1 = 1 - a_off - a_2

        return d_est1, d_est2, a_1, a_2


    def fit_routine(self):

        splitted_experiment = self._splitted_diff_experiments()
        loss_fun = self.pfg_fit_config.get_lmfit_loss_fun_handle()

        fit_plot = plt.figure(figsize=(5,3))

        try:
            d_est1, d_est2, a_1, a_2 = self._estimate_startparams_for_exp_data(splitted_experiment)
            lmfit_fp = self.pfg_fit_config.init_lmfit_params_for_p0exp2(d_est1, d_est2, a_1, a_2)
            out = minimize(loss_fun, lmfit_fp, args=(self.diff_axis, splitted_experiment))
            splitFitPars = []
            for k in out.params.keys(): # type: ignore
                splitFitPars.append(out.params[k] * 1) # type: ignore

            # keeps order of high / low diff coeff in panda df
            if splitFitPars[2] < splitFitPars[3]:
                splitFitPars[2], splitFitPars[3] = splitFitPars[3], splitFitPars[2]
                splitFitPars[0], splitFitPars[1] = splitFitPars[1], splitFitPars[0]

            self.split_fit_result=splitFitPars

        except Exception as exc:
            print(exc)

        plt.semilogy(self.diff_axis, splitted_experiment, 'o', color=self.colorDict['Rich Black'], markerfacecolor="None", markeredgewidth=1.5)
        plt.semilogy(self.diff_axis, self.pfg_function_collection._y0exp2(self.diff_axis, *self.split_fit_result), '-', linewidth=2, label='full_fit', color=self.colorDict['Gamboge'])
        plt.legend(loc='upper right')
        plt.ylim([0.05,1])

        plt.tight_layout()
        self.result_dataframe = self._create_df_from_result()

        exp_df = self._create_exp_data_df(splitted_experiment)

        return fit_plot, self.result_dataframe, exp_df
    

    def _create_df_from_result(self):
        df = pd.DataFrame([self.split_fit_result], columns=['comp1', 'comp2', 'diff1', 'diff2', 'offset'])
        df.insert(1, 'c-sum', df.comp1 + df.offset + df.comp2)
        df['avg-diff'] = (df.comp1.multiply(df.diff1) + df.comp2.multiply(df.diff2)).divide((df.comp1 + df.comp2))

        return df


    def _create_exp_data_df(self, intensity):
        df = pd.DataFrame()
        df['b'] = self.diff_axis
        df['intensity'] = intensity
        
        return df


    def print_stats_in_interval(self, lb, la):
        df = self.result_dataframe.dropna(axis=0)
        print("avg-diff:".ljust(15,' ') + str(df['avg-diff']))
        print("diff-1:".ljust(15,' ')  + str(df['diff1']))
        print("diff-2:".ljust(15,' ')  + str(df['diff2']))





