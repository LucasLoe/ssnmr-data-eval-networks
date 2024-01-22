from lmfit import minimize
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from copy import deepcopy
from statistics import stdev

from .PFGFunctionCollection import PFGFunctionCollection
from .PFGFitConfig import PFGFitConfig

class PFGMultiFitRoutine():

    def __init__(self, data, ppm_axis, diff_axis, left_ppm_limit, right_ppm_limit, slice_step_in_ppm, zero_grad_data) -> None:
        self.pfg_function_collection = PFGFunctionCollection()
        self.pfg_fit_config = PFGFitConfig()
        self.raw_data = data
        self.raw_ppm_axis = ppm_axis

        self.slice_step = slice_step_in_ppm
        self.diff_axis = diff_axis

        self.split_limits_index = {
            'left_limit': np.where(self.raw_ppm_axis < left_ppm_limit)[0][0],
            'right_limit': np.where(self.raw_ppm_axis < right_ppm_limit)[0][0]
        }
        self.spread = self._calc_spread()
        self.ppm_axis = self.raw_ppm_axis[self.split_limits_index['left_limit']:self.split_limits_index['right_limit']]

        self.data = deepcopy(data)[:, self.split_limits_index['left_limit']:self.split_limits_index['right_limit']]
        self.zero_grad_data = self.zero_grad_data[self.split_limits_index['left_limit']:self.split_limits_index['right_limit']]
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
        

    def _calc_spread(self):
        ppm_step_width = abs(self.raw_ppm_axis[1]-self.raw_ppm_axis[0])
        print('max.-spread is: ... ' + str(int(self.slice_step // ppm_step_width - 2)))
        return int(self.slice_step // (2*ppm_step_width) - 1)
    

    def _create_id_eval_data(self):
        ppm_range = np.arange(self.ppm_axis[-1], self.ppm_axis[0], self.slice_step)
        idx_array = []

        for p in ppm_range:
            k = min(self.ppm_axis, key=lambda x: abs(x - p))
            test = self.ppm_axis.tolist()
            idx_array.append(test.index(k))

        str_idx_list = [str(round(self.ppm_axis[x], 1)) + " ppm" for x in idx_array]

        ppm_pos = [self.ppm_axis[x] for x in idx_array]

        return {
            'idx_arr': idx_array,
            'ppm_pos': ppm_pos,
            'ppm_range': ppm_range,
            'str_idx_list': str_idx_list
        }
    

    def _splitted_diff_experiments(self):

        idx_data = self._create_id_eval_data()
        idx_array = idx_data['idx_arr']
        
        splitted_diff_exp = np.zeros((len(self.diff_axis), len(idx_array)))
        data_norm_list = np.zeros(len(idx_array))

        for ip, pidx in enumerate(idx_array):
            for ig, g in enumerate(self.diff_axis):
                try:
                    splitted_diff_exp[ig, ip] = np.mean(self.data[ig, pidx - self.spread: pidx + self.spread])  # slicing in gradspace
                    data_norm_list[ip] = np.mean(self.zero_grad_data[pidx - self.spread: pidx + self.spread])
                except IndexError as ierr:
                    splitted_diff_exp[ig, ip] = -1
                    data_norm_list[ip] = -1
            try:
                splitted_diff_exp[:, ip] /= data_norm_list[ip]  # normalisation
            except ValueError as verr:
                print('ValueError encountered in create_splitted_diff_experiments')
                print(verr)

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

        d_est1 = 1e-10
        d_est2=1e-11

        a_1=0.5
        a_2=0.4
        a_off=0.1

        return d_est1, d_est2, a_1, a_2


    def fit_routine_subplots(self):

        split_result_list = []
        splitted_experiments = self._splitted_diff_experiments()
        idx_data = self._create_id_eval_data()
        loss_fun = self.pfg_fit_config.get_lmfit_loss_fun_handle()

        fit_plot, axs = plt.subplots((len(idx_data['idx_arr']) // 5 + 1), 5, figsize=(14, 22))
        axs = axs.ravel()
        failed_fit_count = 0

        for ii, intervals in enumerate(idx_data['idx_arr']):

            try:
                d_est1, d_est2, a_1, a_2 = self._estimate_startparams_for_exp_data(splitted_experiments[:, ii])
                lmfit_fp = self.pfg_fit_config.init_lmfit_params_for_p0exp2(d_est1, d_est2, a_1, a_2)
                out = minimize(loss_fun, lmfit_fp, args=(self.diff_axis, splitted_experiments[:, ii]))
                splitFitPars = []
                for k in out.params.keys(): # type: ignore
                    splitFitPars.append(out.params[k] * 1) # type: ignore

                # keeps order of high / low diff coeff in panda df
                if splitFitPars[2] < splitFitPars[3]:
                    splitFitPars[2], splitFitPars[3] = splitFitPars[3], splitFitPars[2]
                    splitFitPars[0], splitFitPars[1] = splitFitPars[1], splitFitPars[0]

                split_result_list.append(splitFitPars)

            except RuntimeError as rerr:
                print(rerr)
                failed_fit_count += 1
                print("failed fit count: ... " + str(failed_fit_count), end='\r')
                split_result_list.append(np.zeros(5))  # needs to be changed when num of par changes

            except ValueError as verr:
                print(verr)
                failed_fit_count += 1
                print("failed fit count: ... " + str(failed_fit_count), end='\r')
                split_result_list.append(np.zeros(5))  # needs to be changed when num of par changes


            axs[ii].semilogy(self.diff_axis, splitted_experiments[:, ii], 'o', color=self.colorDict['Rich Black'], markerfacecolor="None", markeredgewidth=1.5, label=round(idx_data['ppm_range'][ii], 2))
            axs[ii].semilogy(self.diff_axis, self.pfg_function_collection._y0exp2(self.diff_axis, *split_result_list[ii]), '-', linewidth=2, label='full_fit', color=self.colorDict['Gamboge'])
            axs[ii].legend(loc='upper right')
            # axs[ii].set_ylim([0.1, 1])

        fit_plot.tight_layout()
        self.split_fit_result = split_result_list
        self.result_dataframe = self._create_df_from_result()

        return fit_plot, split_result_list
    

    def _create_df_from_result(self):
        idx_data = self._create_id_eval_data()
        df = pd.DataFrame(self.split_fit_result, columns=['comp1', 'comp2', 'diff1', 'diff2', 'offset'])
        df.insert(1, 'c-sum', df.comp1 + df.offset + df.comp2)
        df['ppmpos'] = idx_data['ppm_pos']
        df['avg-diff'] = (df.comp1.multiply(df.diff1) + df.comp2.multiply(df.diff2)).divide((df.comp1 + df.comp2))
        df.loc[df.comp1 == 0, 'comp2'] = 0
        df.loc[df.offset >= 0.99, ['comp2', 'comp1', 'offset']] = 0

        return df
        

    def create_slice_plot(self):
        idx_data = self._create_id_eval_data()
        temp_ppm_axis = self.ppm_axis
        ppmPos = [temp_ppm_axis[x] for x in idx_data['idx_arr']]

        slice_plot, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10))

        barWidth = 2 * self.spread * (temp_ppm_axis[2] - temp_ppm_axis[1])

        b1 = ax1.bar(ppmPos, self.result_dataframe.offset, width=barWidth, color=self.colorDict['Gamboge'], label="network")
        b2 = ax1.bar(ppmPos, self.result_dataframe.comp2, bottom=self.result_dataframe.offset, width=barWidth, color=self.colorDict['Viridian Green'], label="diff-2")
        b3 = ax1.bar(ppmPos, self.result_dataframe.comp1, bottom=self.result_dataframe.comp2 + self.result_dataframe.offset, width=barWidth, color=self.colorDict['Rich Black'],
                    label="diff-1")

        ax1.set_xlim([self.ppm_axis[0], self.ppm_axis[-1] - 1])
        ax1.set_ylim([0, 1])
        ax1.grid(which='major', visible=True, linestyle=':')
        ax1.grid(which='minor', visible=True, linestyle=':')
        ax1.set_ylabel('Relative Fractions', fontweight='bold', fontsize=15)
        ax1.set_title('PFG spectrum evaluation', fontsize=17, fontweight='bold')
        ax1.legend(loc='upper right')

        ax2.semilogy(ppmPos, self.result_dataframe.diff1, 'o', markerfacecolor='white', markeredgewidth=1.5, zorder=10, color=self.colorDict['Rich Black'],
                    label='diff-1')
        ax2.semilogy(ppmPos, self.result_dataframe.diff2, 'o', markerfacecolor='white', markeredgewidth=1.5, zorder=10, color=self.colorDict['Viridian Green'],
                    label='diff-2')
        ax2.semilogy(ppmPos, self.result_dataframe['avg-diff'], 's', markeredgewidth=1, markersize=4, zorder=10, label='frac-avg-diff',
                    color=self.colorDict['Ruby Red'])

        ax2.set_ylim([1e-13, 1e-8])
        ax2.set_ylabel('Diff.-coeff. / m2s', fontweight='bold', fontsize=15)
        ax2.set_xlim([self.ppm_axis[0], self.ppm_axis[-1] - 1])
        ax2.grid(which='major', visible=True, linestyle=':')
        ax2.grid(which='minor', visible=True, linestyle=':')
        ax2.legend(loc='upper right').set_zorder(20)

        showGradSteps_percent = [0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9]  # show contour line at percent of gradList
        showGradSteps_index = [int(x * len(self.diff_axis)) for x in showGradSteps_percent]

        for ii, i in enumerate(showGradSteps_index):
            ax3.plot(temp_ppm_axis, self.raw_data[i, self.split_limits_index['left_limit']:self.split_limits_index['right_limit']] / np.amax(self.raw_data[1,:]), 'k', alpha=(1 - ii / len(showGradSteps_index)),
                    label='grad: ' + str(int(100*showGradSteps_percent[ii])) + " %", zorder=10)

        for ip in idx_data['idx_arr']:
            ax3.vlines(temp_ppm_axis[ip], 0, 6e8, linestyles='dashed', colors='lightgray', linewidth=1)

        ax3.set_ylim([0, 1])
        ax3.set_xlim([self.ppm_axis[0], self.ppm_axis[-1] - 1])
        ax3.set_xlabel("ppm", fontweight='bold', fontsize=15)
        ax3.legend(loc='upper right')

        slice_plot.align_ylabels()

        return slice_plot


    def print_stats_in_interval(self, lb, la):
        df = self.result_dataframe.dropna(axis=0)
        print(df['ppmpos'])
        print("avg-diff:".ljust(15,' ') + str(df[df.ppmpos.between(la,lb, inclusive='both')]['avg-diff'].mean()))
        print("avg-diff-std:".ljust(15,' ')  + str(stdev(df[df.ppmpos.between(la,lb, inclusive='both')]['avg-diff'])))
        print("diff-1:".ljust(15,' ')  + str(df[df.ppmpos.between(la,lb, inclusive='both')]['diff1'].mean()))
        print("diff-1-std:".ljust(15,' ')  + str(stdev(df[df.ppmpos.between(la,lb, inclusive='both')]['diff1'])))
        print("diff-2:".ljust(15,' ')  + str(df[df.ppmpos.between(la,lb, inclusive='both')]['diff2'].mean()))
        print("diff-2-std:".ljust(15,' ')  + str(stdev(df[df.ppmpos.between(la,lb, inclusive='both')]['diff2'])))








