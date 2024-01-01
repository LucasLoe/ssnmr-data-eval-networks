#external dependencies
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lmfit import minimize
from scipy.optimize import curve_fit

# internal dependencies
from nmr_styles import ColorManager, TerminalStyleManager
from mq_functions import DataEvaluator
from misc import ProgressBar

class MQFitRoutine:

    def __init__(self, df, dq_cutoff_in_ms = 100):
        self.df = df # deliver input data as a pandas dataframe with named columns for t, iref, isqm, idq
        self.dq_cutoff = dq_cutoff_in_ms # dq data is cutted after this value in ms

        self.idq = []
        self.ismq = []
        self.t = []

        self.color_manager = ColorManager()
        self.terminal_style_manager = TerminalStyleManager()
        self.mq_functions = DataEvaluator()
        self.progressbar = ProgressBar(prefix='Progress:', suffix='Complete')

        self._prepare_data()


    def _prepare_data(self):
        self.df['isum'] = self.df.iref + self.df.idq
        self.df['idq'] /= self.df.isum[0]
        self.df['isum'] /= self.df.isum[0]
        self.idq = self.df.loc[self.df.t < self.dq_cutoff, 'idq'].to_numpy()  # cut dq at self.df.t > x milliseconds
        self.ismq = self.df['isum'].to_numpy()
        self.t = self.df['t'].to_numpy()

    def get_tail_fit_values(self, tail_fun, x_data, y_data, initial_params):
        tail_vals = curve_fit(f=tail_fun, xdata=x_data, ydata=y_data, p0=initial_params)
        tail_vals = tail_vals[0] # only use fit parameters and discard all additional outputs from scipy
        print(f"\t\n {self.terminal_style_manager.get_style('process')} Estimated tail: \n\t fraction: {round(100 * tail_vals[0],2)} % \n\t T2: {round(tail_vals[1],2)} ms \n" )

        return tail_vals

    def fit_routine(self, loss_fun, loss_fun_split, init_fit_params, init_tail_params=[0.4,500], a1_stepnumber = 100, a1_limits = [0.01, 0.99], tail_index = 20, rand = False, rand_reps = 5):

        a1_list = np.linspace(start=a1_limits[0], stop=a1_limits[1], num=a1_stepnumber)
        residuals: dict[str, list[float]] = {
            'smq': [],
            'dq': [],
            'total': []
        }

        a1_rand_temp_res: dict[str, list[float]] = {
            'smq': [],
            'dq': [],
            'total': []
        }

        rand_temp_fit_result = []
        fit_result = []
        rand_num = rand_reps
        progress_count = 0


        # preliminary tail fit

        tail_vals = self.get_tail_fit_values(
            tail_fun=DataEvaluator.tail_fun,
            x_data=self.t[-tail_index:],
            y_data=self.ismq[-tail_index:],
            initial_params=init_tail_params
        )

        # loop for a1 variation

        if not rand:
                print('Fitting procedure started with randomization flag = false')
                rand_num = 1

        self.progressbar.total = a1_stepnumber * rand_num


        for temp_a1 in a1_list:

            # randomize count variable
            ir = 0

            while ir < rand_num:

                fpar_init = init_fit_params(temp_a1, tail_vals)
                fit_out = minimize(loss_fun, fpar_init, args=(self.t, self.ismq, self.idq))
                loss_dq, loss_isum = loss_fun_split(fit_out.params.valuesdict(), self.t, self.ismq, self.idq) # type: ignore
                a1_rand_temp_res['smq'].append(float(np.linalg.norm(loss_isum))/float(np.max(self.ismq)))
                a1_rand_temp_res['dq'].append(np.linalg.norm(loss_dq)/np.max(self.idq)) # type: ignore
                rand_temp_fit_result.append(fit_out)
                
                ir += 1
                progress_count +=1
                self.progressbar.print_progress(progress_count)

            

            #calculate sum residual
            a1_rand_temp_res['total'] = [(x + y) / 2 for (x, y) in
                            zip(a1_rand_temp_res['smq'], a1_rand_temp_res['dq'])]
            
            # get minimum value within a randomization step
            _ , idx_rand = min((val, idx) for (idx, val) in enumerate(a1_rand_temp_res['total']))

            # append the best estimate for the random repetition to the actual array
            residuals['dq'].append(a1_rand_temp_res['dq'][idx_rand])
            residuals['smq'].append(a1_rand_temp_res['smq'][idx_rand])
            fit_result.append(rand_temp_fit_result[idx_rand])

            #clean temporary stuff
            a1_rand_temp_res = {
                'dq': [],
                'smq': [],
                'total': []
            }

            rand_temp_fit_result = []

        residuals['total'] = [(x + y) / 2 for (x, y) in
                            zip(residuals['smq'], residuals['dq'])]


        val, idx = min((val, idx) for (idx, val) in enumerate(residuals['total']))


        return {
        "out_matrix": fit_result,
        "min_idx": idx,
        "min_val": val,
        "a1_list": a1_list,
        "res_dic": {
            "total_res": residuals['total'],
            "sum_res": residuals['smq'],
            "dq_res": residuals['dq'],
         }
        }
    
    def extract_fit_pars_from_minimizer(self, fitresult_dict):
         return fitresult_dict['out_matrix'][fitresult_dict['min_idx']].params
    
    def create_fit_result_dataframe(self, fitresult_dict, factorial_boundary=1.1):
        out_matrix = fitresult_dict['out_matrix']
        total_res = fitresult_dict['res_dic']['total_res']
        min_idx = fitresult_dict['min_idx']

        minval = total_res[min_idx]
        boundary = factorial_boundary * minval

        try:
            lb_idx = np.where(total_res[:min_idx] > boundary)[0][-1]
        except IndexError:
            lb_idx = 0
        try:
            ub_idx = min_idx + np.where(total_res[min_idx:] > boundary)[0][0]
        except IndexError:
            ub_idx = -1

        cutted_out_matrix = out_matrix[lb_idx: ub_idx]

        fit_error_dic = {
            "parameter": [],
            "fit_value": [],
            "lb_fit": [],
            "ub_fit": []
        }

        cutted_matrix_dic = {key: [] for key in out_matrix[0].params.keys()}

        for m in cutted_out_matrix:
            for key in m.params.keys():
                cutted_matrix_dic[key].append(m.params[key] * 1)



        for key in out_matrix[0].params.keys():
            fit_error_dic['parameter'].append(key)
            fit_error_dic['fit_value'].append(out_matrix[min_idx].params[key] * 1)
            fit_error_dic['lb_fit'].append(min(cutted_matrix_dic[key]))
            fit_error_dic['ub_fit'].append(max(cutted_matrix_dic[key]))

        return pd.DataFrame.from_dict(fit_error_dic), cutted_out_matrix
    

    def create_df(self, fit_params):
        keyList = fit_params.keys()
        key_list = []
        value_list = []
        err_list = []

        for k in keyList:
            key_list.append(k)
            value_list.append(fit_params[k] * 1)
            try:
                err_list.append(fit_params[k].stderr * 1)
            except:
                err_list.append('nan')

        result_df = pd.DataFrame(list(zip(key_list, value_list, err_list)), columns=['parameter', 'value', 'std-err'])

        return result_df
    

    def calc_fit_res(self, out):
        fit_sum = self.mq_functions._ismq_fit_fun(self.t, out.valuesdict())
        fit_dq = self.mq_functions._idq_fit_fun(self.t[:len(self.idq)], out.valuesdict())

        res_sum = 100 * (self.ismq - fit_sum) / self.ismq
        res_dq = 100 * (self.idq - fit_dq) / self.idq

        return [res_sum, res_dq]
    

    def calc_predicted_curves(self, fit_params):

        l_isum = self.ismq
        l_idq = self.idq
        l_fit_sum = self.mq_functions._ismq_fit_fun(self.t, fit_params.valuesdict()).tolist()
        l_fit_dq = self.mq_functions._idq_fit_fun(self.t, fit_params.valuesdict()).tolist()
        [comp1, comp2, comp3] = self.mq_functions._idq_comp_fun(self.t, fit_params.valuesdict())
        l_comp1 = comp1
        l_comp2 = comp2
        l_comp3 = comp3

        exp_df = pd.DataFrame(list(zip(self.t, l_isum, l_idq, l_fit_dq, l_fit_sum, l_comp1, l_comp2, l_comp3)),
                            columns=['time', 'I-sum', 'I-DQ', 'I-sum-fit', 'I-DQ-fit', 'comp1-fit', 'comp2-fit',
                                    'comp3-fit'])

        fit_res_plot = plt.figure(figsize=(5, 4))
        plt.semilogy(self.t, l_isum, 'ko', markerfacecolor='none', markeredgewidth=1.5, markersize=4, label=r'$I_{\Sigma MQ}$')
        plt.semilogy(self.t[:len(self.idq)], l_idq, 'ks', markerfacecolor='none', markeredgewidth=1.5, markersize=4, label=r'$I_{DQ}$')
        plt.semilogy(self.t, l_fit_sum, '-', color=self.color_manager.get_color_png('Salmon Red'), linewidth=1.5, label='global_fit')
        plt.semilogy(self.t, l_fit_dq, '-', color=self.color_manager.get_color_png('Salmon Red'), linewidth=1.5, label='_nolegend_')
        plt.semilogy(self.t, l_comp1, '-', color=self.color_manager.get_color_png('Light Blue'), linewidth=1.5, label='SL')
        plt.semilogy(self.t, l_comp2, '-.', color=self.color_manager.get_color_png('Light Blue'), linewidth=1.5, label='DL')
        plt.semilogy(self.t, l_comp3, ':', color=self.color_manager.get_color_png('Light Blue'), linewidth=1.5, label='HOC')
        plt.xlabel('DQ evolution time / ms', fontsize=12)
        plt.ylabel('Norm. intensity / a.u.', fontsize=12)
        plt.ylim(0.001, 1)
        plt.xlim(-2, 300)
        plt.tick_params(axis='both', which='major', labelsize=12)
        plt.legend( loc='lower right',fontsize=12)
        plt.tight_layout()

        return exp_df, fit_res_plot
    

    def create_a1_plot(self, fitresult_dict, file_str , factorial_boundary):

        a1_list = fitresult_dict['a1_list']
        res_dic = fitresult_dict['res_dic']
        res_indexmarks = [fitresult_dict['min_idx']]

        a1_resplot = plt.figure(figsize=(7,5))
        plt.plot(a1_list, res_dic["sum_res"], 's-',markersize=4, color=self.color_manager.get_color_png('Dark Blue'), label=r"$I_{\Sigma MQ}$ res")
        plt.plot(a1_list, res_dic["dq_res"], 'o-',markersize=4, color=self.color_manager.get_color_png('Light Blue'), label=r"$I_{DQ}$ res")
        plt.plot(a1_list, res_dic["total_res"], 'x-',markersize=4, color=self.color_manager.get_color_png('Salmon Red'), label="avg. res")
        plt.hlines(y=factorial_boundary * min(res_dic["total_res"]), xmin=0, xmax=1, linestyle='dotted',
                color=self.color_manager.get_color_png('Salmon Red'), linewidth=1.5, label='confidence interval')

        for ii in res_indexmarks:
            plt.vlines(x=a1_list[ii], ymin=0, ymax=res_dic["total_res"][ii], linestyle='dashed',
                    color=self.color_manager.get_color_png('Salmon Red'), linewidth=1, label='_nolabel_')

        plt.title(f'Exemplary residual surface', fontsize=16)
        plt.legend(fontsize=14, loc='upper right')
        plt.tick_params(axis='both', which='major', labelsize=14)
        plt.ylim(0, 2*np.mean(res_dic["total_res"]))
        plt.xlim(0, 1)
        plt.xlabel(r'$a_1$ fraction (single links)', fontsize=16)
        plt.ylabel('RMSE', fontsize=16)
        plt.tight_layout()

        return a1_resplot
    

    def create_res_plot(self, best_opt_params):

        res_plot, (ax2) = plt.subplots(1, 1, figsize=(7, 5))

        [rs, rdq] = self.calc_fit_res(best_opt_params)

        ax2.plot(self.t, rs, 'o-',color=self.color_manager.get_color_png('Dark Blue'), markerfacecolor='none', label=r'$I_{\Sigma MQ}$ res')
        ax2.plot(self.t[:len(rdq)], rdq, 'o-', color=self.color_manager.get_color_png('Light Blue'), markerfacecolor='none', label=r'$I_{DQ}$ res')
        ax2.set_xlabel('DQ evolution time / ms', fontsize=16)
        ax2.set_ylabel('Norm. res. / %', fontsize=16)
        ax2.set_ylim(-25, 25)
        ax2.set_xlim(0, 150)
        ax2.legend(fontsize=14,loc='upper right')
        ax2.grid(which='major', linestyle='--')
        ax2.tick_params(axis='both', which='major', labelsize=14)
        plt.title('Percentual deviations of model from data',fontsize=16)
        plt.tight_layout()
        res_plot.align_ylabels()

        return res_plot