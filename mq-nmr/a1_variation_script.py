import matplotlib as mpl
import os
import numpy as np
import pandas as pd
from lmfit import Parameters
import matplotlib.pyplot as plt
import mqnmr_functions as mqn
import random


styledUnicodeDict = {
    "success": "\033[1;38;5;82m\u2714\033[0m",   # Light neon green checkmark
    "failure": "\033[1;38;5;196m\u2717\033[0m",  # Light neon red cross
    "process": "\033[1;38;5;87m\u25B6\uFE0E\033[0m"  # Light neon yellow arrow
}


abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)


def smooth_heaviside(diff_in_vals, amplitude, steepness):
    return 1 + amplitude * 0.5 * (1 + np.tanh((diff_in_vals) / steepness))


def init_fit_params_randomized(a1_set, tv):

    fit_params = Parameters()

    fit_params.add('a1', value=a1_set, vary=False)
    fit_params.add('a2', value=random.uniform(0.1, 0.6), min=0.01, max=0.6)
    fit_params.add('a4', value=tv[0], min=0.9*tv[0], max=1.1*tv[0])
    fit_params.add('a3', expr='1-a1-a2-a4', min=0.01)

    fit_params.add('rdc1', value=random.uniform(0.04, 0.1), min=0.02, max=0.7)
    fit_params.add('rdc2', value=random.uniform(
        0.01, 0.04), min=0.005, max=0.04)
    fit_params.add('rdc3', value=random.uniform(
        0.001, 0.01), min=0.0005, max=0.007)

    fit_params.add('t21', value=random.uniform(3, 15), min=3, max=30)
    fit_params.add('t22', value=random.uniform(35, 80), min=25, max=180)
    fit_params.add('t23', value=random.uniform(80, 150), min=40, max=250)
    fit_params.add('t24', value=tv[1], min=0.9*tv[1], max=1.1*tv[1])

    fit_params.add('b1', value=random.uniform(1.2, 2.0), min=1.5, max=2.0)
    fit_params.add('b2', value=random.uniform(1.2, 2.0), min=1.5, max=2.0)
    fit_params.add('b3', value=random.uniform(1.0, 2.0), min=1.0, max=2.0)

    return fit_params


def loss_fun_split(parDict, t, ISMQ, IDQ):
    tdq = t[0:len(IDQ)]
    res_isum = (ISMQ - mqn.ismq_fit_fun(t, parDict))
    res_dq = (len(t)/len(tdq))*(IDQ - mqn.idq_fit_fun(tdq, parDict))/max(IDQ)

    if use_heaviside_constraint:
        overlap_penalty = 1
        overlap_penalty *= smooth_heaviside(
            parDict['rdc2']-parDict['rdc1'], 3, 0.01)
        overlap_penalty *= smooth_heaviside(
            parDict['rdc3']-parDict['rdc2'], 3, 0.001)
        overlap_penalty *= smooth_heaviside(
            parDict['t21']-parDict['t22'], 3, 0.5)
        overlap_penalty *= smooth_heaviside(
            parDict['t22']-parDict['t23'], 3, 0.5)
        overlap_penalty *= smooth_heaviside(
            parDict['t23']-parDict['t24'], 3, 0.5)
        res_dq *= overlap_penalty
        res_isum *= overlap_penalty

    return res_dq, res_isum


def loss_fun(parDict, t, ISMQ, IDQ):
    # parDict is the dictionary that I pass with all the values
    tdq = t[0:len(IDQ)]
    res_isum = (ISMQ - mqn.ismq_fit_fun(t, parDict))
    res_dq = (IDQ - mqn.idq_fit_fun(tdq, parDict))/max(IDQ)

    res_total = np.append(res_isum, res_dq)

    if use_heaviside_constraint:
        # easy mode switching by using True/False here
        overlap_penalty = 1
        overlap_penalty *= smooth_heaviside(parDict['rdc2']-parDict['rdc1'], 3, 0.01)
        overlap_penalty *= smooth_heaviside(parDict['rdc3']-parDict['rdc2'], 3, 0.001)
        overlap_penalty *= smooth_heaviside(parDict['t21']-parDict['t22'], 3, 0.5)
        overlap_penalty *= smooth_heaviside(parDict['t22']-parDict['t23'], 3, 0.5)
        overlap_penalty *= smooth_heaviside(parDict['t23']-parDict['t24'], 3, 0.5)
        res_total *= overlap_penalty

    return res_total 


# file_str_array = ["4pc_BP","6pc_BP","10pc_BP","15pc_BP","20pc_BP"]
file_str_array = ["15pc_BP"]

data_dir_str = "./txt data/"
save_dir_folder_name = "fit_results"
save_dir_subfolder_ending = "_heaviside"

a1_limits = [0.05, 0.9]
dq_cutoff_after_ms = 200
num_a1_steps = int((max(a1_limits) - min(a1_limits))/0.01)
num_of_repetitions = 1
x_last_points_for_tail = 30
use_heaviside_constraint = True


print(
    f'\t\n {styledUnicodeDict["process"]} Started MQ-NMR mult-fit routine with the following array: {file_str_array}')

for fileStr in file_str_array:

    print(
        f'\t\n {styledUnicodeDict["process"]} Started subroutine for: {fileStr}')

    save_dir_str = f"{save_dir_folder_name}/{fileStr}{save_dir_subfolder_ending}/"

    if not os.path.exists(f"{save_dir_folder_name}"):
        os.mkdir(f"{save_dir_folder_name}")

    if not os.path.exists(f"{save_dir_folder_name}/{fileStr}{save_dir_subfolder_ending}"):
        os.mkdir(f"{save_dir_folder_name}/{fileStr}{save_dir_subfolder_ending}")
        print(
            f'\t\n {styledUnicodeDict["process"]} Result subdirectory not found. A new one was created \n')
    else:
        print(
            f'\t\n {styledUnicodeDict["process"]} Result subdirectory found. Existing files will be overwritten \n')

    try:

        df = pd.read_csv(data_dir_str + fileStr + ".txt",
                         sep="\t", names=['t', 'iref', 'idq', 'imag'])

        data = mqn.prepare_data(df, dq_cutoff=dq_cutoff_after_ms)

        fitresult_dict = mqn.a1_fit_routine_randomized(
            data=data,
            loss_fun=loss_fun,
            loss_fun_split=loss_fun_split,
            init_fit_params=init_fit_params_randomized,
            a1_stepnumber=num_a1_steps,
            rand_reps=num_of_repetitions,
            tail_index=x_last_points_for_tail,
            a1_limits=a1_limits
        )

        best_opt_params = fitresult_dict['out_matrix'][fitresult_dict['min_idx']].params
        [best_sum_res, best_dq_res] = mqn.calc_fit_res(data, best_opt_params)

        fit_result_df, cutted_matrix = mqn.calculate_fit_errors(
            fitresult_dict, 1.1)

        fit_result_df.to_csv(save_dir_str + fileStr + '_fitparams' + '.txt', header=['parameter', 'fit_value', 'lb_fit', 'ub_fit'],
                             sep='\t')

        result_df, exp_df, best_fitresult_plot_c = mqn.create_result_df(
            data=data,
            best_opt_params=best_opt_params,
            file_str=fileStr
        )

        exp_df.to_csv(save_dir_str + fileStr + '_expData' + '.txt',
                      header=['time', 'I-sum', 'I-DQ', 'I-sum-fit', 'I-DQ-fit', 'comp1-fit', 'comp2-fit', 'comp3-fit'], sep='\t')

        plt.savefig(save_dir_str + "global_fit_" + fileStr + ".jpg", dpi=400)

        a1_plot = mqn.create_a1_resplot(
            a1_list=fitresult_dict['a1_list'],
            res_dic=fitresult_dict['res_dic'],
            file_str=fileStr,
            res_indexmarks=[fitresult_dict['min_idx'],],
            factorial_boundary=1.1
        )
        plt.savefig(save_dir_str + "a1_surface_" + fileStr + ".jpg", dpi=400)

        resplot = mqn.create_resplot(
            data, best_sum_res, best_dq_res, exp_df, fileStr)
        plt.savefig(save_dir_str + "res_plot_" + fileStr + ".jpg", dpi=400)

        conf_boundary_fitplot = mqn.plot_confidence_interval(
            data,  cutted_matrix, best_opt_params)
        plt.savefig(save_dir_str + "conf_boundary_fitplot_" +
                    fileStr + ".jpg", dpi=400)

        print(
            f'\t\n {styledUnicodeDict["success"]} FINISHED SAMPLE WITH FILE STRING: ', fileStr)
        print('--------------------------------------')

    except Exception as err:
        print(
            f'\t\n {styledUnicodeDict["failure"]} The following exception occured at sample ${fileStr}: \n')
        print(err)
        print('--------------------------------------')
