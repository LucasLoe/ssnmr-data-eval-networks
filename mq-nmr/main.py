import os
import pandas as pd
import matplotlib.pyplot as plt

from mq_fit import MQFitRoutine
from config import Config
from nmr_styles import TerminalStyleManager

config = Config(['15pc_BP'])
terminal_styles = TerminalStyleManager()

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

init_fit_params = config.get_fit_param_initializer()
loss_fun = config.get_loss_fun()
loss_fun_split = config.get_loss_fun_split()

data_dir_str = "./"
save_dir_folder_name = "fit_results"
save_dir_subfolder_ending = "_heaviside"

print(
    f'\t\n {terminal_styles.get_style("process")} Started MQ-NMR mult-fit routine with the following array: {config.file_str_arr}')


for file_str in config.file_str_arr:
    print(
        f'\t\n {terminal_styles.get_style("process")} Started subroutine for: {file_str}')
    
    save_dir_str = f"{save_dir_folder_name}/{file_str}{save_dir_subfolder_ending}/"

    if not os.path.exists(f"{save_dir_folder_name}"):
        os.mkdir(f"{save_dir_folder_name}")

    if not os.path.exists(f"{save_dir_folder_name}/{file_str}{save_dir_subfolder_ending}"):
        os.mkdir(f"{save_dir_folder_name}/{file_str}{save_dir_subfolder_ending}")
        print(
            f'\t\n {terminal_styles.get_style("process")}  Result subdirectory not found. A new one was created \n')
    else:
        print(
            f'\t\n {terminal_styles.get_style("process")}  Result subdirectory found. Existing files will be overwritten \n')

    try:
        df = pd.read_csv(data_dir_str + file_str + ".txt",
                         sep="\t", names=['t', 'iref', 'idq', 'imag'])

        mq_fit = MQFitRoutine(df)

        fitresult_dict = mq_fit.fit_routine(
            loss_fun=loss_fun,
            loss_fun_split=loss_fun_split,
            init_fit_params=init_fit_params,
            init_tail_params=[0.4, 500],
            a1_stepnumber=int((max([0.1]) - min([0.8]))/0.01),
            a1_limits=[0.1, 0.8],
            tail_index=20,
            rand=True,
            rand_reps=4
            )

        best_opt_params = mq_fit.extract_fit_pars_from_minimizer(fitresult_dict)

        fit_result_df, cutted_matrix = mq_fit.create_fit_result_dataframe(fitresult_dict)

        fit_result_df.to_csv(save_dir_str + file_str + '_fitparams' + '.txt', header=['parameter', 'fit_value', 'lb_fit', 'ub_fit'],
                             sep='\t')
        
        res_df = mq_fit.create_df(best_opt_params)

        exp_df, fit_res_plot = mq_fit.calc_predicted_curves(best_opt_params)
        plt.savefig(save_dir_str + "global_fit_" + file_str + ".jpg", dpi=400)
        exp_df.to_csv(save_dir_str + file_str + '_expData' + '.txt',
                      header=['time', 'I-sum', 'I-DQ', 'I-sum-fit', 'I-DQ-fit', 'comp1-fit', 'comp2-fit', 'comp3-fit'], sep='\t')

        a1_plot = mq_fit.create_a1_plot(fitresult_dict, file_str, 1.1)
        plt.savefig(save_dir_str + "a1_surface_" + file_str + ".jpg", dpi=400)

        res_plot = mq_fit.create_res_plot(best_opt_params)
        plt.savefig(save_dir_str + "res_plot_" + file_str + ".jpg", dpi=400)


        print(
            f'\t\n {terminal_styles.get_style("success")} FINISHED SAMPLE WITH FILE STRING: ', file_str)
        print('--------------------------------------')
    
    except Exception as err:
        print(
            f'\t\n {terminal_styles.get_style("failure")} The following exception occured at sample ${file_str}: \n')
        print(err)
        print('--------------------------------------')