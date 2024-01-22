# external dependencies
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt

# internal dependencies
from .mq_fit import MQFitRoutine
from .config import Config
from .nmr_styles import TerminalStyleManager

### user config starts here ###

sample_name_arr = ['4pc_BP', '6pc_BP', '10pc_BP', '15pc_BP', '20pc_BP'] # sample names without the .txt
data_dir_str = "./txt data/" # samples are located here - use relative path notation and end with slash "./folder_name/"
save_dir_folder_name = "fit_results" # folder for results - string only
save_dir_subfolder_ending = "_heaviside" # postfix string for result files - string only

protocol = {
    'dq_cutoff_in_ms': 200,
    'init_tail_params': [0.4, 500], # [norm_fraction, t2_in_ms]
    'a1_limits': [0.1, 0.8], # lower and upper limit for the single link fraction variation
    'tail_index': 20, # last X points for estimating the tail fraction
    'rand': False, # if True: repeat the whole procedure with random starting parameters for rand_reps times
    'rand_reps': 1 # ... see above
}

### user config ends here ###

class Main():

    def __init__(self, sample_name_arr, data_dir_str, save_dir_folder_name='results', save_dir_subfolder_ending='') -> None:

        self.config = Config(sample_name_arr)
        self.terminal_styles = TerminalStyleManager()

        self.data_dir_str = data_dir_str
        self.save_dir_folder_name = save_dir_folder_name
        self.save_dir_subfolder_ending = save_dir_subfolder_ending


    def run(self, protocol):

        init_fit_params = self.config.get_fit_param_initializer()
        loss_fun = self.config.get_loss_fun()
        loss_fun_split = self.config.get_loss_fun_split()

        print(
            f'\t\n {self.terminal_styles.get_style("process")} Started MQ-NMR mult-fit routine with the following array: {self.config.file_str_arr}')


        for file_str in self.config.file_str_arr:
            print(
                f'\t\n {self.terminal_styles.get_style("process")} Started subroutine for: {file_str}')
            
            save_dir_str = f"{self.save_dir_folder_name}/{file_str}{self.save_dir_subfolder_ending}/"

            if not os.path.exists(f"{self.save_dir_folder_name}"):
                os.mkdir(f"{self.save_dir_folder_name}")

            if not os.path.exists(f"{self.save_dir_folder_name}/{file_str}{self.save_dir_subfolder_ending}"):
                os.mkdir(f"{self.save_dir_folder_name}/{file_str}{self.save_dir_subfolder_ending}")
                print(
                    f'\t\n {self.terminal_styles.get_style("process")} Result subdirectory not found. A new one was created \n')
            else:
                print(
                    f'\t\n {self.terminal_styles.get_style("process")} Result subdirectory found. Existing files will be overwritten \n')

            try:
                df = pd.read_csv(self.data_dir_str + file_str + ".txt",
                                sep="\t", names=['t', 'iref', 'idq', 'imag'])

                mq_fit = MQFitRoutine(df, protocol['dq_cutoff_in_ms'])

                fitresult_dict = mq_fit.fit_routine(
                    loss_fun=loss_fun,
                    loss_fun_split=loss_fun_split,
                    init_fit_params=init_fit_params,
                    init_tail_params=protocol['init_tail_params'],
                    a1_stepnumber=int((protocol['a1_limits'][1] - protocol['a1_limits'][0])/0.01), # make sure 1% steps are used - change if needed
                    a1_limits=protocol['a1_limits'],
                    tail_index=protocol['tail_index'],
                    rand=protocol['rand'],
                    rand_reps=protocol['rand_reps']
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
                    f'\t\n {self.terminal_styles.get_style("success")} FINISHED SAMPLE WITH FILE STRING: ', file_str)
                print('--------------------------------------')
            
            except Exception as err:
                print(
                    f'\t\n {self.terminal_styles.get_style("failure")} The following exception occured at sample ${file_str}: \n')
                print(err)

                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1] # type: ignore
                print(exc_type, fname, exc_tb.tb_lineno) # type: ignore
                print('--------------------------------------')