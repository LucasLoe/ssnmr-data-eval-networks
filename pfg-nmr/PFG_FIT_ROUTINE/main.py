from .DataLoader import DataLoader
from .PFGMultiFitRoutine import PFGMultiFitRoutine
from .PFGRegionSummationRoutine import PFGRegionSummationRoutine
from .Spectral import Spectral

import sys

class Main():

    def __init__(self, file_str, folder_str, save_str, grad_str, protocol, stop_for_spectral_correction, region_summation_eval, slicing_eval) -> None:
        self.file_str = file_str
        self.folder_str = folder_str
        self.save_str = save_str
        self.grad_str = grad_str
        self.protocol = protocol
        self.stop_for_spectral_correction = stop_for_spectral_correction
        self.region_summation_eval = region_summation_eval
        self.slicing_eval = slicing_eval


    def run(self):
        
        # initialize data
        data_loader = DataLoader(self.folder_str, self.grad_str)
        data_loader.initialize_diff_axis()
        init_data = data_loader.get_full_data()
        #

        # handle spectral data
        spectral = Spectral(bruker_dict=init_data['bruker_dict'], raw_data=init_data['raw_data'], diff_axis = init_data['diff_axis'])
        spectral.protocol_1(**self.protocol) # type: ignore
        spectral_data = spectral.get_full_data()
        #

        # plot data for visualization
        spectral_plot = spectral.get_overview_plot()
        spectral_plot.savefig(f'{self.save_str}_spectral.jpg')
        #


        if self.stop_for_spectral_correction:

            sys.exit('Stopped because stop_for_spectral_correction flag was set to True. Adjust spectral parameters as needed.')


        if self.region_summation_eval:

            pfg_region_summation = PFGRegionSummationRoutine(
                data=spectral_data['data'],
                ppm_axis=spectral_data['ppm_axis'],
                diff_axis=spectral_data['diff_axis'],
                left_ppm_limit=5.0,
                right_ppm_limit=0,
                zero_grad_data=spectral_data['zero_inc_data']
            )

            fit_plot, result_dataframe, exp_dataframe = pfg_region_summation.fit_routine()
            fit_plot.savefig(f'{self.save_str}_fit_plot_region_summ.jpg')
            result_dataframe.to_csv(f'{self.save_str}_result_df',index=False)
            exp_dataframe.to_csv(f'{self.save_str}_exp_df',index=False)
            print(result_dataframe)

            pfg_region_summation.print_stats_in_interval(3,0)


        if self.slicing_eval:

            # initialize 
            pfg_multifit_routine = PFGMultiFitRoutine(
                data=spectral_data['data'],
                ppm_axis=spectral_data['ppm_axis'],
                diff_axis=spectral_data['diff_axis'],
                left_ppm_limit=4,
                right_ppm_limit=-1,
                slice_step_in_ppm=0.2,
                zero_grad_data=spectral_data['zero_inc_data']
            )
            #

            # run fit routine and plots
            fit_plot, split_result_list = pfg_multifit_routine.fit_routine_subplots()
            fit_plot.savefig(f'{self.save_str}_fit_plot.jpg')

            slice_plot = pfg_multifit_routine.create_slice_plot()
            slice_plot.savefig(f'{self.save_str}_slice_plot.jpg')

            print('daasd')

            pfg_multifit_routine.print_stats_in_interval(3,1)

