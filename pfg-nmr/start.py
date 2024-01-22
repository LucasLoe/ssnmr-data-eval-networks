from PFG_FIT_ROUTINE import main
import os

# change working directory to the directory of this file
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
#

file_str = '101'
folder_str = f'./{file_str}'
grad_str = f'./{folder_str}/difflist'
save_str = file_str.replace('/','-')

stop_for_spectral_correction= False
region_summation_eval = True
slicing_eval = False

protocol_1_dict = {
    'cut_pts_or_auto': 'auto',
    'num_of_zeros': 32678,
    'lb':  0.0005,
    'p0': -6,
    'p1_or_None':  None,
    'ppm_shift': -2,
    'bl_left':  10,
    'bl_right': -5,
    'blspread': 30,
    'grad_start': 0,
    'grad_end': -1
}

main = main.Main(
    file_str=file_str,
    folder_str=folder_str,
    save_str=save_str,
    grad_str=grad_str,
    protocol=protocol_1_dict,
    stop_for_spectral_correction=stop_for_spectral_correction,
    region_summation_eval=region_summation_eval,
    slicing_eval=slicing_eval
)

main.run()
