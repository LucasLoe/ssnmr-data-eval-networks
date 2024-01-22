from MQ_MULTIFIT.main import Main
import os

# Set the current working directory to the directory of the main file
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

### user config starts here ###
sample_arr = ['4pc_BP', '6pc_BP', '10pc_BP', '15pc_BP', '20pc_BP'] # sample names without the .txt
data_dir = "./txt data/" # samples are located here - use relative path notation and end with slash "./folder_name/"
save_dir = "fit_results" # folder for results - string only

protocol = {
    'dq_cutoff_in_ms': 200,
    'init_tail_params': [0.4, 500], # [norm_fraction, t2_in_ms]
    'a1_limits': [0.1, 0.8], # lower and upper limit for the single link fraction variation
    'tail_index': 20, # last X points for estimating the tail fraction
    'rand': False, # if True: repeat the whole procedure with random starting parameters for rand_reps times
    'rand_reps': 1 # ... see above
}
### user config ends here ###

# run
main = Main(
    sample_name_arr=sample_arr,
    data_dir_str=data_dir,
    save_dir_folder_name=save_dir,
    )

main.run(protocol=protocol)