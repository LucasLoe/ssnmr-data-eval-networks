import nmrglue as nmr
import numpy as np

class DataLoader:

    def __init__(self, file_str: str, grad_str: str) -> None:
        self.file_str = file_str
        self.bruker_dict, self.raw_data = nmr.bruker.read(file_str)
        self.grad_list = np.loadtxt(grad_str)
        self.file_params: dict
        self.diff_time: float
        self.grad_time: float
        self.diff_axis = []

        self._infer_exp_params()


    def _set_diff_params_from_file(self):
        self.file_params = {
            'td': self.bruker_dict['acqus']['TD'],
            'sw': self.bruker_dict['acqus']['SW_h'],
            'd1': self.bruker_dict['acqus']['DE'],
            'b1': self.bruker_dict['acqus']['BF1'],
            'delays': self.bruker_dict['acqus']['D'],
            'pulses': self.bruker_dict['acqus']['P'],
            'pulse_program': self.bruker_dict['acqus']['PULPROG'],
        }

        
    def _infer_exp_params(self):

        self._set_diff_params_from_file()

        feedback = []
        grad_time: float | None
        delta_diff: float | None

        if self.file_params['pulse_program'] == "diffSe":
            grad_time = round(1000 * (np.sum(self.file_params['delays'][17] / 2 + self.file_params['delays'][16] / 2 + self.file_params['delays'][18])), 1)
            delta_diff = round(1000 * (np.sum(self.file_params['delays'][16:19]) + 2 * self.file_params['delays'][2] + self.file_params['delays'][9] + self.file_params['delays'][11] + self.file_params['delays'][10] + self.file_params['pulses'][19]/10**6), 0) / 2
        elif self.file_params['pulse_program'] == "diffSte":
            grad_time = round(1000 * (np.sum(self.file_params['delays'][17] / 2 + self.file_params['delays'][16] / 2 + self.file_params['delays'][18])), 1)
            delta_diff = round(1000 * (np.sum(self.file_params['delays'][16:19]) + 2 * self.file_params['delays'][2] + self.file_params['delays'][9] + 2 * self.file_params['delays'][11] + self.file_params['delays'][5] + self.file_params['pulses'][19]/10**6), 0)
        elif self.file_params['pulse_program'] == "diffSteBp":
            grad_time = 2*round(1000 * (np.sum(self.file_params['delays'][17] / 2 + self.file_params['delays'][16] / 2 + self.file_params['delays'][18])), 1)
            delta_diff = round(1000 * (2*np.sum(self.file_params['delays'][16:19]) + 2 * self.file_params['delays'][2] + self.file_params['delays'][9] + 2 * self.file_params['delays'][11] + self.file_params['delays'][5] + self.file_params['pulses'][19]/10**6), 0)
        else:
            delta_diff = None
            grad_time = None
            feedback.append("Diffusion time: pulse program not detected! Set diffustion time manually manually! \n")
            feedback.append("Gradient duration: pulse program not detected! Set gradient time manually! \n")

        if grad_time:
            self.grad_time = grad_time
            feedback.append(f'Gradient duration set as {grad_time} ms. Use set_grad_time(value_in_ms) to change it manually.')

        if delta_diff:
            self.diff_time = delta_diff
            feedback.append(f'Diffusion time set as {delta_diff} ms. Use set_diff_time(value_in_ms) to change it manually.')

        if len(feedback) != 0:
            for message in feedback:
                print(message)


    def set_grad_time(self, value_in_ms: float):
        self.grad_time = value_in_ms


    def set_diff_time(self, value_in_ms: float):
        self.diff_time = value_in_ms    


    def initialize_diff_axis(self):
        if self.diff_time is None or self.grad_time is None:
            raise Exception('Initialize diff_time and grad_time first by calling infer_exp_params() or the manual setters set_diff_time() and set_grad_time()')

        self.diff_axis = ((267.522 * 10 ** 6) * self.grad_list * self.grad_time * 0.00001) ** 2 * (self.diff_time * 0.001 - self.grad_time * 0.001 / 3)


    def info(self):
        print('Current file: ... ' + self.file_str)
        print('Current pulse sequence: ... ' + self.bruker_dict['pulse_program'])
        print("Number of gradient steps: ... " + str(len(self.grad_list)))


    def get_full_data(self):
        return {
            'bruker_dict': self.bruker_dict,
            'raw_data': self.raw_data,
            'grad_list': self.grad_list,
            'diff_axis': self.diff_axis,
        }