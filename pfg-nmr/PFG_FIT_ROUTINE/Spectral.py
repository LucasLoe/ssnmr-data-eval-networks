import numpy as np
import nmrglue as nmr
import matplotlib.pyplot as plt
from copy import deepcopy

class Spectral:
    

    def __init__(self, bruker_dict, raw_data, diff_axis) -> None:
        self.bruker_dict = bruker_dict
        self.raw_data = raw_data # do not touch this
        self.data = self._initialize_data() # manipulate this
        self.ppm_axis = np.array([])
        self.diff_axis = diff_axis
        self.zero_inc_data = []


    def _initialize_ppm_axis(self):
        freq_axis = (self.bruker_dict['acqus']['SW_h'] / 2) * np.linspace(1, -1, len(self.data[1, :]))
        ppm_axis = freq_axis / self.bruker_dict['acqus']['BF1']
        return np.array(ppm_axis)
    

    def _initialize_data(self):
        data = deepcopy(self.raw_data)
        data = nmr.bruker.remove_digital_filter(self.bruker_dict, data)
        return data
    

    def shift_ppm_axis(self, ppm_shift):
        self.ppm_axis = self.ppm_axis + ppm_shift


    def correct_baseline(self, left_ppm_limit: float, right_ppm_limit: float, blspread_in_points: int):

        if left_ppm_limit is not None:
            a = np.where(self.ppm_axis < left_ppm_limit)[0][0] 
        else:
            a = 0
        if right_ppm_limit is not None:
            b = np.where(self.ppm_axis > right_ppm_limit)[0][-1] # type: ignore
        else:
            b = -blspread_in_points-1

        for index, val in enumerate(self.data[:, 1]):
            difference = (np.mean(self.data[index, a:a + blspread_in_points]) + np.mean(self.data[index, b:b + blspread_in_points]))/2 # type: ignore
            self.data[index, :] = self.data[index, :] - difference
    

    def cut_spectral_range(self, pts_or_auto):
        cutoff_spec = 0

        if pts_or_auto == 'auto':
            cutoff_spec = int(0.5 * len(self.data[0,:]))
        self.data = self.data[:, :-cutoff_spec]
        


    def zerofill(self, num_of_zeros):
        self.data = nmr.proc_base.zf_size(self.data, num_of_zeros)


    def linebroaden(self, line_broad):
        self.data = nmr.proc_base.em(self.data, line_broad)


    def get_first_fid(self):
        return self.data[0,:]
    

    def fft(self):
        self.data = nmr.proc_base.fft(self.data[:])


    def phase_corr(self, p0, p1=None):
        if p1 is None:
            self.data = nmr.proc_base.ps(self.data, p0=p0)
        else:
            self.data = nmr.proc_base.ps(self.data, p0=p0, p1=p1)


    def discard_imag(self):
        self.data = nmr.proc_base.di(self.data)


    def reverse_freq_data(self):
        self.data = nmr.proc_base.rev(self.data)


    def store_zero_inc_data(self):
        # store first data set (no increment in second dimension, index = 0)
        self.zero_inc_data = self.data.copy()[0,:]


    def cut_gradient_range(self, gradient_num_start, gradient_num_end):
        self.data = self.data[gradient_num_start:gradient_num_end,:]
        self.diff_axis = self.diff_axis[gradient_num_start:gradient_num_end]


    def get_overview_plot(self, left_ppm_lim=20, right_ppm_lim=-20):
        overview_plot = plt.figure()

        for index, val in enumerate(self.data[:,1]):
            plt.plot(self.ppm_axis[:], self.data[index, :], linewidth=0.8)

        plt.axhline(xmin=8, xmax=-2, y=0)

        if left_ppm_lim is not None:
            plt.xlim(left=left_ppm_lim)
        if right_ppm_lim is not None:
            plt.xlim(right=right_ppm_lim)

        return overview_plot


    def protocol_1(self,cut_pts_or_auto, num_of_zeros, lb, p0, p1_or_None,ppm_shift, bl_left, bl_right, blspread, grad_start, grad_end):
        self.data = self._initialize_data()
        self.cut_spectral_range(cut_pts_or_auto)
        self.zerofill(num_of_zeros)
        self.linebroaden(lb)
        self.fft()
        self.phase_corr(p0,p1_or_None)
        self.discard_imag()
        self.reverse_freq_data()

        self.ppm_axis = self._initialize_ppm_axis()
        self.shift_ppm_axis(ppm_shift)

        self.correct_baseline(bl_left, bl_right, blspread)
        self.store_zero_inc_data()
        self.cut_gradient_range(grad_start, grad_end)


    def get_full_data(self):
        return {
            'bruker_dict': self.bruker_dict,
            'data': self.data,
            'ppm_axis': self.ppm_axis,
            'diff_axis': self.diff_axis,
            'zero_inc_data': self.zero_inc_data
        }
    