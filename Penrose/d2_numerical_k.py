import numpy as np
import pandas as pd
import time
import numba as nb
from pathlib import Path
import os
from d2_utils import D2utils

DIR_PATH = Path(os.path.abspath(__file__)).parents[0]
SOURCE_PATH = os.path.join(DIR_PATH, 'Source')


class D2num(D2utils):
    def __init__(self):
        try:
            self.config = self.read_json()['d2_numerical_k']
            self.kx_range = self.config['kx_range']
            self.ky_range = self.config['ky_range']
            self.kx_step = self.config['kx_step']
            self.ky_step = self.config['ky_step']
            self.source_file = self.config["source_file"]
            self.threshold = self.config["threshold"]
            self.df_pen = self.read_source_data(self.source_file)
            self.kx = np.arange(0, self.kx_range, self.kx_step)
            self.ky = np.arange(0, self.ky_range, self.ky_step)
        except KeyError as exc:
            raise Exception(f"No parameter {exc} specified for d2_numerical_k script in config.json file")

    @staticmethod
    @nb.jit(nopython=True)
    def fourier_2d(kx, ky, df_pen):
        fou, iw_fou = [], []
        k1, k2 = [], []
        for ki in kx:
            for kj in ky:
                k1.append(ki)
                k2.append(kj)
                fou.append(np.sum(np.array([complex(np.cos(ki * df_pen[i][0] + kj * df_pen[i][1]),
                                                    np.sin(ki * df_pen[i][0] + kj * df_pen[i][1]))
                                            for i in np.arange(len(df_pen))], dtype=np.complex128)))
                iw_fou.append(np.absolute(fou[-1])**2 / (2*(len(df_pen)**2)))
                print(ki, kj)
        return k1, k2, fou, iw_fou

    def execute(self):
        k1, k2, fou, iw_fou = self.fourier_2d(self.kx, self.ky, self.df_pen.to_numpy())
        k1, k2, iw_fou, fou = zip(*sorted(zip(k1, k2, iw_fou, fou)))
        df_output = pd.DataFrame({"k1": k1, "k2": k2, "iw_fou": iw_fou, "fou": fou})
        self.saving_data(df_output, 'd2_numerical.xlsx')
        self.plotting_k_numerical_penrose(df_output, 'Penrose_plot_k_scatter', self.threshold)


if __name__ == "__main__":
    tic = time.time()
    x = D2num()
    x.execute()
    toc = time.time()
    print('Overall time: ', round(toc - tic, 2))
