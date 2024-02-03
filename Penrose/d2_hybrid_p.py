import numpy as np
import pandas as pd
from numba import jit
import time
from d2_utils import D2utils


class D2hybridp(D2utils):
    def __init__(self):
        try:
            self.config = self.read_json()['d2_hybrid_p']
            self.tau = (1 + np.sqrt(5)) * 0.5
            self.ux_range = self.config['ux_range']
            self.uy_range = self.config['uy_range']
            self.ux_step = self.config['ux_step']
            self.uy_step = self.config['uy_step']
            self.source_file = self.config["source_file"]
            self.threshold = self.config["threshold"]
            self.u1 = pd.DataFrame(np.arange(-self.ux_range, self.ux_range, self.ux_step), columns=['u1'])
            self.u2 = pd.DataFrame(np.arange(-self.uy_range, self.uy_range, self.uy_step), columns=['u2'])
            self.df_points = self.u1.merge(self.u2, how="cross")
            self.df_pen = self.read_diffraction_data(self.source_file).astype({'fou': complex})
        except KeyError as exc:
            raise Exception(f"No parameter {exc} specified for d2_numerical_p script in config.json file")

    @staticmethod
    @jit(nopython=True)
    def inv_fourier_2d_hybrid(df_points, w1, w2, fou):
        inv_fou = [np.float64(x) for x in range(0)]
        for i in np.arange(len(df_points)):
            print(df_points[i][0], df_points[i][1])
            inv_fou.append((np.sum(np.array([fou[j] * complex(np.cos(df_points[i][0] * w1[j] + df_points[i][1] * w2[j]),
                                                              np.sin(-df_points[i][0] * w1[j] - df_points[i][1] * w2[j]))
                                            for j in np.arange(len(w1))], dtype=np.complex128)) / len(w1)).real)
        return inv_fou

    def execute(self):
        inv_fou = self.inv_fourier_2d_hybrid(self.df_points.to_numpy(),
                                             self.df_pen["w1"].to_numpy(),
                                             self.df_pen["w2"].to_numpy(),
                                             self.df_pen["fou"].to_numpy(dtype=np.complex128))
        self.df_points["inv_fou"] = inv_fou
        self.saving_data_csv(self.df_points, 'd2_hybrid_p.csv')
        self.plotting_p(self.df_points, 'Penrose_plot_p_hybrid', self.threshold)


if __name__ == "__main__":
    tic = time.time()
    x = D2hybridp()
    x.execute()
    toc = time.time()
    print('Overall time: ', round(toc - tic, 2))
