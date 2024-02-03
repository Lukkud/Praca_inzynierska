import numpy as np
import pandas as pd
import time
from numba import jit
from d2_utils import D2utils


class D2hybridk(D2utils):
    def __init__(self):
        try:
            self.config = self.read_json()['d2_hybrid_k']
            self.tau = (1 + np.sqrt(5)) * 0.5
            self.a = 1
            self.k01 = 2 * np.pi / (5 * self.a)
            self.k02 = 2 * np.pi * self.tau * (self.tau + 2) ** 0.5 / (5 * self.a)
            self.n01_range = self.config['n01_range']
            self.n02_range = self.config['n02_range']
            self.m01_range = self.config['m01_range']
            self.m02_range = self.config['m02_range']
            self.source_file = self.config["source_file"]
            self.threshold = self.config["threshold"]
            self.df_pen = self.read_source_data(self.source_file)

            self.n01 = pd.DataFrame(np.arange(-self.n01_range, self.n01_range), columns=['n1'])
            self.n02 = pd.DataFrame(np.arange(-self.n02_range, self.n02_range), columns=['n2'])
            self.m01 = pd.DataFrame(np.arange(-self.m01_range, self.m01_range), columns=['m1'])
            self.m02 = pd.DataFrame(np.arange(-self.m02_range, self.m02_range), columns=['m2'])
            self.df_points = self.n01.merge(self.n02, how="cross")
            self.df_points = self.df_points.merge(self.m01, how="cross")
            self.df_points = self.df_points.merge(self.m02, how="cross")
            self.df_points["k1"] = (self.df_points["n1"] + self.df_points["n2"] +
                                    (self.df_points["m1"] + self.df_points["m2"]) / self.tau) * self.k01
            self.df_points["k2"] = (self.df_points["n1"] - self.df_points["n2"] +
                                    (self.df_points["m1"] - self.df_points["m2"]) / self.tau) * self.k02
        except KeyError as exc:
            raise Exception(f"No parameter {exc} specified for d2_numerical_k script in config.json file")

    @staticmethod
    @jit(nopython=True)
    def fourier_2d_hybrid(df_points, df_pen):
        fou, iw_fou, = [np.complex128(x) for x in range(0)], [np.float64(x) for x in range(0)]
        for i in np.arange(len(df_points)):
            # print(df_points[i][0], df_points[i][1], df_points[i][2], df_points[i][3])
            fou.append(np.sum(np.array([complex(np.cos(df_points[i][4] * df_pen[j][0] + df_points[i][5] * df_pen[j][1]),
                                                np.sin(df_points[i][4] * df_pen[j][0] + df_points[i][5] * df_pen[j][1]))
                                        for j in np.arange(len(df_pen))], dtype=np.complex128)))
            iw_fou.append(np.absolute(fou[-1]) ** 2 / (len(df_pen) ** 2))
        return fou, iw_fou

    def fourier_2d_hybrid_pandas_style(self):
        """
        Much slower function than numba's version
        """
        self.df_points['fou'] = self.df_points.apply(lambda row: np.sum(np.cos(row["k1"] * self.df_pen["x"] + row["k2"] * self.df_pen["y"]) +
                                                                        1j * np.sin(row["k1"] * self.df_pen["x"] + row["k2"] * self.df_pen["y"])), axis=1)
        self.df_points['iw_fou'] = self.df_points['fou'].abs().pow(2) / (len(self.df_pen) ** 2)

    def execute(self):
        fou, iw_fou = self.fourier_2d_hybrid(self.df_points.to_numpy(), self.df_pen.to_numpy())
        self.df_points["fou"] = fou
        self.df_points["iw_fou"] = iw_fou
        # self.fourier_2d_hybrid_pandas_style()
        self.saving_data_csv(self.df_points, 'd2_hybrid_k.csv')
        self.plotting_k(self.df_points, 'Penrose_plot_k_hybrid', self.threshold)


if __name__ == "__main__":
    tic = time.time()
    x = D2hybridk()
    x.execute()
    toc = time.time()
    print('Overall time: ', round(toc - tic, 2))
