import numpy as np
import pandas as pd
import time
from d2_utils import D2utils


class D2hybridw(D2utils):
    def __init__(self):
        try:
            self.config = self.read_json()['d2_hybrid_w']
            self.tau = (1 + np.sqrt(5)) * 0.5
            self.border_1 = self.config["border_1"]
            self.border_2 = self.config["border_2"]
            self.source_file = self.config["source_file"]
            self.threshold = self.config["threshold"]
            self.df_points = self.read_diffraction_data(self.source_file).astype({'fou': complex})
        except KeyError as exc:
            raise Exception(f"No parameter {exc} specified for d2_numerical_k script in config.json file")

    def execute(self):
        self.df_points["w1"] = 4 * np.pi * self.tau * np.cos(2 * np.pi / 5) * (self.df_points["n1"] + self.df_points["n2"] + (self.df_points["m1"] + self.df_points["m2"]) / self.tau)
        self.df_points["w2"] = 4 * np.pi * self.tau * np.sin(2 * np.pi / 5) * (self.df_points["n1"] - self.df_points["n2"] + (self.df_points["m1"] - self.df_points["m2"]) / self.tau)
        self.df_points["phase"] = self.df_points.apply(lambda row: np.arctan(row["fou"].imag / row["fou"].real), axis=1)
        self.df_points["group"] = self.df_points.apply(lambda row: 3 if (row["phase"] > -self.border_1) and (row["phase"] < self.border_1)
                                                                     else (1 if (row["phase"] < -self.border_2) or (row["phase"] > self.border_2) else 2), axis=1)
        self.saving_data_csv(self.df_points, 'd2_hybrid_w.csv')
        self.plotting_w_phase(self.df_points, 'Penrose_plot_w_phase.png', self.threshold, self.border_1, self.border_2)
        self.plotting_w(self.df_points, 'Penrose_plot_w_hybrid', self.threshold)


if __name__ == "__main__":
    tic = time.time()
    x = D2hybridw()
    x.execute()
    toc = time.time()
    print('Overall time: ', round(toc - tic, 2))
