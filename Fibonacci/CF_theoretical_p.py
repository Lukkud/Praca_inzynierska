import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import os
import sys
import time
from CF_generator import cf_transform, cf_projection


DIR_PATH = Path(os.path.abspath(__file__)).parents[0]
PLOT_PATH = os.path.join(DIR_PATH, 'plot_files')
DATA_PATH = os.path.join(DIR_PATH, 'data_files')


class CFteo_stat:
    def __init__(self, at, func, bins=250):
        self.tau = (1 + np.sqrt(5)) * 0.5
        self.k0 = 2 * np.pi * self.tau ** 2 / (1 + self.tau ** 2)
        self.net_const = 2 * np.pi / self.k0
        self.at = at
        self.func = func
        self.bins = bins

        self.calculating()

    def calculating(self):
        if self.func == 1:
            cf_transform(self.at)
            fibonacci_str = pd.read_csv(os.path.join(DATA_PATH, 'fibo_transform.csv'), index_col=0)
        elif self.func == 2:
            cf_projection(self.at)
            fibonacci_str = pd.read_csv(os.path.join(DATA_PATH, 'fibo_projection.csv'), index_col=0)
        else:
            print('Choose 1 to use cf_transform or 2 to use cf_projection')
            sys.exit(0)

        regular_df = pd.DataFrame({'number': np.arange(0, fibonacci_str['string'].iloc[-1] + 10, self.net_const)})

        stat_num = []
        for i in fibonacci_str['string']:
            # print('Complete: {}%'.format((i/fibonacci_str['string'].iloc[-1]*100).round(2)))
            regular_df['stat'] = regular_df['number'] - i
            regular_df['abs_stat'] = abs(regular_df['number'] - i)
            stat_num.append(regular_df['stat'].loc[regular_df[['abs_stat']].idxmin()].values)

        self.plotting(stat_num)

    def plotting(self, stat_num):
        hist, bins = np.histogram(stat_num, bins=np.linspace(-1, 1, self.bins), density=True)
        plt.rcParams.update({'font.size': 22})
        plt.figure(figsize=(15, 10))
        plt.plot(bins[:-1], hist, "-", color="black")
        plt.xlabel(r"$u$")
        plt.ylabel(r"$P(u)$")
        plt.grid(True)
        plt.savefig(os.path.join(PLOT_PATH,'CF_theoretical_p_stat.png'), format='png')
        plt.show()


tic = time.time()
x = CFteo_stat(50000, 1, 250)
toc = time.time()
print(toc - tic)
