import numpy as np
import os
import time
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from CF_utils import CFutils


np.seterr(divide='ignore', invalid='ignore')
DIR_PATH = Path(os.path.abspath(__file__)).parents[0]
PLOT_PATH = os.path.join(DIR_PATH, 'plot_files')
Path(PLOT_PATH).mkdir(parents=True, exist_ok=True)


class CFteo(CFutils):
    def __init__(self, n, m, save_file=True, plot_group_k=True, plot_k=True, plot_w=True,  inv_fou=True):
        self.tau = (1 + np.sqrt(5)) * 0.5
        self.k0 = 2 * np.pi * self.tau ** 2 / (1 + self.tau ** 2)
        self.q0 = self.k0 / self.tau
        self.k1 = 5 ** 0.5 * self.k0
        self.n_range = n
        self.m_range = m
        self.save_file = save_file
        self.plot_group_k = plot_group_k
        self.plot_k = plot_k
        self.plot_w = plot_w
        self.inv_fou = inv_fou
        self.df = None

    def execute(self):
        num_lst = [[n, m] for n in range(self.n_range) for m in range(self.m_range)]
        num_lst = np.array(num_lst).T
        self.df = pd.DataFrame({'n': num_lst[0], 'm': num_lst[1]})
        self.df['k'] = self.k0 * self.df['n'] + self.q0 * self.df['m']
        self.df['w'] = self.k0 * self.df['n'] + (self.q0 - self.k1) * self.df['m']
        self.df['chi'] = self.k0 * (self.df['n'] - self.tau * self.df['m']) / (2 * self.tau)
        self.df['F(w)'] = np.where(self.df['chi'] != 0, np.sin(self.df['chi']) / self.df['chi'], 1)
        self.df['I(w)'] = np.where(self.df['chi'] != 0, abs(np.sin(self.df['chi']) / self.df['chi']) ** 2, 1)

        self.saving_data(self.df, 'cf_theoretical_values.xlsx') if self.save_file else None
        self.plotting_group_k(self.df) if self.plot_group_k else 0
        self.plotting_k(self.df['k'], self.df['I(w)'], 'CF_theoretical_k.png') if self.plot_k else None
        self.plotting_w(self.df, 'w', 'I(w)', 'CF_theoretical_w.png') if self.plot_w else None
        self.inv_fourier(self.df) if self.inv_fou else None

    def plotting_group_k(self, df):
        k_i = np.reshape(df['k'].tolist(), (self.m_range, self.n_range)).T
        iw_fou_i = np.reshape(df['I(w)'].tolist(), (self.m_range, self.n_range)).T
        plt.rcParams.update({'font.size': 25})
        plt.figure(figsize=(20, 10))

        for m in range(5):
            plt.bar(k_i[m], iw_fou_i[m], label="m={}".format(m), width=0.2)
            k_x = np.linspace(k_i[m][0], k_i[m][-1], 1000)
            plt.plot(k_x,
                     abs(np.sin((k_x - m * self.k1) / (2 * self.tau)) / ((k_x - m * self.k1) / (2 * self.tau))) ** 2,
                     zorder=m+1)
        plt.axis([0, 50, -0.05, 1.05])
        plt.xlabel(r"$k$")
        plt.ylabel(r"$I(k)$")
        plt.grid(True)
        plt.legend(bbox_to_anchor=[0.9, 0.6])
        plt.savefig(os.path.join(PLOT_PATH, 'CF_theoretical_k_group.png'), format='png')
        plt.show()

    def inv_fourier(self, df):
        u = np.arange(-1, 1, 0.001)
        inv_df = pd.DataFrame({'u': u, 'F(u)': np.nan})
        df = df.sort_values('w')
        for ui in range(len(inv_df)):
            df['tmp_feuw'] = df['F(w)'] * (np.cos(inv_df['u'].iloc[ui] * df['w']) -
                                           1j * np.sin(inv_df['u'].iloc[ui] * df['w']))
            inv_df.loc[ui, 'F(u)'] = df['tmp_feuw'].sum() / (self.n_range - 1)

        self.plotting_p(inv_df, 'u', 'F(u)', 'CF_theoretical_p_fourier.png')


if __name__ == "__main__":
    tic = time.time()
    x = CFteo(50, 50)
    x.execute()
    toc = time.time()
    print('Overall time: ', round(toc - tic, 2))
