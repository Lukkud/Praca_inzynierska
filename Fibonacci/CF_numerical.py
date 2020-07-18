import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import os
import sys
import time
from CF_generator import cf_transform, cf_projection
from numba import jit


DIR_PATH = Path(os.path.abspath(__file__)).parents[0]
PLOT_PATH = os.path.join(DIR_PATH, 'plot_files')
DATA_PATH = os.path.join(DIR_PATH, 'data_files')


class CFnum:
    def __init__(self, at, func, step, eps, tup_range):
        self.at = at
        self.tau = (1 + np.sqrt(5)) * 0.5
        self.k0 = 2 * np.pi * self.tau ** 2 / (1 + self.tau ** 2)
        self.q0 = self.k0 / self.tau
        self.k1 = 5 ** 0.5 * self.k0
        self.step = step
        self.eps = eps
        self.n_range = tup_range[0]
        self.m_range = tup_range[1]
        self.tictoc = []

        if func == 1:
            cf_transform(self.at)
            self.fibonacci_str = pd.read_csv(os.path.join(DATA_PATH, 'fibo_transform.csv'), index_col=0)
        elif func == 2:
            cf_projection(self.at)
            self.fibonacci_str = pd.read_csv(os.path.join(DATA_PATH, 'fibo_projection.csv'), index_col=0)
        else:
            print('Choose 1 to use cf_transform or 2 to use cf_projection')
            sys.exit(0)

        print('Preparing k part')
        self.tictoc.append(time.time())
        self.df_k = pd.DataFrame({'k': np.arange(0, 100, self.step), 'fourier': np.nan, 'int_fourier': np.nan})
        self.df_k['fourier'] = self.fourier(self.df_k['k'].values, self.fibonacci_str['string'].values)
        self.df_k['int_fourier'] = np.absolute(self.df_k['fourier']) ** 2
        CFnum.saving_data_k(self.df_k)
        CFnum.plotting_k(self.df_k)

        print('Preparing w part')
        self.tictoc.append(time.time())
        self.df_w = self.prepare_w(self.df_k)
        CFnum.saving_data_w(self.df_w)
        CFnum.plotting_w(self.df_w)

        print('Preparing p part')
        self.tictoc.append(time.time())
        self.df_p = self.inv_fourier()
        CFnum.saving_data_p(self.df_p)
        CFnum.plotting_p(self.df_p)
        self.tictoc.append(time.time())
        print('Done')
        print('Time first (k) part: ', self.tictoc[1] - self.tictoc[0])
        print('Time second (w) part: ', self.tictoc[2] - self.tictoc[1])
        print('Time third (p) part: ', self.tictoc[3] - self.tictoc[2])

    @staticmethod
    @jit(nopython=True)
    def fourier(k, fibo):
        fou = []
        for ki in k:
            fou.append(np.sum(np.array([complex(np.cos(ki * x), np.sin(ki * x)) for x in fibo])))
        return np.array(fou) / len(fibo)

    def prepare_w(self, df_k):
        df_teo = pd.DataFrame({'k': [n * self.k0 + m * self.q0 for n in range(self.n_range) for m in range(self.m_range)]})
        df_teo['n'] = df_teo.index // self.n_range
        df_teo['m'] = df_teo.index % self.n_range
        df_w = pd.DataFrame({'n': [], 'm': [], 'k': [], 'w': [], 'fourier': [], 'int_fourier': []})

        for index, rows in df_k.iterrows():
            df_teo['tmp'] = abs(df_teo['k'] - rows.k)
            if df_teo['tmp'].min() < self.eps:
                lst_df_teo = df_teo[df_teo.tmp == df_teo.tmp.min()].values[0][1:3]
                df_w = df_w.append(pd.Series(list(lst_df_teo) + [rows.k, rows.k - lst_df_teo[1] * self.k1, rows.fourier, rows.int_fourier], index=df_w.columns), ignore_index=True)
        df_w = df_w.astype({'n': int, 'm': int, 'k': float, 'w': float, 'int_fourier': float})
        return df_w

    def inv_fourier(self):
        u = np.arange(-1, 1, 0.001)
        inv_df = pd.DataFrame({'u': u, 'inv_fourier': np.nan})
        self.df_w = self.df_w.sort_values('w')

        for ui in range(len(inv_df)):
            self.df_w['tmp_prob'] = self.df_w['fourier'] * (np.cos(inv_df['u'].iloc[ui] * self.df_w['w']) -
                                                            1j * np.sin(inv_df['u'].iloc[ui] * self.df_w['w']))
            inv_df.loc[ui, 'inv_fourier'] = self.df_w['tmp_prob'].sum()
        return inv_df

    @staticmethod
    def saving_data_k(df):
        df = df.round(6)
        df.to_excel(os.path.join(DATA_PATH, 'cf_numerical_k_data.xlsx'), index=False)

    @staticmethod
    def saving_data_w(df):
        df = df.round(6)
        df.to_excel(os.path.join(DATA_PATH, 'cf_numerical_w_data.xlsx'), index=False)

    @staticmethod
    def saving_data_p(df):
        df = df.round(6)
        df.to_excel(os.path.join(DATA_PATH, 'cf_numerical_p_data.xlsx'), index=False)

    @staticmethod
    def plotting_k(df):
        plt.rcParams.update({'font.size': 22})
        plt.figure(figsize=(15, 10))
        plt.bar(df['k'].loc[df['int_fourier'] > 0.001], df['int_fourier'].loc[df['int_fourier'] > 0.001], color="black",
                width=0.3)
        plt.axis([-2, 50, -0.05, 1.05])
        plt.xlabel(r"$k$")
        plt.ylabel(r"$I(k)$")
        plt.grid(True)
        plt.savefig(os.path.join(PLOT_PATH, 'CF_numerical_k.png'), format='png')
        plt.show()

    @staticmethod
    def plotting_w(df):
        df = df.sort_values(by='w')
        plt.rcParams.update({'font.size': 22})
        plt.figure(figsize=(15, 10))
        plt.plot(df['w'], df['int_fourier'], "-", color="black", linewidth=3)
        plt.axis([-60, 60, -0.05, 1.05])
        plt.xlabel(r"$w$")
        plt.ylabel(r"$I(w)$")
        plt.grid(True)
        plt.savefig(os.path.join(PLOT_PATH, 'CF_numerical_w.png'), format='png')
        plt.show()

    @staticmethod
    def plotting_p(df):
        plt.rcParams.update({'font.size': 22})
        plt.figure(figsize=(15, 10))
        plt.plot(df['u'], df['inv_fourier'], "-", color="black", linewidth=3)
        plt.xlabel(r"$u$")
        plt.ylabel(r"$P(u)$")
        plt.grid(True)
        plt.savefig(os.path.join(PLOT_PATH, 'CF_numerical_p.png'), format='png')
        plt.show()


tic = time.time()
x = CFnum(500, 1, 0.0001, 0.00005, (30, 30))
toc = time.time()
print('Time: ', toc - tic)
