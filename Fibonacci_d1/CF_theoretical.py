import numpy as np
import matplotlib.pyplot as mpl
import pandas as pd
from pathlib import Path
import os
import time


np.seterr(divide='ignore', invalid='ignore')
DIR_PATH = Path(os.path.abspath(__file__)).parents[0]
FILE_PATH = os.path.join(DIR_PATH, 'files')


class CFteo:
    def __init__(self, n, m, csv_data=1, plotting=(1, 1, 1), inv_fou=1):
        self.tau = (1 + np.sqrt(5)) * 0.5
        self.k0 = 2 * np.pi * self.tau ** 2 / (1 + self.tau ** 2)
        self.q0 = self.k0 / self.tau
        self.k1 = 5 ** 0.5 * self.k0
        self.n_range = n
        self.m_range = m

        self.plotting = plotting
        self.inv_fou = inv_fou
        self.csv_data = csv_data

    def scheduler(self):
        df = self.calculating()
        CFteo.saving_data(df) if self.csv_data else 0
        self.plotting_group_k(df) if self.plotting[0] else 0
        CFteo.plotting_k(df) if self.plotting[1] else 0
        CFteo.plotting_w(df) if self.plotting[2] else 0
        self.inv_fourier(df) if self.inv_fou else 0

    def calculating(self):
        num_lst = [[n, m] for n in range(self.n_range) for m in range(self.m_range)]
        num_lst = np.array(num_lst).T
        df = pd.DataFrame({'n': num_lst[0], 'm': num_lst[1]})
        df['k'] = self.k0 * df['n'] + self.q0 * df['m']
        df['w'] = self.k0 * df['n'] + (self.q0 - self.k1) * df['m']
        df['chi'] = self.k0 * (df['n'] - self.tau * df['m']) / (2 * self.tau)
        df['F(w)'] = np.where(df['chi'] != 0, np.sin(df['chi']) / df['chi'], 1)
        df['I(w)'] = np.where(df['chi'] != 0, abs(np.sin(df['chi']) / df['chi']) ** 2, 1)
        return df

    @staticmethod
    def saving_data(df):
        df = df.round(6)
        df.to_excel(os.path.join(FILE_PATH, 'cf_theoretical_values.xlsx'))

    def plotting_group_k(self, df):
        k_i = np.reshape(df['k'].tolist(), (self.m_range, self.n_range)).T
        iw_fou_i = np.reshape(df['I(w)'].tolist(), (self.m_range, self.n_range)).T
        mpl.rcParams.update({'font.size': 25})
        mpl.figure(figsize=(20, 10))

        for m in range(5):
            mpl.bar(k_i[m], iw_fou_i[m], label="m={}".format(m), width=0.2)
            k_x = np.linspace(k_i[m][0], k_i[m][-1], 1000)
            mpl.plot(k_x,
                     abs(np.sin((k_x - m * self.k1) / (2 * self.tau)) / ((k_x - m * self.k1) / (2 * self.tau))) ** 2,
                     zorder=m+1)
        mpl.axis([0, 50, -0.05, 1.05])
        mpl.xlabel(r"$k$")
        mpl.ylabel(r"$I(k)$")
        mpl.grid(True)
        mpl.legend(bbox_to_anchor=[0.9, 0.6])
        mpl.savefig(os.path.join(FILE_PATH, 'CF_theoretical_k_group.png'), format='png')
        mpl.show()

    @staticmethod
    def plotting_k(df):
        mpl.rcParams.update({'font.size': 22})
        mpl.figure(figsize=(15, 10))
        mpl.bar(df['k'].tolist(), df['I(w)'].tolist(), width=0.3, color='black')
        mpl.axis([-2, 50, -0.05, 1.05])
        mpl.xlabel(r"$k$")
        mpl.ylabel(r"$I(k)$")
        mpl.grid(True)
        mpl.savefig(os.path.join(FILE_PATH, 'CF_theoretical_k.png'), format='png')
        mpl.show()

    @staticmethod
    def plotting_w(df):
        df = df.sort_values('w')
        mpl.figure(figsize=(15, 10))
        mpl.plot(df['w'].tolist(), df['I(w)'].tolist(), "-", color="black")
        mpl.axis([-60, 60, -0.05, 1.05])
        mpl.xlabel(r"$w$")
        mpl.ylabel(r"$I(w)$")
        mpl.grid(True)
        mpl.savefig(os.path.join(FILE_PATH, 'CF_theoretical_w.png'), format='png')
        mpl.show()

    def inv_fourier(self, df):
        u = np.arange(-1, 1, 0.001)
        inv_df = pd.DataFrame({'u': u, 'F(u)': np.nan})
        df = df.sort_values('w')

        for ui in range(len(inv_df)):
            df['tmp_feuw'] = df['F(w)'] * (np.cos(inv_df['u'].iloc[ui] * df['w']) -
                                           1j * np.sin(inv_df['u'].iloc[ui] * df['w']))
            inv_df.loc[ui, 'F(u)'] = df['tmp_feuw'].sum() / (self.n_range - 1)

        mpl.figure(figsize=(15, 10))
        mpl.plot(inv_df['u'].tolist(), inv_df['F(u)'].tolist(), "-", color="black")
        mpl.xlabel(r"$u$")
        mpl.ylabel(r"$P(u)$")
        mpl.grid(True)
        mpl.savefig(os.path.join(FILE_PATH, 'CF_theoretical_p_fourier.png'), format='png')
        mpl.show()


tic = time.time()
x = CFteo(50, 50, 1, (1, 1, 1), 1)
x.scheduler()
toc = time.time()
print(toc - tic)