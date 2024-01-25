import numpy as np
import pandas as pd
import time
from CF_utils import CFutils
from numba import jit


class CFnum(CFutils):
    def __init__(self):
        try:
            self.config = self.read_json()['cf_numerical']
            self.at = self.config['at']
            self.func = self.config['func']
            self.step = self.config['step']
            self.eps = self.config['eps']
            self.n_range = self.config['tup_range'][0]
            self.m_range = self.config['tup_range'][1]
        except KeyError as exc:
            raise Exception(f"No parameter {exc} specified for CF_numerical script in config.json file")

        self.tau = (1 + np.sqrt(5)) * 0.5
        self.k0 = 2 * np.pi * self.tau ** 2 / (1 + self.tau ** 2)
        self.q0 = self.k0 / self.tau
        self.k1 = 5 ** 0.5 * self.k0
        self.tictoc = []
        self.fibonacci_sequence = None
        self.df_k = None
        self.df_w = None
        self.df_p = None

    def execute(self):
        self.fibonacci_sequence = self.choose_generator_function(self.func, self.at)

        print('Preparing k part')
        self.tictoc.append(time.time())
        self.df_k = pd.DataFrame({'k': np.arange(0, 100, self.step), 'fourier': np.nan, 'int_fourier': np.nan})
        # Much slower solution
        # self.df_k['fourier'] = self.df_k.apply(lambda row: np.sum([complex(np.cos(row['k'] * x), np.sin(row['k'] * x)) for x in self.fibonacci_str['string']]), axis=1)
        self.df_k['fourier'] = self.fourier(self.df_k['k'].values, self.fibonacci_sequence['string'].values)
        self.df_k['int_fourier'] = np.absolute(self.df_k['fourier']) ** 2
        self.saving_data(self.df_k, 'cf_numerical_k_data.xlsx')
        self.plotting_k(self.df_k['k'].loc[self.df_k['int_fourier'] > 0.001],
                        self.df_k['int_fourier'].loc[self.df_k['int_fourier'] > 0.001],
                        'CF_numerical_k.png')

        print('Comparing models')
        self.compare()

        print('Preparing w part')
        self.tictoc.append(time.time())
        self.df_w = self.prepare_w(self.df_k)
        self.saving_data(self.df_w, 'cf_numerical_w_data.xlsx')
        self.plotting_w(self.df_w, 'w', 'int_fourier', 'CF_numerical_w.png')

        print('Preparing p part')
        self.tictoc.append(time.time())
        self.df_p = self.inv_fourier()
        self.saving_data(self.df_p, 'cf_numerical_p_data.xlsx')
        self.plotting_p(self.df_p, 'u', 'inv_fourier', 'CF_numerical_p.png')
        self.tictoc.append(time.time())

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
                # print(pd.Series(list(lst_df_teo) + [rows.k, rows.k - lst_df_teo[1] * self.k1, rows.fourier, rows.int_fourier], index=df_w.columns).to_frame().T)
                df_w = pd.concat([df_w, pd.Series(list(lst_df_teo) + [rows.k, rows.k - lst_df_teo[1] * self.k1, rows.fourier, rows.int_fourier], index=df_w.columns).to_frame().T], ignore_index=True)
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

    def compare(self):
        df = self.df_k
        k_tmp = np.array([n * self.k0 + m * self.q0 for n in range(self.n_range) for m in range(self.m_range)])
        iw_fou_num, iw_fou_teo = [], []
        for i in range(1, len(k_tmp)):
            if k_tmp[i] < 100:
                df['tmp_k'] = abs(df['k'] - k_tmp[i])
                n, m = i // self.n_range, i % self.n_range
                iw_fou_num.append(df['int_fourier'].loc[df['tmp_k'].idxmin()])
                X = self.k0 * (n - m * self.tau) / (2 * self.tau)
                iw_fou_teo.append(abs(np.sin(X) / X) ** 2)

        iw_fou_teo, iw_fou_num = np.array(iw_fou_teo[1:]), np.array(iw_fou_num[1:])
        print("R:", 100 * abs(iw_fou_teo - iw_fou_num).sum() / sum(iw_fou_teo))
        self.compare_plots(iw_fou_teo, iw_fou_num)


if __name__ == "__main__":
    x = CFnum()
    x.execute()
    print('Done')
    print('Time first (k) part: ', round(x.tictoc[1] - x.tictoc[0], 2))
    print('Time second (w) part: ', round(x.tictoc[2] - x.tictoc[1], 2))
    print('Time third (p) part: ', round(x.tictoc[3] - x.tictoc[2], 2))
    print('Overall time: ', round(x.tictoc[3] - x.tictoc[0], 2))
