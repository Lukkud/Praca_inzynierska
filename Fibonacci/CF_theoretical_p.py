import numpy as np
import pandas as pd
import time
from CF_utils import CFutils


class CFteostat(CFutils):
    def __init__(self):
        try:
            self.config = self.read_json()['cf_theoretical_p']
            self.at = self.config['at']
            self.func = self.config['func']
            self.bins = self.config['bins']
        except KeyError as exc:
            raise Exception(f"No parameter {exc} specified for CF_theoretical_p script in config.json file")

        self.tau = (1 + np.sqrt(5)) * 0.5
        self.k0 = 2 * np.pi * self.tau ** 2 / (1 + self.tau ** 2)
        self.net_const = 2 * np.pi / self.k0
        self.fibonacci_sequence = None
        self.df = None

    def execute(self):
        self.fibonacci_sequence = self.choose_generator_function(self.func, self.at)
        self.df = pd.DataFrame({'number': np.arange(0, self.fibonacci_sequence['string'].iloc[-1] + 10, self.net_const)})

        stat_num = []
        for i in self.fibonacci_sequence['string']:
            # print('Complete: {}%'.format((i/fibonacci_str['string'].iloc[-1]*100).round(2)))
            self.df['stat'] = self.df['number'] - i
            self.df['abs_stat'] = abs(self.df['number'] - i)
            stat_num.append(self.df['stat'].loc[self.df[['abs_stat']].idxmin()].values)

        hist, bins = np.histogram(stat_num, bins=np.linspace(-1, 1, self.bins), density=True)
        self.plotting_p(pd.DataFrame({'hist': hist, 'bins': bins[:-1]}), 'bins', 'hist', 'CF_theoretical_p_stat.png')


if __name__ == "__main__":
    tic = time.time()
    x = CFteostat()
    x.execute()
    toc = time.time()
    print('Overall time: ', round(toc - tic, 2))
