import matplotlib.pyplot as plt
from pathlib import Path
import os
from CF_generator import cf_transform, cf_projection


DIR_PATH = Path(os.path.abspath(__file__)).parents[0]
PLOT_PATH = os.path.join(DIR_PATH, 'plot_files')
DATA_PATH = os.path.join(DIR_PATH, 'data_files')
Path(PLOT_PATH).mkdir(parents=True, exist_ok=True)
Path(DATA_PATH).mkdir(parents=True, exist_ok=True)


class CFutils:
    @staticmethod
    def choose_generator_function(func, at):
        if func == 1:
            df = cf_transform(at)
            df.to_csv(os.path.join(DATA_PATH, 'fibo_transform.csv'))
            return df
        elif func == 2:
            df = cf_projection(at)
            df.to_csv(os.path.join(DATA_PATH, 'fibo_projection.csv'))
            return df
        else:
            raise Exception('Choose 1 to use cf_transform or 2 to use cf_projection')

    @staticmethod
    def saving_data(df, file_name):
        df = df.round(6)
        df.to_excel(os.path.join(DATA_PATH, file_name), index=False)

    @staticmethod
    def plotting_k(x, y, file_name):
        plt.rcParams.update({'font.size': 22})
        plt.figure(figsize=(15, 10))
        plt.bar(x, y, color="black", width=0.3)
        plt.axis([-2, 50, -0.05, 1.05])
        plt.xlabel(r"$k$")
        plt.ylabel(r"$I(k)$")
        plt.grid(True)
        plt.savefig(os.path.join(PLOT_PATH, file_name), format='png')
        plt.show()

    @staticmethod
    def plotting_w(df, x_col_name, y_col_name, file_name):
        df = df.sort_values(by='w')
        plt.rcParams.update({'font.size': 22})
        plt.figure(figsize=(15, 10))
        plt.plot(df[x_col_name], df[y_col_name], "-", color="black", linewidth=3)
        plt.axis([-60, 60, -0.05, 1.05])
        plt.xlabel(r"$w$")
        plt.ylabel(r"$I(w)$")
        plt.grid(True)
        plt.savefig(os.path.join(PLOT_PATH, file_name), format='png')
        plt.show()

    @staticmethod
    def plotting_p(df, x_col_name, y_col_name, file_name):
        plt.rcParams.update({'font.size': 22})
        plt.figure(figsize=(15, 10))
        plt.plot(df[x_col_name], df[y_col_name], "-", color="black", linewidth=3)
        plt.xlabel(r"$u$")
        plt.ylabel(r"$P(u)$")
        plt.grid(True)
        plt.savefig(os.path.join(PLOT_PATH, file_name), format='png')
        plt.show()

    @staticmethod
    def compare_plots(iw_fou_teo, iw_fou_num):
        plt.rcParams.update({'font.size': 22})
        plt.figure(figsize=(10, 10))
        plt.plot(iw_fou_teo, iw_fou_teo, '-', color='red', zorder=1)
        plt.plot(iw_fou_teo, iw_fou_num, 'o', ms=3, color='black', zorder=2)
        plt.xlabel(r"$I(k)_{teo}$")
        plt.ylabel(r"$I(k)_{num}$")
        plt.grid(True)
        plt.savefig(os.path.join(PLOT_PATH, 'CF_numerical_compare.png'), format='png')
        plt.show()

        plt.rcParams.update({'font.size': 22})
        plt.figure(figsize=(10, 10))
        plt.plot(iw_fou_teo, iw_fou_teo, '-', color='red', zorder=1)
        plt.plot(iw_fou_teo, iw_fou_num, 'o', ms=3, color='black', zorder=2)
        plt.xlabel(r"$I(k)_{teo}$")
        plt.ylabel(r"$I(k)_{num}$")
        plt.axis([10e-7, 1, 10e-7, 1])
        plt.xscale('log')
        plt.yscale('log')
        plt.grid(True)
        plt.savefig(os.path.join(PLOT_PATH, 'CF_numerical_compare_log.png'), format='png')
        plt.show()