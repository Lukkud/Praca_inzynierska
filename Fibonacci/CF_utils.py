import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import os


DIR_PATH = Path(os.path.abspath(__file__)).parents[0]
PLOT_PATH = os.path.join(DIR_PATH, 'plot_files')
DATA_PATH = os.path.join(DIR_PATH, 'data_files')
Path(PLOT_PATH).mkdir(parents=True, exist_ok=True)
Path(DATA_PATH).mkdir(parents=True, exist_ok=True)


class CFutils:
    @staticmethod
    def saving_data(df, file_name):
        df = df.round(6)
        df.to_excel(os.path.join(DATA_PATH, file_name), index=False)

    @staticmethod
    def plotting_k(df):
        plt.rcParams.update({'font.size': 22})
        plt.figure(figsize=(15, 10))
        plt.bar(df['k'].loc[df['int_fourier'] > 0.001], df['int_fourier'].loc[df['int_fourier'] > 0.001],
                color="black",
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