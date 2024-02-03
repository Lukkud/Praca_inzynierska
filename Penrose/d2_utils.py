import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import os
import json


DIR_PATH = Path(os.path.abspath(__file__)).parents[0]
PLOT_PATH = os.path.join(DIR_PATH, 'plot_files')
DATA_PATH = os.path.join(DIR_PATH, 'data_files')
SOURCE_PATH = os.path.join(DIR_PATH, 'Source')
Path(PLOT_PATH).mkdir(parents=True, exist_ok=True)
Path(DATA_PATH).mkdir(parents=True, exist_ok=True)
Path(SOURCE_PATH).mkdir(parents=True, exist_ok=True)


class D2utils:
    def read_json(self):
        f = open(os.path.join(DIR_PATH, "config.json"))
        config = json.load(f)
        f.close()
        return config

    @staticmethod
    def saving_data_csv(df, file_name):
        df = df.round(6)
        df.to_csv(os.path.join(DATA_PATH, file_name), index=False)

    @staticmethod
    def read_source_data(file_name):
        df = pd.read_csv(os.path.join(SOURCE_PATH, file_name),
                         sep='\t', header=None, names=["x", "y"])
        return df

    @staticmethod
    def read_diffraction_data(file_name):
        df = pd.read_csv(os.path.join(DATA_PATH, file_name))
        return df

    @staticmethod
    def plotting_k_penrose(df, file_name, threshold):
        df = df.loc[df["iw_fou"] > threshold]
        plt.rcParams.update({'font.size': 16})
        fig = plt.figure(figsize=(15, 10))
        ax = plt.axes(projection="3d")
        ax.scatter3D(df.k1, df.k2, df.iw_fou, color="black")
        ax.set_xlabel(r"$k_{x}$")
        ax.set_ylabel(r"$k_{y}$")
        ax.set_zlabel(r"$I(k_{x}, k_{y})$")
        plt.grid(True)
        ax.view_init(0, 0)
        plt.savefig(os.path.join(PLOT_PATH, f"{file_name}_0deg.png"), format='png')
        ax.view_init(90, 0)
        plt.savefig(os.path.join(PLOT_PATH, f"{file_name}_90deg.png"), format='png')
        ax.view_init(45, 0)
        plt.savefig(os.path.join(PLOT_PATH, f"{file_name}_45deg.png"), format='png')
        ax.view_init(15, 0)
        plt.savefig(os.path.join(PLOT_PATH, f"{file_name}_15deg.png"), format='png')
        plt.show()

    @staticmethod
    def plotting_w_penrose(df, file_name, threshold):
        df = df.loc[df["iw_fou"] > threshold]
        plt.rcParams.update({'font.size': 16})
        fig = plt.figure(figsize=(15, 10))
        ax = plt.axes(projection="3d")
        ax.scatter3D(df.k1, df.k2, df.iw_fou, color="black")
        ax.set_xlabel(r"$k_{x}$")
        ax.set_ylabel(r"$k_{y}$")
        ax.set_zlabel(r"$I(k_{x}, k_{y})$")
        plt.grid(True)
        ax.view_init(0, 0)
        plt.savefig(os.path.join(PLOT_PATH, f"{file_name}_0deg.png"), format='png')
        ax.view_init(90, 0)
        plt.savefig(os.path.join(PLOT_PATH, f"{file_name}_90deg.png"), format='png')
        ax.view_init(45, 0)
        plt.savefig(os.path.join(PLOT_PATH, f"{file_name}_45deg.png"), format='png')
        ax.view_init(15, 0)
        plt.savefig(os.path.join(PLOT_PATH, f"{file_name}_15deg.png"), format='png')
        plt.show()

    @staticmethod
    def plotting_w_phase(df, file_name, threshold, border_1, border_2):
        df = df.loc[df["iw_fou"] > threshold].sort_values('phase').reset_index(drop=True)

        df_group_1 = df.loc[df["group"] == 1]
        df_group_2 = df.loc[df["group"] == 2]
        df_group_3 = df.loc[df["group"] == 3]

        plt.rcParams.update({'font.size': 22})
        fig = plt.figure(figsize=(15, 10))
        plt.plot(df_group_1.index.values, df_group_1.phase, 'o', color=(0.1, 0.9, 0.1), ms=1, zorder=1)
        plt.plot(df_group_2.index.values, df_group_2.phase, 'o', color="red", ms=1, zorder=2)
        plt.plot(df_group_3.index.values, df_group_3.phase, 'o', color="blue", ms=1, zorder=3)
        plt.axhline(y=border_1, ls='--', linewidth=2, color='blue')
        plt.axhline(y=border_2, ls='--', linewidth=2, color='red')
        plt.axhline(y=-border_1, ls='--', linewidth=2, color='blue')
        plt.axhline(y=-border_2, ls='--', linewidth=2, color='red')
        plt.xlabel(r"Numer piku")
        plt.ylabel(r"Wartość fazy $\phi$")
        plt.grid(True)
        plt.savefig(os.path.join(PLOT_PATH, file_name), format='png')
        plt.show()

    @staticmethod
    def plotting_w(df, file_name, threshold):
        df = df.loc[df["iw_fou"] > threshold]
        plt.rcParams.update({'font.size': 16})
        fig = plt.figure(figsize=(10, 10))
        ax = plt.axes(projection='3d')
        ax.scatter3D(df.w1, df.w2, df.iw_fou, s=4, color="black")
        ax.set_xlabel(r"$w_{x}$")
        ax.set_ylabel(r"$w_{y}$")
        ax.set_zlabel(r"$I(w_{x}, w_{y})$")
        plt.grid(True)
        ax.view_init(0, 0)
        plt.savefig(os.path.join(PLOT_PATH, f"{file_name}_0deg.png"), format='png')
        ax.view_init(90, 0)
        plt.savefig(os.path.join(PLOT_PATH, f"{file_name}_90deg.png"), format='png')
        ax.view_init(45, 0)
        plt.savefig(os.path.join(PLOT_PATH, f"{file_name}_45deg.png"), format='png')
        ax.view_init(15, 0)
        plt.savefig(os.path.join(PLOT_PATH, f"{file_name}_15deg.png"), format='png')
        plt.show()
