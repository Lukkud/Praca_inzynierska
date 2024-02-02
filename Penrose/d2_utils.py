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
    def saving_data_excel(df, file_name):
        df = df.round(6)
        df.to_excel(os.path.join(DATA_PATH, file_name), index=False)

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
