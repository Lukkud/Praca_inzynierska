import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import os

DIR_PATH = Path(os.path.abspath(__file__)).parents[0]
PLOT_PATH = os.path.join(DIR_PATH, 'plot_files')
SOURCE_PATH = os.path.join(DIR_PATH, 'Source')
Path(PLOT_PATH).mkdir(parents=True, exist_ok=True)

df = pd.read_csv(os.path.join(SOURCE_PATH, 'Penrose2300pkt.txt'), sep='\t', header=None, names=["x", "y"])

plt.figure(figsize=(15, 10))
plt.rcParams.update({'font.size': 16})
plt.plot(df.x, df.y, ".", ms=5, color="black")
plt.xlabel(r"$x_{||}$")
plt.ylabel(r"$y_{||}$")
plt.grid(True)
plt.savefig(os.path.join(PLOT_PATH, 'Penrose_tilling.png'), format='png')
plt.show()
