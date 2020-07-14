import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os


DIR_PATH = Path(os.path.abspath(__file__)).parents[0]
PLOT_PATH = os.path.join(DIR_PATH, 'plot_files')
DATA_PATH = os.path.join(DIR_PATH, 'data_files')

tau = (1 + np.sqrt(5)) * 0.5
num, fibo = [], "L"
for i in range(25):
    fibo = ''.join([j.replace('S', 'A') for j in fibo])
    fibo = ''.join([j.replace('L', 'LS') for j in fibo])
    fibo = ''.join([j.replace('A', 'L') for j in fibo])
    num.append(fibo.count('L')/fibo.count('S'))

plt.rcParams.update({'font.size': 20})
plt.figure(figsize=(12, 8))
plt.axis([1, 20, 1.45, 2.05])
plt.plot(range(0, 25), [1.618 for i in range(0, 25)], "--", linewidth=3,   color='red', label=r"$\tau$")
plt.plot(range(1, 26), num, ".", color="blue", ms=12,  label=r"$\frac{\upsilon_{L}}{\upsilon_{S}}(n)$")
plt.xlabel("Iteration $n$")
plt.ylabel(r"$\frac{\upsilon_{L}}{\upsilon_{S}}$")
plt.xticks(np.arange(2, 22, 2))
plt.grid(True)
plt.legend()
plt.savefig(os.path.join(PLOT_PATH, 'CF_convergence.png'), format='png')
plt.show()
