import numpy as np
import pandas as pd
from pathlib import Path
import os


DIR_PATH = Path(os.path.abspath(__file__)).parents[0]
PLOT_PATH = os.path.join(DIR_PATH, 'plot_files')
DATA_PATH = os.path.join(DIR_PATH, 'data_files')


def cf_transform(atoms):
    """Method is used to generate fibonacci string. To get fibonacci string I use long (L) and short (S) sections. In
    ever step program replace sections with given formula: L -> LS, S -> L. The overall length for every component of
    list is our fibonacci string. In program I use sections lengths: L = tau = 1.618... and S = 1. More about this
    idea you can read in my engineer's thesis in chapter 'Rekurencyjna metoda generacji ciągu Fibonacciego'."""

    tau = (1 + np.sqrt(5)) * 0.5
    num_fibo, count_l, count_s, fibo = [], [], [], "S"
    while len(fibo) <= atoms:
        fibo = ''.join([j.replace('S', 'A') for j in fibo])
        fibo = ''.join([j.replace('L', 'LS') for j in fibo])
        fibo = ''.join([j.replace('A', 'L') for j in fibo])
    for i in range(atoms):
        count_l.append(fibo[:i].count('L'))
        count_s.append(fibo[:i].count('S'))
        num_fibo.append(round(count_l[-1] * tau + count_s[-1], 6))
    df = pd.DataFrame({'string': num_fibo})
    df.to_csv(os.path.join(DATA_PATH, 'fibo_transform.csv'))


def cf_projection(length):
    """Method is used to generate fibonacci string. To get fibonacci string I use 2 dimensional, regular, square
    structure described in cartesian coordinates. The idea is to choose part of this structure and project every atom
    from it on a new base. The new x axis is given by axis of our section. For very specific section tilt angle (alpha)
    and distance between adjoining atoms (A) it is possible to get fibonacci chain like in first method. Both methods
    are correct More about this idea you can read in my engineer's thesis in chapter 'Metoda generacji ciągu
    Fibonacciego za pomocą rzutowania'."""

    tau = (1 + np.sqrt(5)) * 0.5
    A = (1 + tau ** 2) ** 0.5
    alpha = np.arctan(1 / tau)
    num_fibo = []

    border_down = [i * A * np.tan(alpha) for i in range(length)]
    border_up = [i * A * np.tan(alpha) + tau ** 2 / np.cos(alpha) for i in range(length)]
    surface = [i * A for i in range(length)]
    for j in range(length):
        for i in range(length):
            if (surface[i] >= border_down[j]) and (surface[i] <= border_up[j]):
                num_fibo.append(round((surface[i] - border_down[j]) * np.sin(alpha) + j * A / np.cos(alpha) - 1, 6))
    df = pd.DataFrame({'string': num_fibo[1:]})
    df.to_csv(os.path.join(DATA_PATH, 'fibo_projection.csv'))
