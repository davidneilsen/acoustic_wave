import numpy as np
import csv
import argparse
from bandedCompactDerivative import CompactDerivative

def calfunc(x):
    L = x[-1] - x[0]
    k1 = 17.0/L
    k2 = 29.0/L
    amp = 1.0
    f = amp*(np.sin(k1*x) + np.cos(k2*x))
    df = amp*(k1*np.cos(k1*x) - k2*np.sin(k2*x))
    return f, df

def write_csv(filename, x, u, v, w, y):
    """
    Write the arrays x, u, v, w, y to a CSV file.

    Parameters:
        filename (str): Output CSV file path.
        x, u, v, w (np.ndarray): 1D arrays of shape (N,)
    """
    assert len(x) == len(u) == len(v) == len(w), "Arrays must have the same length"

    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['x', 'f', 'df', 'dfex', 'dferr'])  # header row
        for xi, ui, vi, wi, yi in zip(x, u, v, w, y):
            writer.writerow([xi, ui, vi, wi, yi])

def main():
    parser = argparse.ArgumentParser(description="Test compact derivative implementations.")
    parser.add_argument("-a", "--xmin", type=float, default=-1.0, help="xmin")
    parser.add_argument("-b", "--xmax", type=float, default=1.0, help="xmax")
    parser.add_argument("-l", "--level", type=int, default=0, help="level")
    parser.add_argument("-d", "--deriv", type=str, default="E4", help="Derivative type")
    parser.add_argument("npts", type=int, help="N points")

    args = parser.parse_args()
    N = args.npts
    if N % 2 == 1:
        N -= 1

    nx = (2**args.level) * N + 1
    x_L = args.xmin
    x_R =  args.xmax
    x = np.linspace(x_L, x_R, nx)
    f, dfex = calfunc(x)

    deriv = CompactDerivative(x, order=args.deriv, lusolve=True) # E4, JTT4, JTT6, JTP6, KP4
    df = deriv.evaluate(f)

    dferr = df - dfex
    write_csv('tderiv.csv', x, f, df, dfex, dferr)

    #print(deriv.ab)
    #print(deriv.B)


if __name__ == "__main__":
    main()

