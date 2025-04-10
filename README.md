# acoustic_wave

This is the acoustic wave example for compact finite differivatives
from the paper by Wu and Kim (2024).

## Testing Derivative Operators

This code takes the derivative of a known function and compares the results to the exact solution.
The function is

    f(x) = A * (sin(k1*x/L) + cos(k2*x/L))

for k1 = 17 and k2 = 29, and L is the size of the domain.

The derivative is evaluated on a grid with N points.  To check convergence, the `level` option
changes to the number of points to be 

    Nx = 2^level * N0 + 1.

Options for the code are found by executing
    `python3 testderiv.py --help`

Results are written to a CSV file, which has columns for: x, f, df, df_ex, dferr.
        x 
        f = function value f(x)
       df = finite derivative approximation to df/dx
    df_ex = analytic derivative function, df/dx
    dferr = df - df_ex

Example usage:

    `python3 testderiv.py -a 0.0 -b 1.0 -l 2 -d KP4 100`


## Acoustic Wave Code

The linear acoustic wave equation.  

Parameter files are located in the pars directory.

Example usage:

    `python3 acoustic_wave.py id.pars`
