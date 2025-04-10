# acoustic_wave

This is the acoustic wave example for compact finite differivatives
from the paper by Wu and Kim (2024).

## Compact Finite Derivatives

These are defined in the `bandedCompactDerivatives.py` file.  The linear system is solved using LAPACK.

## Testing Derivative Operators

This code takes the derivative of a known function and compares the results to the exact solution.
The function is

    f(x) = A * (sin(k1*x/L) + cos(k2*x/L))

for k1 = 17 and k2 = 29, and L is the size of the domain.

The derivative is evaluated on a grid with N points.  To check convergence, the `level` option
changes to the number of points to be 

    Nx = 2^level * N0 + 1.

Options for the code are found by executing
    python3 testderiv.py --help

Results are written to a CSV file, which has columns for: x, f, df, df_ex, dferr.
        x 
        f = function value f(x)
       df = finite derivative approximation to df/dx
    df_ex = analytic derivative function, df/dx
    dferr = df - df_ex

Example usage:

    python3 testderiv.py -a 0.0 -b 1.0 -l 2 -d KP4 100


## Acoustic Wave Code

The linear acoustic wave equation.  

    du/dt + dp/dx = 0
    dp/dt + du/dx = 0

with the no-penetration boundary conditions

    u(xL,t) = u(xR,t) = 0.

The boundary condition for P is either left free, or the first derivative is set to zero after each RK step.

The equations are solved using the Method of Lines with RK4.  The spatial derivatives are computed using either explicit or compact finite differences.  The available derivatives are:

    * E4 = explicit fourth-order
    * E6 = explicit sixth-order
    * JTT4 = compact fourth-order, tridiagonal system
    * JTT6 = compact sixth-order, tridiagonal system
    * JTP6 = compact sixth-order, pentadiagonal system
    * KP4 = compact fourth-order, pentadiagonal system
    * SP4 = compact fourth-order, pentadiagonal system

Parameter files are located in the pars directory.

Example usage:

    python3 acoustic_wave.py id.pars
