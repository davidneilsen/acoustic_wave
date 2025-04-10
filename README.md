# acoustic_wave

This is the acoustic wave example for compact finite differivatives
from the paper by Wu and Kim (2024).

## Compact Finite Derivatives

Several compact finite difference operators are defined in the `bandedCompactDerivatives.py` file.  These derivatives are written as a linear system
```math
{\bf P}\, f' = {\bf Q}\, f
```
where $\bf P$ is either a tridiagonal or pentadiagonal matrix, $\bf Q$ is a banded matrix, and $f'$ is the derivative of the function $f(x)$.  The linear system is solved as
```math
{\bf P} f' = b,
```
for $b = {\bf Q}f$.  The linear system is solved using LAPACK routines in `scipy.linalg`.  The solution method for the linear system is chosen when the class is instantiated.  When the parameter `lusolve=True`, the linear system is solved directly with LU decomposition.  When `lusolve=False`, the system is solved using `scipy.linalg.solve_banded`.

The options for the derivatives are

    1. E4 Explicit finite difference, O(h^4)
    2. E6 Explicit finite difference, O(h^6) 
    3. JTT4 Compact finite difference, O(h^4), Tridiagonal system, Jonathan Tyler 
    4. JTT6 Compact finite difference, O(h^6), Tridiagonal system, Jonathan Tyler 
    5. JTP6 Compact finite difference, O(h^6), Pentadiagonal system, Jonathan Tyler 
    6. KP4 Compact finite difference, O(h^4), optimized Pentadiagonal system, Wu and Kim (Table III)
    7. SP4 Compact finite difference, O(h^4), "standard" Pentadiagonal system, Wu and Kim  (Table I)

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

The equations are solved using the Method of Lines with RK4.  The spatial derivatives are computed using either explicit or compact finite differences.

Parameter files are located in the `pars` directory.  Options are:

    - Nt = number of iterations
    - N = Base number of points
    - output_interval = frequency for writing output files
    - x_L = left boundary
    - x_R = right boundary
    - level = level for convergence testing.
    - cfl = Courant factor
    - deriv_type = spatial derivative (E4, E6, JTT4, JTT6, JTP6, KP4, SP4)
    - bctype = boundary condition for p.  
        - 0 = no condition on p
        - 2 = dp/dx = 0, 2nd order
        - 4 = dp/dx = 0, 4th order
    - idtype = initial data type
        -  0 = simple Gaussian
        -  1 = simple Gaussian * cos(x) [Used by Wu and Kim]
    - pulse_center = Gaussian center
    - pulse_width = Gaussian width
    - output_dir = directory for output

To simplify convergence testing, use the `level` parameter.  The number of points is 
```math
    Nx = 2^\ell N + 1
```
where $\ell$ is the level and $N$ is the Base number of points (an even integer).

Example usage:

    python3 acoustic_wave.py id.pars

Output files are written to a subdirectory.  These are simple 1D ascii files in the `curve` format that is read by ViSIT.
