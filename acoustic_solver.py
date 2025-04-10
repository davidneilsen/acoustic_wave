import numpy as np
import configparser
import os
import argparse
from math import *
from bandedCompactDerivative import CompactDerivative

def read_parameters(filename="params.ini"):
    config = configparser.ConfigParser()
    config.read(filename)
    p = config['parameters']
    return {
        "x_L": float(p['x_L']),
        "x_R": float(p['x_R']),
        "N": int(p['N']),
        "level": int(p['level']),
        "cfl": float(p['cfl']),
        "Nt": int(p['Nt']),
        "deriv_type": p['deriv_type'],
        "pulse_center": float(p['pulse_center']),
        "pulse_width": float(p['pulse_width']),
        "output_interval": int(p['output_interval']),
        "output_dir": p['output_dir'],
        "bctype": int(p['bctype']),
        "idtype": int(p['idtype'])
    }

def initdata_gauss(x, center, width):
    p = np.exp(-((x - center) / width) ** 2)
    u = np.zeros_like(x)
    return u, p

def initdata(x, center, width):
    d = log(2.0)
    p = 2*np.cos(0.5*np.pi*(x-center))*np.exp(-d*((x-center)/width)**2)
    u = np.zeros_like(x)
    return u, p

def rhs(u, p, dfdx):
    du_dt = -dfdx.evaluate(p)
    dp_dt = -dfdx.evaluate(u)
    return du_dt, dp_dt

def apply_bcs(u, p, bctype):
    # Apply no penetration boundary condition: u = 0 at boundaries
    u[0] = 0.0
    u[-1] = 0.0

    # boundary condition on p
    if bctype == 2:
        p[0] = (4.0*p[1] - p[2]) / 3.0
        p[-1] = (4.0*p[-2] - p[-3]) / 3.0
    elif bctype == 4:
        p[0] = (48.0*p[1] - 36.0*p[2] + 16.0*p[3] - 3.0*p[4])/25.0
        p[-1] = (48.0*p[-2] - 36.0*p[-3] + 16.0*p[-4] - 3.0*p[-5])/25.0;

def rk4(u, p, dt, dfdx, bctype):
    # Stage 1
    k1_u, k1_p = rhs(u, p, dfdx)
    u1 = u + 0.5*dt*k1_u
    p1  = p + 0.5*dt*k1_p
    apply_bcs(u1, p1, bctype)

    # Stage 2
    k2_u, k2_p = rhs(u1, p1, dfdx)
    u1 = u + 0.5*dt*k2_u
    p1 = p + 0.5*dt*k2_p
    apply_bcs(u1, p1, bctype)

    # Stage 3
    k3_u, k3_p = rhs(u1, p1, dfdx)
    u1 = u + dt*k3_u
    p1 = p + dt*k3_p
    apply_bcs(u1, p1, bctype)

    # Stage 4
    k4_u, k4_p = rhs(u1, p1, dfdx)
    u += dt / 6 * (k1_u + 2*k2_u + 2*k3_u + k4_u)
    p += dt / 6 * (k1_p + 2*k2_p + 2*k3_p + k4_p)
    apply_bcs(u, p, bctype)

def write_curve(filename, time, x, u, p):
    with open(filename, "w") as f:
        f.write(f"# TIME {time}\n")
        f.write(f"# U\n")
        for xi, di in zip(x, u):
            f.write(f"{xi:.8e} {di:.8e}\n")
        f.write(f"# P\n")
        for xi, di in zip(x, p):
            f.write(f"{xi:.8e} {di:.8e}\n")

def main():
    parser = argparse.ArgumentParser(prog='acoustic_solver', description='Solve the Acoustic Wave Equation')
    parser.add_argument('filename')
    args = parser.parse_args()
    fname = args.filename
    params = read_parameters(fname)

    x_L, x_R = params['x_L'], params['x_R']
    level = params['level']
    N = params['N']
    if N % 2 == 1:
        N -= 1
    Nx = (2**level) * N + 1
    output_interval = (2**level) * params['output_interval']
    Nt = (2**level) * params['Nt']

    dtype = params['deriv_type']
    bctype = params['bctype']

    cfl = params['cfl']
    pulse_center, pulse_width = params['pulse_center'], params['pulse_width']
    output_dir = params['output_dir']

    os.makedirs(output_dir, exist_ok=True)

    x = np.linspace(x_L, x_R, Nx)
    dx = x[1] - x[0]
    dt = cfl * dx  # Assuming wave speed = 1

    time = 0.0
    dfdx = CompactDerivative(x, order=dtype, lusolve=True)

    idtype = params['idtype']
    if idtype == 0:
        u, p = initdata_gauss(x, pulse_center, pulse_width)
    else:
        u, p = initdata(x, pulse_center, pulse_width)
  
    for n in range(Nt+1):
        if n % output_interval == 0:
            write_curve(f"{output_dir}/wave_{n:05d}.curve", time, x, u, p)
        rk4(u, p, dt, dfdx, bctype)
        time += dt

if __name__ == "__main__":
    main()

