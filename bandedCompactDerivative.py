import numpy as np
from scipy.linalg import solve_banded
import bandedLUSolve as blu

class CompactDerivative:
    def __init__(self, x, order='E4', lusolve=True):
        self.x = x
        self.N = len(x)
        self.dx = x[1] - x[0]
        self.order = order
        self.overwrite = False
        self.checkf = True
        self.use_lu_solve = lusolve 

        if order == 'JTT4':
            self.ab, self.B = self._init_JTT4()
            self.bandwidth = 1
            self.bands = (1,1)
        elif order == 'JTT6':
            self.ab, self.B = self._init_JTT6()
            self.bandwidth = 1
            self.bands = (1,1)
        elif order == 'JTP6':
            self.ab, self.B = self._init_JTP6()
            self.bandwidth = 2
            self.bands = (2,2)
        elif order == 'KP4':
            self.ab, self.B = self._init_KP4()
            self.bandwidth = 2
            self.bands = (2,2)
        elif order == 'SP4':
            self.ab, self.B = self._init_SP4()
            self.bandwidth = 2
            self.bands = (2,2)
        elif order == 'E4':
            self.bandwidth = 0
            self.B = self._init_E4()
            self.bands = (0,0)
        elif order == 'E6':
            self.bandwidth = 0
            self.B = self._init_E6()
            self.bands = (0,0)
        else:
            raise ValueError("Only 4th and 6th order schemes are supported. order = " + order)
        if self.bandwidth != 0 and lusolve:
            self.lu_factorization = blu.lu_banded(self.bands, self.ab, overwrite_ab=self.overwrite, check_finite=self.checkf)

    def _init_JTT4(self):
        '''
        #   This is the 4th-order tridiagonal operator from Jonathan Tyler
        '''
        N= self.N
        alpha = 1/4
        coeffs = np.array([-3.0/4.0, 0, 3.0/4.0])

        # Construct tridiagonal A matrix in banded form
        ab = np.zeros((3, N))
        ab[0, 1:] = alpha   # upper diagonal
        ab[1, :] = 1.0      # main diagonal
        ab[2, :-2] = alpha  # lower diagonal

        B = np.zeros((N, N))
        for i in range(1, N-1):
            B[i, i-1:i+2] = coeffs

        # One-sided stencils
        ab[0, 1] = 3.0
        ab[2, -2] = 3.0
        B[0, 0:4]  = np.array([-17.0, 9.0, 9.0, -1.0]) / 6.0
        B[-1, -4:] = np.array([1.0, -9.0, -9.0, 17.0]) / (6.0)

        return ab, B

    def _init_JTT6(self):
        '''
        #   This is the 6th-order tridiagonal operator from Jonathan Tyler
        '''
        N = self.N
        alpha = 1/3
        coeffs = np.array([-1/36, -14/18, 0, 14/18, 1/36])

        # Construct tridiagonal A matrix in banded form
        ab = np.zeros((3, N))
        ab[0, 1:] = alpha   # upper diagonal
        ab[1, :] = 1.0      # main diagonal
        ab[2, :-2] = alpha  # lower diagonal

        B = np.zeros((N, N))
        for i in range(2, N-2):
            B[i, i-2:i+3] = coeffs

        # One-sided stencils
        ab[0, 1] = 5.0
        ab[0, 2] = 0.75
        ab[0, -1] = 1/8

        ab[2, 0] = 1/8
        ab[2, -3] = 0.75
        ab[2, -2] = 5.0

        coeffb0 = np.array([ -197/60, -5/12, 5, -5/3, 5/12, -1/20 ]) 
        coeffb1 = np.array([ -43/96, -5/6, 9/8, 1/6, -1/96 ])
        B[0, 0:6]  = coeffb0
        B[1, 0:5]  = coeffb1
        B[-2, -5:] = -coeffb1[::-1]
        B[-1, -6:] = -coeffb0[::-1]

        return ab, B


    def _init_JTP6(self):
        '''
        #   This is the 6th-order pentadiagonal operator from Jonathan Tyler
        '''
        N = self.N

        alpha = 17.0/57.0
        beta = -1.0/114.0
        a = 30.0/19
        coeffs = np.array([-a/2, 0.0, a/2])

        # Construct tridiagonal A matrix in banded form
        ab = np.zeros((5, N))
        ab[0, 2:] = beta    # 2nd upper diagonal
        ab[1, 1:] = alpha   # 1st upper diagonal
        ab[2, :] = 1.0      # main diagonal
        ab[3, :-2] = alpha  # 1st lower diagonal
        ab[4, :-3] = beta   # 2nd lower diagonal

        B = np.zeros((N, N))
        for i in range(1, N-1):
            B[i, i-1:i+2] = coeffs

        # One-sided stencils
        ab[0, 2] = 6.0
        ab[1, 1] = 8.0
        ab[3, -2] = 8.0
        ab[4, -3] = 6.0
        coeffsb0 = np.array([ -43/12, -20/3, 9, 4/3, -1/12 ])
        B[0, 0:5]  = coeffsb0
        B[-1, -5:] = -coeffsb0[::-1]

        return ab, B

    def _init_KP4(self):
        '''
        #  This is the 4th-order compact operator defined in Wu and Kim (2024).
        #  The terms are defined in Table 3.
        '''
        N = self.N

        alpha = 0.5
        beta = 1.0 / 20

        a1 = 17.0 / 24
        a2 = 101.0 / 600
        a3 = 1.0 / 600
        coeffs = np.array([ -a3, -a2, -a1, 0.0, a1, a2, a3 ])

        # i = 0
        alpha01 = 43.65980335321481
        beta02  = 92.40143116322876

        b01     = -86.92242000231872
        b02     = 47.58661913475775
        b03     = 57.30693626084370
        b04     = -13.71254216556246
        b05     = 2.659826729790792
        b06     = -0.2598929200600359

        # i = 1
        alpha10 = 0.08351537442980239
        alpha12 = 1.961483362670730
        beta13  = 0.8789761422182460

        b10     = -0.3199960780333493
        b12     = 0.07735499170041915
        b13     = 1.496612372811008
        b14     = 0.2046919801608821
        b15     = -0.02229717539815850
        b16     = 0.001702365014746567

        # i = 2
        beta20  = 0.008073091519768687
        alpha21 = 0.2162434143850924
        alpha23 = 1.052242062502679
        beta24  = 0.2116022463346598

        b20     = -0.03644974757120792
        b21     = -0.4997030280694729
        b23     = 0.7439822445654316
        b24     = 0.5629384925762924
        b25     = 0.01563884275691290
        b26     = -0.0003043666146108995
    
        b00     = -(b01 + b02 + b03 + b04 + b05 + b06)
        b11     = -(b10 + b12 + b13 + b14 + b15 + b16)
        b22     = -(b20 + b21 + b23 + b24 + b25 + b26)

        # Construct tridiagonal A matrix in banded form
        ab = np.zeros((5, N))
        ab[0, 2:] = beta    # 2nd upper diagonal
        ab[1, 1:] = alpha   # 1st upper diagonal
        ab[2, :] = 1.0      # main diagonal
        ab[3, :-2] = alpha  # 1st lower diagonal
        ab[4, :-3] = beta   # 2nd lower diagonal

        B = np.zeros((N, N))
        for i in range(3, N-3):
            B[i, i-3:i+4] = coeffs

        # One-sided stencils
        ab[0, 2] = beta02
        ab[0, 3] = beta13
        ab[0, 4] = beta24
        ab[0, -1] = beta20

        ab[1, 1] = alpha01
        ab[1, 2] = alpha12
        ab[1, 3] = alpha23
        ab[1, -2] = alpha21
        ab[1, -1] = alpha10

        ab[3,  0] = alpha10
        ab[3,  1] = alpha21
        ab[3, -4] = alpha23
        ab[3, -3] = alpha12
        ab[3, -2] = alpha01

        ab[4, 0] = beta20
        ab[4, -5] = beta24
        ab[4, -4] = beta13
        ab[4, -3] = beta02

        coeffsb0 = np.array([ b00, b01, b02, b03, b04, b05, b06 ])
        coeffsb1 = np.array([ b10, b11, b12, b13, b14, b15, b16 ])
        coeffsb2 = np.array([ b20, b21, b22, b23, b24, b25, b26 ])

        B[0, 0:7]  = coeffsb0
        B[1, 0:7]  = coeffsb1
        B[2, 0:7]  = coeffsb2
        B[-3, -7:] = -coeffsb2[::-1]
        B[-2, -7:] = -coeffsb1[::-1]
        B[-1, -7:] = -coeffsb0[::-1]

        return ab, B

    def _init_SP4(self):
        '''
        #  This is the standard pentadiagonal compact finite difference
        #  operator based on a seven-point stencil defined in Table 1 of
        #  Wu and Kim (2024).
        '''
        N = self.N

        alpha = 0.5
        beta = 1.0 / 20

        a1 = 17.0 / 24
        a2 = 101.0 / 600
        a3 = 1.0 / 600
        coeffs = np.array([ -a3, -a2, -a1, 0.0, a1, a2, a3 ])

        # i = 0
        alpha01 = 12
        beta02  = 15

        b01     = -77/5
        b02     = 55/4
        b03     = 20/3
        b04     = -5/24
        b05     = 1/5
        b06     = -1/60

        # i = 1
        alpha10 = 1/18
        alpha12 = 5/2
        beta13  = 10/9

        b10     = -257/1080
        b12     = -5/24
        b13     = 55/27
        b14     = 5/24
        b15     = -1/60
        b16     = 1/1080

        # i = 2
        beta20  = 1/90
        alpha21 = 4/15
        alpha23 = 8/9
        beta24  = 1/6

        b20     = -34/675
        b21     = -127/225
        b23     = 20/27
        b24     = 4/9
        b25     = 1/75
        b26     = -1/2700
    
        b00     = -(b01 + b02 + b03 + b04 + b05 + b06)
        b11     = -(b10 + b12 + b13 + b14 + b15 + b16)
        b22     = -(b20 + b21 + b23 + b24 + b25 + b26)

        # Construct tridiagonal A matrix in banded form
        ab = np.zeros((5, N))
        ab[0, 2:] = beta    # 2nd upper diagonal
        ab[1, 1:] = alpha   # 1st upper diagonal
        ab[2, :] = 1.0      # main diagonal
        ab[3, :-2] = alpha  # 1st lower diagonal
        ab[4, :-3] = beta   # 2nd lower diagonal

        B = np.zeros((N, N))
        for i in range(3, N-3):
            B[i, i-3:i+4] = coeffs

        # One-sided stencils
        ab[0, 2] = beta02
        ab[0, 3] = beta13
        ab[0, 4] = beta24
        ab[0, -1] = beta20

        ab[1, 1] = alpha01
        ab[1, 2] = alpha12
        ab[1, 3] = alpha23
        ab[1, -2] = alpha21
        ab[1, -1] = alpha10

        ab[3,  0] = alpha10
        ab[3,  1] = alpha21
        ab[3, -4] = alpha23
        ab[3, -3] = alpha12
        ab[3, -2] = alpha01

        ab[4, 0] = beta20
        ab[4, -5] = beta24
        ab[4, -4] = beta13
        ab[4, -3] = beta02

        coeffsb0 = np.array([ b00, b01, b02, b03, b04, b05, b06 ])
        coeffsb1 = np.array([ b10, b11, b12, b13, b14, b15, b16 ])
        coeffsb2 = np.array([ b20, b21, b22, b23, b24, b25, b26 ])

        B[0, 0:7]  = coeffsb0
        B[1, 0:7]  = coeffsb1
        B[2, 0:7]  = coeffsb2
        B[-3, -7:] = -coeffsb2[::-1]
        B[-2, -7:] = -coeffsb1[::-1]
        B[-1, -7:] = -coeffsb0[::-1]

        return ab, B


    def _init_E4(self):
        '''
        #  This is the standard 4th-order explicit derivative operator
        #  written in matrix form.
        '''
        N = self.N
        coeffs = np.array([1.0, -8.0, 0.0, 8.0, -1.0]) / (12.0)

        B = np.zeros((N, N))
        for i in range(2, N-2):
            B[i,i-2:i+3] = coeffs

        # One-sided stencils
        coeff0 = np.array([-25.0, 48.0, -36.0, 16.0, -3.0]) / (12.0)
        coeff1 = np.array([-3.0, -10.0, 18.0, -6.0, 1.0]) / (12.0)
        B[0, 0:5] = coeff0
        B[1, 0:5] = coeff1
        B[-2, -5:] = -coeff1[::-1]
        B[-1, -5:] = -coeff0[::-1]

        return B

    def _init_E6(self):
        '''
        #  This is the standard 6th-order explicit derivative operator
        #  written in matrix form.  The boundary terms are 6-4-2 (2nd order
        #  at points i = 0,1 and 4th order at point i=2.)
        '''
        N = self.N
        coeffs = np.array([-1.0, 9.0, -45.0, 0.0, 45.0, -9.0, 1.0]) / (60.0)

        B = np.zeros((N, N))
        for i in range(3, N-3):
            B[i,i-3:i+4] = coeffs

        # One-sided stencils
        coeff0 = np.array([-3.0, 4.0, -1.0]) / (2.0)
        coeff1 = np.array([-1.0, 0.0, 1.0]) / (2.0)
        coeff2 = np.array([1.0, -8.0, 0.0, 8.0, -1.0]) / (12.0)
        B[0, 0:3] = coeff0
        B[1, 0:3] = coeff1
        B[2, 0:5] = coeff2
        B[-3, -5:] = -coeff2[::-1]
        B[-2, -3:] = -coeff1[::-1]
        B[-1, -3:] = -coeff0[::-1]

        return B


    def evaluate(self, f):
        idx = 1.0 / self.dx
        rhs = self.B @ f

        if self.bandwidth == 0:
            # explicit
            return rhs * idx

        if self.use_lu_solve:
            xx = blu.lu_solve_banded(self.lu_factorization, rhs, overwrite_b=self.overwrite, check_finite=self.checkf)
            return xx*idx
        else:
            return solve_banded(self.bands, self.ab, rhs) * idx


