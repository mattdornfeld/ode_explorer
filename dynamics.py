import os
import sys 
from math import tanh, cosh, sin, cos
from contextlib import contextmanager
from itertools import izip_longest
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy import integrate
from math import pi
from joblib import Parallel, delayed
from constants import *
import calc_parameters as cp
from IPython import embed

eps = 0
plt.style.use('bmh')
@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:  
            yield
        finally:
            sys.stdout = old_stdout

def odeint(flow, inits, params, fs, T, jac=None, solver='dopri5', method='bdf'):
	ode = integrate.ode(flow, jac)
	ode.set_integrator(solver)
	ode.set_initial_value(inits, 0.0)
	ode.set_f_params(params)
	nm = params[0]
	with suppress_stdout():
		q = [inits] + [ode.integrate(ode.t + 1/fs) for _ in range( int(T*fs) + 1 )]

	return np.array(q)

def flow(y, t, params):
	"""
	y[0]=u0(t), y[1]=f_vortex(t), y[2:nm+2]= q(t), y[nm+2]=f_vortex'(t), y[nm+3:]= q'(t)
	ydot[0]=u0'(t), ydot[1]=f_vortex'(t), ydot[2:nm+2]= q'(t), ydot[nm+2]=f_vortex''(t), ydot[nm+3:]= q''(t)
	"""
	nm, p, mu, gamma, alpha, omega, omega_d, Gamma, beta, A, B, M = params

	p_src = Gamma * y[1]
	dp_src = Gamma * y[nm+2]

	A.shape = (nm,)
	B.shape = (nm,)
	
	ydot = np.zeros(2*nm+3)

	ydot[0] = (p / mu + p_src + sum( y[nm+1:] ) / mu - 0.5 * gamma * y[0]**2 /mu)

	ydot[1] = y[nm+2]

	ydot[2:nm+2] = y[nm+3:]

	ydot[nm+2] = - omega_d**2 * y[1]
	
	ydot[nm+3:] = ( -np.dot(M, beta * y[nm+3:])  
		- np.dot(M, (omega**2 - alpha**2) * (y[2:nm+2])) 
		-  + B * gamma / mu**2 * y[0] * sum(y[nm+3:]) 
		+ (- A - B * p * gamma / mu**2) * y[0] 
		+ 0.5 * B * gamma**2 / mu**2 * y[0]**3 
		+ B * gamma / mu * dp_src - B * gamma / mu**2 * p_src * y[0])

	return ydot

def calc_fixed_points(params):
	nm, p, mu, gamma, alpha, omega, omega_d, Gamma, beta, A, B, M = params

	a = np.dot(np.linalg.inv(M), A)

	u0_star = np.sqrt(2 * gamma * p) / gamma 
	qn_star = - a / (omega**2 - alpha**2) * u0_star

	fp1 = np.hstack( (u0_star, np.zeros(1), qn_star, np.zeros(nm + 1)) )
	fp2 = -fp1

	return fp1, fp2

def calc_jacobian(q0, params):
	"""
	y[0]=u0(t), y[1]=f_vortex(t), y[2:nm+2]= q(t), y[nm+2]=f_vortex'(t), y[nm+3:]= q'(t)
	ydot[0]=u0'(t), ydot[1]=f_vortex'(t), ydot[2:nm+2]= q'(t), ydot[nm+2]=f_vortex''(t), ydot[nm+3:]= q''(t)
	"""
	nm, p, mu, gamma, alpha, omega, omega_d, Gamma, beta, A, B, M = params

	print(A)
	print(B)
	u0 = q0[0]
	f_vort = q0[1]
	qn = q0[2:nm+2]
	df_vort = q0[nm+2]
	qn_dot = q0[nm+3:]
	jac = np.zeros((2*nm+3, 2*nm+3))

	jac[0, 0] =  -gamma / mu * u0   
	jac[0, 1] = Gamma / mu
	jac[0, 2:nm+2] = 0
	jac[0, nm+2] = 0
	jac[0, nm+3:2*nm+3] = 1 / mu

	jac[1, 0] = 0
	jac[1, 1] = 0
	jac[1, 2:nm+2] = 0
	jac[1, nm+2] = 1
	jac[1, nm+3:2*nm+3] = 0

	jac[2:nm+2, 0] = 0
	jac[2:nm+2, 1] = 0
	jac[2:nm+2, 2:nm+2] = 0 
	jac[2:nm+2, nm+2] = 0
	jac[2:nm+2, nm+3:2*nm+3] = np.eye(nm)

	jac[nm+2, 0] = 0
	jac[nm+2, 1] = - omega_d**2
	jac[nm+2, 2:nm+2] = 0
	jac[nm+2, nm+2] = 0
	jac[nm+2, nm+3:2*nm+3] = 0

	jac[nm+3:2*nm+3, 0] = (
		-B * gamma / mu**2 * Gamma * f_vort - A - B * gamma / mu**2 * p 
		- B * gamma / mu**2 * sum(qn_dot) 
		+ 1.5 * B * gamma**2 / mu**2 * u0**2 )
	jac[nm+3:2*nm+3, 1] = - B * gamma / mu**2 * Gamma * u0
	jac[nm+3:2*nm+3, 2:nm+2] = - M * ( omega**2 - alpha**2 )
	jac[nm+3:2*nm+3, nm+2] = B * gamma * Gamma / mu 
	jac[nm+3:2*nm+3, nm+3:2*nm+3] = ( 
		- M * beta - np.tile( B, (nm,1) ).T * gamma / mu**2 * u0 )

	return jac

def by_term(q, params):
	nm, p, mu, gamma, alpha, omega, omega_d, Gamma, beta, A, B, M = params
	A.shape = (nm,1)
	B.shape = (nm,1)
	

	radiation = np.dot(M, (beta * q[:,nm+1:]).T).T
	shedding = (B * gamma / mu**2 * q[:,0] * q[:,nm+1:].sum(1)).T
	vocal_fold_friction = (B * zeta / mu**2  * q[:,nm+1:].sum(1)).T
	dissipation = radiation + shedding + vocal_fold_friction
	elastic = np.dot(M, ( (omega**2 - alpha**2) * (q[:,1:nm+1]) ).T ).T
	linear = ((B * zeta**2 / mu**2 - A - B * p * gamma / mu**2) * q[:, 0]).T
	quadratic = (1.5 * B * zeta * gamma / mu**2 * q[:,0]**2).T
	cubic = (0.5 * B * gamma**2 / mu**2 * q[:,0]**3).T

	A.shape = (nm,)
	B.shape = (nm,)

	return ( radiation, shedding, vocal_fold_friction, 
		dissipation, elastic, linear, quadratic, cubic )


if __name__ == '__main__':
	nm = 3
	p = 0.01
	r_front = 0.002	
	r_back = 0.002
	r_mouth = 0.00001
	zeta = 0
	s = cp.calc_spatial_evs(r_mouth=r_mouth/L, nm=nm, num_seeds=100, max_x=20, max_y=3)
	params = cp.calc_parameters(p, r_front, r_back, s.imag, s.real, nm, zeta)
	nm, p, mu, gamma, zeta, alpha, omega, beta, A, B, M = params
	fs = 10. #time sampling frequency
	T = 1000. #total time
	t = np.arange(0, T, 1/fs)
	fp1, fp2 = calc_fixed_points(params)
	#q0 = fp1 + 1e-3 * abs(np.random.rand(2*nm+1))
	q0 = np.zeros(2*nm+1)
	q = integrate.odeint(flow, q0, t, args = (params,) )
	radiation = np.dot(M, (beta * q[:,nm+1:]).T).T
	elastic = np.dot(M, ( (omega**2 - alpha**2) * (q[:,1:nm+1]) ).T ).T
	B.shape = (nm,1)
	A.shape = (nm,1)
	shedding = (B * gamma / mu**2 * q[:,0] * sum(q[:,nm+1:], 1)).T
	vocal_fold_friction = (B * zeta / mu**2  * sum(q[:,nm+1:], 1)).T
	dissipation = radiation + shedding + vocal_fold_friction
	linear = ((B * zeta**2 / mu**2 - A - B * p * gamma / mu**2) * q[:, 0]).T
	quadratic = (1.5 * B * zeta * gamma / mu**2 * q[:,0]**2).T
	cubic = (0.5 * B * gamma**2 / mu**2 * q[:,0]**3).T
	fig, ax = plt.subplots(4,2)
	ax[0,0].plot(radiation)
	ax[0,0].set_title("radiation")
	ax[1,0].plot(shedding)
	ax[1,0].set_title("shedding")
	ax[2,0].plot(vocal_fold_friction)
	ax[2,0].set_title("vocal_fold_friction")
	ax[3,0].plot(dissipation)
	ax[3,0].set_title("dissipation")
	ax[0,1].plot(elastic)
	ax[0,1].set_title("elastic")
	ax[1,1].plot(linear)
	ax[1,1].set_title("linear")
	ax[2,1].plot(quadratic)
	ax[2,1].set_title("quadratic")
	ax[3,1].plot(cubic)
	ax[3,1].set_title('cubic')
	labels=['mode1', 'mode2', 'mode3']
	ax[0,0].legend(ax[0,0].lines, labels)

