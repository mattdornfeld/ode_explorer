import warnings
import numpy as np
from numpy import sin, arctan, cos, cosh, sinh
from numpy.random import rand
from scipy.special import j1, struve
from scipy.optimize import fsolve
from math import pi
from cmath import sqrt
from constants import *
import matplotlib.pyplot as plt 

warnings.filterwarnings("error")

def dagger(mat):
	""" Calculates conjugate transpose of an array"""
	return np.conj(np.transpose(mat))

def calc_phi(x, s):
	"""Calculates spatial eigenfunctions """
	return np.cosh(s*x)

def chi(omega, r_mouth):
	"""Reactive part of impedance"""
	return struve(1, omega * r_mouth) / omega / r_mouth

def R(omega, r_mouth):
	"""Resistive part of impedance"""
	return 1 - j1(2 * omega * r_mouth) / omega / r_mouth

def real_bc(omega, alpha, r_mouth):
	return ( cos(omega) * cosh(alpha) - chi(omega, r_mouth) * sin(omega) * 
		cosh(alpha) - R(omega, r_mouth) * cos(omega) * sinh(alpha) )

def imag_bc(omega, alpha, r_mouth):
	return ( R(omega, r_mouth) * sin(omega) * cosh(alpha) - 
		chi(omega, r_mouth) * cos(omega) * sinh(alpha) - 
		sin(omega) * sinh(alpha) )

def eqns(s, r_mouth):
	"""Solve real_bc and imag_bc to get spatial eigenvalues"""
	omega, alpha = s 
	return ( real_bc(omega, alpha, r_mouth), imag_bc(omega, alpha, r_mouth) )

def generate_seeds(num_seeds, max_x, max_y):
	seeds = rand(num_seeds, 2)
	seeds[:, 0] = max_x * seeds[:, 0]
	seeds[:, 1] = max_y * seeds[:, 1]

	return seeds

def unique_rows(data):
	""" Finds unique rows of array """

	ncols = data.shape[1]
	dtype = data.dtype.descr * ncols
	struct = data.view(dtype)

	uniq = np.unique(struct)
	uniq = uniq.view(data.dtype).reshape(-1, ncols)
	
	return uniq

def process(evs, nm):
	""" Combines evs into single array, 
	deletes copies, and evs less than 0 """

	evs = np.array(evs)
	evs = np.around(evs, 6)
	evs = unique_rows(evs)
	idx = evs[:,0]>0
	evs = evs[idx, :]
	evs = evs[0:nm]

	return evs

def plot_evs(evs, seeds, r_mouth, max_x, max_y):
	omega = np.linspace(0.0001, max_x)
	alpha = np.linspace(0, max_y)[:, None]
	plt.close("all")
	plt.style.use('ggplot')
	fig, ax = plt.subplots()
	ax.contour(omega, alpha.ravel(), real_bc(omega, alpha, r_mouth), [0], 
		colors='blue')
	ax.contour(omega, alpha.ravel(), imag_bc(omega, alpha, r_mouth), [0], 
		colors='orange')
	ax.plot(seeds[:,0], seeds[:,1], 'go')
	ax.plot(evs[:,0], evs[:,1], 'ro')
	ax.set_xlabel(r'$\omega$')
	ax.set_ylabel(r'$\alpha$')
	fig.show()

def calc_spatial_evs(r_mouth, nm, num_seeds, max_x, max_y):
	#evs = Parallel(n_jobs=4)(delayed(fsolve)(eqns, x0, args=(r_mouth)) 
	#		x0 fro in seeds)
	i = 1
	while i<10:
		try:
			seeds = generate_seeds(100, max_x, max_y)
			evs = [fsolve(eqns, x0, args=(r_mouth)) for x0 in seeds]
			break
		except:
			print("Spatial eigenvalues were not found. Trying again.")
			i+=1

	evs = process(evs, nm)
	s = evs[:, 0] + j * evs[:, 1]
	plot_evs(evs, seeds, r_mouth, max_x, max_y)
	return s

def calc_spatial_parameters(s, dx = 0.00001):
	x = np.arange(0, 1 , dx)

	phi = np.array([calc_phi(x, _ ) for _ in s])
	phi = phi.transpose()

	overlap = np.dot( phi.conj().transpose(), phi * dx)
	evs, T = np.linalg.eig(overlap)
	lambda_inv = np.diag( 1 / evs )

	phi_tilde = np.dot(phi, T)
	a = T.dot( lambda_inv.dot( dagger(phi_tilde).dot( np.ones(len(x)) * dx ) ) )
	b = T.dot( lambda_inv.dot( dagger(phi_tilde).dot( 0.5*(x-1)**2 ) * dx) )

	return a, b

def calc_parameters(p, r_front, r_back, omega_d, Gamma, alpha, omega, nm):
	ce = r_pharynx**2 / r_front**2 
	cc = r_front**2 / r_trachea**2
	cf = r_back**2 / r_front**2
	le = l / np.sqrt(cf)
	ce=1
	mu = ce*le + 0.5
	gamma = ce**2 * (1 - cc**2)

	#zeta = R * ce * le

	u0_star = np.sqrt( 2 * p / gamma ) 
	print(2 * u0_star * L / r_front)
	omega_d = 2 * u0_star * L / r_front * omega_d

	a, b = calc_spatial_parameters(alpha + j*omega)

	M_inv = np.eye(nm) - np.tile( b.real, (nm,1) ).T / mu
	M = np.linalg.inv(M_inv)

	A = np.dot(M, a.real)
	B = np.dot(M, b.real)

	beta = (2 * np.sqrt(omega**2 - alpha**2) 
		* sin( 0.5 * arctan( 2*alpha*omega / (omega**2 - alpha**2) ) ) )

	return nm, p, mu, gamma, alpha, omega, omega_d, Gamma, beta, A, B, M

def calc_parameters2(p, mu, gamma, alpha, omega, nm, zeta):
	a, b = calc_spatial_parameters(alpha + j*omega)

	M_inv = np.eye(nm) - np.tile( b.real, (nm,1) ).T / mu
	M = np.linalg.inv(M_inv)

	A = np.dot(M, a.real)
	B = np.dot(M, b.real)

	beta = (2 * np.sqrt(omega**2 - alpha**2) 
		* sin( 0.5 * arctan( 2*alpha*omega / (omega**2 - alpha**2) ) ) )

	return (nm, p, mu, gamma, zeta, alpha, omega, beta, A, B, M)

if __name__ == "__main__":
	r_mouth = 0.0005
	nm = 3
	p = 0.01
	r_front = 0.001
	r_back = 0.0015
	zeta = 1.1
	s = calc_spatial_evs(r_mouth=r_mouth/L, nm=nm, num_seeds=100, max_x=20, max_y=3)
	params = calc_parameters(p, r_front, r_back, s.imag, s.real, nm, zeta)
	nm, p, mu, gamma, zeta, alpha, omega, omega_d, Gamma, beta, A, B, M = params