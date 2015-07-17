# -*- coding: utf-8 -*-
"""
Didactic Runge-Kutta for learning of Python.

@author: Daniel Casas-Orozco
"""
from __future__ import division
import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt

def r_k(func, y_0, tspan, n):
    """Integrates the ODE system using Runge-Kutta 4
    
    Parameters
    ----------
    func : function
        Iteration function, written as a first order system.
    y_0 : array_like (n_var)
        Initial conditions for the variables.
    tspan : tuple
        Initial and end time.
    n : integer
        Number of steps
        
    Returns
    -------
    step : array_like (n)
        Time vector.
    y_n : array (n_var, n)
        Solution.
    h : float
        Time step.
    
    """
    
    n_var = np.size(y_0)  
    y_n = np.zeros((n_var,n+1))    
    y_n[:,0] = y_0
    h = (tspan[1] - tspan[0])/n
    step = np.linspace(tspan[0], tspan[-1], n+1)
    
    for pos, tn in enumerate(step[:-1]):
        yn = y_n[:,pos]
                
        k1 = func(tn,yn)
        k2 = func(tn + h/2, yn + h/2*k1)
        k3 = func(tn + h/2, yn + h/2*k2)
        k4 = func(tn + h, yn + h*k3)
    
        
        y_n[:,pos+1] =  yn + h/6*(k1 + k4) + h/3*(k2 + k3)
        
    return step, y_n, h
    
    
if __name__ == "__main__":
    # Simple ODE
    def f(t, y):
        return -2*y


    # Simple pendulum
    def pendulum(t, x):
        """Simple pendulum"""
        return np.array([x[1], -np.sin(x[0])])

    # Harmonic oscilator
    def harm_osc(t, x):
        """Simple pendulum"""
        return np.array([x[1], -x[0]])    
        
    # Solve simple ODE
    t, y, h = r_k(f, 1., [0., 4.], 40)
    plt.figure()
    plt.plot(t, y[0,:])
    plt.title('Simple ODE')
    plt.xlabel('Time')
    plt.ylabel('Position')

    # Pendulum       
    time, y_vec, h = r_k(pendulum, [1., 0.], [0., 4*np.pi], 100)
    plt.figure()
    plt.plot(time, y_vec[0,:])
    plt.title('Simple pendulum')
    plt.xlabel('Time')
    plt.ylabel('Position')

    # Harmonic oscilator
    plt.figure()
    N_vec = np.array([100, 200, 500,1000, 2000, 5000, 10000])
    error_vec = np.zeros((len(N_vec)))
    for cont, N in enumerate(N_vec):
        time, y_vec, _ = r_k(harm_osc, [1., 0.], [0., 4*np.pi], N)
        error_vec[cont] = norm(y_vec[0,:] - np.cos(time))/norm(np.cos(time))
    
    coef = np.polyfit(np.log(4*np.pi/N_vec), np.log(error_vec), 1)
    print "The approximated order of convergence is: %g" % coef[0]
    plt.loglog(4*np.pi/N_vec, error_vec)
    plt.title('Harmonic oscilator')
    plt.xlabel('Step')
    plt.ylabel('Error')
    
    plt.show()