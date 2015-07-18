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
    r"""Integrates the ODE system using Runge-Kutta 4
    
    Parameters
    ----------
    func : function
        Iteration function, written as a first order system.
    y_0 : array_like (n_var)
        Initial conditions for the variables.
    tspan : array_like (2)
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
        
    Integration formula
    ---------

        Given a set of of ODE's, written as a first-order system:
        
        .. math::
        
            \frac{d \mathbf{y}}{dt} = f(t,\mathbf{y}),
        
        the generalized Runge Kutta integration formula has the form:
        
        .. math::
        
            \begin{align}
            y_{n+1} &= y_n + h \sum_{i = 1}^{s} b_i k_i \\
            k_i &= f \left(t_n + c_i h , y_n + h \sum_{j=1}^{s} a_{i,j} k_j \right).
            \end{align}
            
            \begin{array}{c|cccc}
            c_1    & a_{11} & a_{12}& \dots & a_{1s}\\
            c_2    & a_{21} & a_{22}& \dots & a_{2s}\\
            \vdots & \vdots & \vdots& \ddots& \vdots\\
            c_s    & a_{s1} & a_{s2}& \dots & a_{ss}\\
            \hline\\ \\
            & b_1    & b_2   & \dots & b_s\\
            \end{array}
        
        
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
    plt.figure(dpi=300)
    t_ho, y_ho, h_ho = r_k(harm_osc, [1.,0.], [0.,4*np.pi], 256)
    ho_plot = plt.plot(t_ho, y_ho.T)
    plt.legend(ho_plot, ('Position','Velocity'),loc='upper left')
    plt.xlabel('Time', position = (1, 1))
    plt.ylabel('Position/Velocity')
    plt.xlim(t_ho.min()*.9, t_ho.max()*1.1)
    
    plt.xticks(np.arange(0, 5*np.pi, np.pi),
               [r'$0$', r'$\pi$', r'$2 \pi$', r'$3 \pi$', r'$4 \pi$'])


    plt.ylim(np.min(y_ho)*1.1, np.max(y_ho)*1.1)
    plt.yticks(np.arange(-1, 1, .5),
               [r'$-1$', r'$-0.5$', r'$0$', r'$0.5$', r'$1$'])
    
    
    ax = plt.gca()
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.spines['bottom'].set_position(('data', 0))
    ax.spines['left'].set_position(('data', 0))
    
#    plt.scatter([np.pi, ], [0, ], 50, color='green')
#    plt.annotate('When velocity goes from negative to possitive...', xy=(np.pi, 0),
#                 xycoords='data', xytext=(+10, +30), textcoords='offset points')
    
    
    N_vec = np.array([100, 200, 500,1000, 2000, 5000, 10000])
    error_vec = np.zeros((len(N_vec)))
    for cont, N in enumerate(N_vec):
        time, y_vec, _ = r_k(harm_osc, [1., 0.], [0., 4*np.pi], N)
        error_vec[cont] = norm(y_vec[0,:] - np.cos(time))/norm(np.cos(time))
    
    coef = np.polyfit(np.log(4*np.pi/N_vec), np.log(error_vec), 1)
    print "The approximated order of convergence is: %g" % coef[0]
    plt.figure()
    plt.loglog(4*np.pi/N_vec, error_vec)
    plt.title('Harmonic oscilator')
    plt.xlabel('Step')
    plt.ylabel('Error')
    
    plt.show()
