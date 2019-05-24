# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.1'
#       jupytext_version: 0.8.5
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
#   language_info:
#     codemirror_mode:
#       name: ipython
#       version: 3
#     file_extension: .py
#     mimetype: text/x-python
#     name: python
#     nbconvert_exporter: python
#     pygments_lexer: ipython3
#     version: 3.7.1
# ---

# %% [markdown]
# # Numerical Solution of the Abel/Hayashi "q" investment model
#
# ## [Mateo Velásquez-Giraldo](https://github.com/Mv77)

# %% {"code_folding": []}
# Preamble
import numpy as np
import matplotlib.pyplot as plt

from scipy import interpolate
from scipy import optimize

# %% {"code_folding": [7, 31, 34, 37, 41, 59, 67, 81, 108, 121]}
# Class implementation

class Qmod:
    """
    A class representing the Q investment model.
    """
    
    def __init__(self,beta,tau,alpha,omega,zeta,delta):
        """
        Inputs:
        - Beta:
        - Tau:
        - Alpha:
        - Omega:
        - Zeta:
        - Delta
        """
        # Assign parameter values
        self.beta = beta
        self.tau = tau
        self.alpha = alpha
        self.omega = omega
        self.zeta = zeta
        self.delta = delta
        
        # Create empty consumption function
        self.k1Func = None
        
        # Steady state capital
        self.kss = ((1-(1-self.delta)*self.beta)/((1-self.tau)*self.alpha))**(1/(self.alpha-1))
        
    def f_k(self,k):
        return(self.alpha*k**(self.alpha-1))
    
    def j(self,i,k):
        return(k/2*((i-self.delta*k)/k)**2*self.omega)
    
    def j_i(self,i,k):
        iota = i/k - self.delta
        return(iota*self.omega)
    
    def j_k(self,i,k):
        iota = i/k - self.delta
        return(-(iota**2/2+iota*self.delta)*self.omega)
    
    # Error in the euler equation implied by a k_0, k_1, k_2 triad.
    # This can be solved to obtain the adequate triads.
    def eulerError(self,k0,k1,k2):
        
        i0 = k1 - (1-self.delta)*k0
        i1 = k2 - (1-self.delta)*k1
        error = (1+self.j_i(i0,k0)) -\
        ((1-self.tau)*self.f_k(k1) +\
         ((1-self.delta) + (1-self.delta)*self.j_i(i1,k1) - self.j_k(i1,k1))*self.beta)
        
        return(error)
    
    # Find the k_2 implied by the euler equation for an initial k_0,
    # k_1.
    def k2(self,k0,k1):
        
        sol = optimize.root_scalar(lambda x: self.eulerError(k0,k1,x), x0=k0, x1=self.kss).root
        
        return(sol)
    
    # Find the capital trajectory implied by the euler equation for
    # an initial k_0, k_1.
    def shoot(self,k0,k1,t):
        k = np.zeros(t)
        k[0] = k0
        k[1] = k1
        for i in range(2,t):
            
            k[i] = self.k2(k[i-2],k[i-1])
            if k[i]<0 or (abs(k[i]-self.kss) > 2*abs(k0-self.kss)):
                k[i:] = k[i]
                return(k)
            
        return(k)
    
    # Shooting algorithm to find k_1 given k_0.
    def find_k1(self,k0,T=30,tol = 10**(-6),maxiter = 200):
            
        top = max(self.kss,k0)
        bot = min(self.kss,k0)
        
        for k in range(maxiter):
            
            init = (top+bot)/2
            path = self.shoot(k0,init,T)
    
            k_f = path[-1]
            if np.isnan(k_f):
                bot = init
            else:
                if abs(k_f - self.kss)<tol:
                    return(init)
                else:
                    if k_f >= self.kss:
                        top = init
                    else:
                        bot = init
            
        return(init)
    
    # Construction of the policy rule by solving for k_1 given
    # k_0 over a grid of points and then finding an interpolating
    # function
    def solve(self,k_min=10**(-4), n_points = 50):
        
        k_max = 4*self.kss
        k0 = np.linspace(k_min,k_max,n_points)
        k1 = np.zeros(len(k0))
        
        for i in range(len(k0)):
            
            k1[i] = self.find_k1(k0[i])
        
        self.k1Func = interpolate.interp1d(k0,k1)
    
    # Simulation of capital dynamics from a starting k_0
    def simulate(self,k0,t):
        k = np.zeros(t)
        k[0]=k0
        for i in range(1,t):
            k[i] = self.k1Func(k[i-1])
        return(k)
        
    def iota(self,lam_1):
        iota = (lam_1-1)/self.omega
        return(iota)
    
    def jkl(self,lam_1):
        iota = self.iota(lam_1)
        jk = -(iota**2/2+iota*self.delta)*self.omega
        return(jk)
        
    def dLambda(self,k,lam):
        
        bdel = self.beta*(1-self.delta)
        # dLambda solves the following equation:
        error = lambda x: ((1-bdel)*lam-(1-self.tau)*self.f_k(k) + self.jkl(lam+x)*self.beta)/bdel - x
        sol = optimize.root_scalar(error, bracket = [-1,1])
        
        return(sol)
    
    def dK(self,k,lam):
        iota = (lam-1)/self.omega
        return(iota*k)
    
    def lambda0locus(self,k):
        
        if k > self.kss:
            x1 = 0.5
        else:
            x1 = 1.5
            
        bdel = self.beta*(1-self.delta)
        # Lambda solves the following equation:
        error = lambda x: (1-bdel)*x - (1-self.tau)*self.f_k(k) + self.jkl(x)*self.beta
        lam = optimize.root_scalar(error, x0 = 1, x1 = x1).root
        
        return(lam)
        
    def phase_diagram(self, npoints = 200, arrows = False, n_arrows = 5):
        """
        Plots the model's phase diagram.
        - npoints:  number of ticks in the k axis.
        - arrows:   boolean to indicate whether or not to draw arrow
                    grid.
        - n_arrows: controls the number of arrows in the grid
        """
        
        k = np.linspace(0.9*self.kss,1.1*self.kss,npoints)
        
        # Plot k0 locus
        #plt.plot(k,self.k0locus(k),label = '$\\dot{k}=0$ locus')
        # Plot lambda0 locus
        plt.plot(k,[self.lambda0locus(x) for x in k],label = '$\\dot{\\lambda}=0$ locus')
        # Plot saddle path
        #plt.plot(k,self.cFunc(k), label = 'Saddle path')
        # Plot steady state
        plt.plot(self.kss,1,'*r', label = 'Steady state')
        
        # Add arrows ilustrating behavior in different parts of
        # the diagram.
        # Taken from:
        # http://systems-sciences.uni-graz.at/etextbook/sw2/phpl_python.html
        if arrows:
            x = np.linspace(k[0],k[-1],n_arrows)
            y = np.linspace(0.5,1.5,n_arrows)
            
            X, Y = np.meshgrid(x,y)
            print(type(X))
            dLambda = self.dLambda(X,Y)
            dK = self.dK(X,Y)
            
            M = (np.hypot(dK, dLambda))
            M[ M == 0] = 1.
            dK /= M
            dLambda /= M
            plt.quiver(X, Y, dK, dLambda, M, pivot='mid', alpha = 0.3)
        
        # Labels
        plt.title('Phase diagram and consumption rule\n(normalized by efficiency units)')
        plt.xlabel('K')
        plt.ylabel('Lambda')
        plt.legend()
        plt.show()


# %% [markdown]
# ## _Examples_

# %%
# Create model object
Qexample = Qmod(beta = 0.99,tau = 0, alpha = 0.33, omega =  0.5, zeta =  0, delta = 0.05)
# Solve to find the policy rule (k[t+1] in terms of k[t])
Qexample.solve()

Qexample.phase_diagram(arrows = False, n_arrows = 5)
# %%
# Plot policy rule

k = np.linspace(1,3*Qexample.kss,20)

plt.plot(k,[Qexample.k1Func(x) for x in k])
plt.title('Policy Rule')
plt.xlabel('k(t)')
plt.ylabel('k(t+1)')
plt.show()

# %%
# Find capital dynamics from a given starting capital
k0 = 23
t = 50
k = Qexample.simulate(k0,t)

# Plot
plt.figure()
plt.plot(k)
plt.axhline(y = Qexample.kss,linestyle = '--',color = 'k', label = '$\\bar{k}$')
plt.title('Capital')
plt.xlabel('Time')
plt.legend()
plt.show()

# %% [markdown]
# ### The impact of adjustment costs

# %%
# Create and solve two instances, one with high and one with low adjustment costs omega
Qlow  = Qmod(beta = 0.99,tau = 0, alpha = 0.33, omega =  0.2, zeta =  0, delta = 0.05)
Qhigh = Qmod(beta = 0.99,tau = 0, alpha = 0.33, omega =  2.0, zeta =  0, delta = 0.05)

Qlow.solve()
Qhigh.solve()

# Simulate adjustment from an initial capital level
k0 = 20
t = 50
k_low = Qlow.simulate(k0,t)
k_high = Qhigh.simulate(k0,t)

# Plot
plt.figure()
plt.plot(k_low, label = 'Low $\\omega$')
plt.plot(k_high, label = 'High $\\omega$')
plt.axhline(y = Qexample.kss,linestyle = '--',color = 'k', label = 'Steady state ${k}$')
plt.title('Capital')
plt.xlabel('Time')
plt.legend()
plt.show()
