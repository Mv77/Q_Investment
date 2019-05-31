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

from scipy import optimize

from Q_investment import Qmod
# %% {"code_folding": [40, 45]}
# Function definitions

def pathUtil(inv,mod1,mod2,k0,t):
    k = np.zeros(t)
    k[0] = k0
    value = 0
    for i in range(t-1):
        flow = mod1.flow(k[i],inv[i])
        value += flow*mod1.beta**i
        k[i+1] = k[i]*(1-mod1.delta) + inv[i]
    
    value += (mod1.beta**t)*mod2.value_func(k[t-1])
    return(value)
            
def future_change(mod1,mod2,k0,t,T,npoints = 100):
    
    fobj = lambda x: -1*pathUtil(x,mod1,mod2,k0,t)
    inv = optimize.minimize(fobj,x0 = np.ones(t-1)*mod2.kss*mod2.delta, options = {'disp': True}).x
    
    # Find path of capital and lambda
    k = np.zeros(T)
    lam = np.zeros(T)
    k[0] = k0 
    for i in range(0,T-1):
    
        if i < (t-1):
            k[i+1] = k[i]*(1-mod1.delta) + inv[i]
            lam[i] = mod1.findLambda(k[i],k[i+1])
        else:
            k[i+1] = mod2.k1Func(k[i])
            lam[i] = mod2.findLambda(k[i],k[i+1])
    
    lam[T-1] = mod2.findLambda(k[T-1],mod2.k1Func(k[T-1]))
    
    plt.figure()
    
    # Plot k,lambda path
    plt.plot(k,lam,'.k')
    plt.plot(k[t-1],lam[t-1],'.r')
    
    k_range = np.linspace(0.1*min(mod1.kss,mod2.kss),2*max(mod1.kss,mod2.kss),npoints)
    mods = [mod1,mod2]
    colors = ['r','b']
    labels = ['Pre-change','Post-change']
    for i in range(2):

        # Plot k0 locus
        plt.plot(k_range,mods[i].P*np.ones(npoints), linestyle = '--', color = colors[i],label = labels[i])
        # Plot lambda0 locus
        plt.plot(k_range,[mods[i].lambda0locus(x) for x in k_range], color = colors[i])
        # Plot steady state
        plt.plot(mods[i].kss,mods[i].P,marker = '*', color = colors[i])
    
    plt.title('Phase diagrams and model dynamics')
    plt.xlabel('K')
    plt.ylabel('Lambda')
    plt.legend()
    
    return((k,lam))

#%%

Q1 = Qmod(omega = 0.1, zeta = 0.1)
Q1.solve()
Q2 = Qmod(omega = 0.1, zeta = 0)
Q2.solve()
    
t = 10
T = 20
k0 = Q1.kss
sol = future_change(mod1 = Q1, mod2 = Q2, k0 = k0, t = t,T=T,npoints = 200)