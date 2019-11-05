# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.2'
#       jupytext_version: 1.2.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # YOUR NAME
#
# This notebook is a template showing you how to use some tools that will be useful for problem set 5

# %% {"code_folding": []}
# Preamble: import the packages we will be using
import numpy as np
import matplotlib.pyplot as plt

from copy import deepcopy
from scipy import optimize

from dolo import *
import dolo.algos.perfect_foresight as pf
import dolo.algos.value_iteration as vi

import pandas as pd

# Import the Qmod python code from external file

# Since the Qmod class is in other folder we need to
# change the path.
import sys
sys.path.append('../')
from Qmod.Q_investment import Qmod

# %% {"code_folding": [0]}
def phase_diagrams(mod1,mod2,npoints = 300):
    """
    Computes (optimal) capital and lambda dynamics in face of a structural
    change in the Q investment model.

    Parameters:
        - mod1    : Qmod object representing the parameter values prevailing
                    from time 0 to t_change-1.
        - mod2    : Qmod object representing the parameter values prevailing
                    from time t_change onwards.
        - k0      : initial value for capital.
        - t_change: time period at which the structural change takes place. It
                    is assumed that the change is announced at period 0.
        - T_sim   : final time period of the simulation.
        - npoints : number of points in the capital grid to be used for phase
                    diagram plots.
    """

    # Create a figure
    fig, ax = plt.subplots()

    # Plot the loci of the pre and post-change models.
    k_range = np.linspace(0.1*min(mod1.kss,mod2.kss),2*max(mod1.kss,mod2.kss),
                          npoints)
    mods = [mod1,mod2]
    colors = ['r','b']
    labels = ['Pre-change','Post-change']
    for i in range(2):

        # Plot k0 locus
        ax.plot(k_range,mods[i].P*np.ones(npoints),
                 linestyle = '--', color = colors[i],label = labels[i])
        # Plot lambda0 locus
        ax.plot(k_range,[mods[i].lambda0locus(x) for x in k_range],
                 linestyle = '--', color = colors[i])
        # Plot steady state
        ax.plot(mods[i].kss,mods[i].P,marker = '*', color = colors[i])

    return(ax)

def simul_change_dolo(model, k0,  exog0, exog1, t_change, T_sim):

    # The first step is to create time series for the exogenous variables
    exog = np.array([exog1,]*(T_sim - t_change))
    if t_change > 0:
        exog = np.concatenate((np.array([exog0,]*(t_change)),
                               exog),
                              axis = 0)
    exog = pd.DataFrame(exog, columns = ['R','tau','itc_1','psi'])

    # Simpulate the optimal response
    dr = pf.deterministic_solve(model = model,shocks = exog, T=T_sim,
                                verbose=True, s1 = k0)

    # Dolo uses the first period to report the steady state
    # so we ommit it.
    return(dr[1:])


# %% [markdown]
# Now I create a base model parametrization using both the Qmod class and the Dolo implementation.

# %%
# Base parameters

# Discount factor and return factor
beta = 0.98
R = 1/beta

# Tax rate
tau = 0.05

# Share of capital in production
alpha = 0.33

# Adjustment costs
omega = 1

# Investment tax credit
zeta = 0

# Depreciation rate
delta = 0.1

# Technological factor
psi = 1


## Qmod python class
Qmodel = Qmod(beta, tau, alpha, omega, zeta, delta, psi)
Qmodel.solve()

## Dolo
QDolo = yaml_import("../Dolo/Q_model.yaml")
# We do not pass psi, tau, or zeta since they are handled not as parameters
# but exogenous variables.
QDolo.set_calibration(R = R, alpha = alpha, delta = delta, omega = omega)

# %% [markdown]
# ## Examples: changes in productivity.
# %% {"code_folding": [0]}
# Set up the basic features of the simulation

# Total simulation time
T = 20
# Initial level of capital
k0 = Qmodel.kss

# Level to be changed
psi_high = 1.3

# 
# Copy the initial model, set the higher psi and re-solve
Q_high_psi = deepcopy(Qmodel)
Q_high_psi.psi = psi_high
Q_high_psi.solve()

# Draw the two phase diagrams and save them in an object
# to add the plots of dynamics later.
ax = phase_diagrams(mod1 = Qmodel, mod2 = Q_high_psi)

# Now use dolo to find the optimal response to the shock.

# First design the shock:
# for this example, I am assuming that productivity increases and then
# goes back to the original level.
Psi_sequence = np.array([psi]*3+
                        [psi_high]*5+
                        [psi]*(T-3-5))
# Check the pattern of the shock
plt.figure()
plt.plot(Psi_sequence)

# Dolo receives a DataFrame with the full future paths for ALL exogenous
# variables. So we create one:
Exog = pd.DataFrame({'R':[R]*T,
                     'tau':[tau]*T,
                     'itc_1':[zeta]*T,
                     'psi':Psi_sequence})

soldolo = simul_change_dolo(model = QDolo, k0 = np.array([k0]),
                            exog0 = [R,tau,zeta,psi],
                            exog1 = [R,tau,zeta,psi_new],
                            t_change = t, T_sim = T)

# Plot the path of capital under both solutions
time = range(T)
plt.figure()
plt.plot(time, sol['k'], 'x', label = 'Qmod', alpha = 0.8)
plt.plot(time, soldolo['k'], '+', label = 'Dolo', alpha = 0.8)
plt.legend()
plt.title('Capital dynamics')
plt.ylabel('$k_t$ : capital')
plt.xlabel('$t$ : time')
# %% [markdown]
# ## 2. An increase in productivity announced at t=0 but taking effect at t=5
# %% {"code_folding": []}
# Repeat the calculation now assuming the change happens at t=5
t = 5

# Qmod class
sol = structural_change(mod1 = Qmodel, mod2 = Q_high_psi,
                        k0 = k0, t_change = t,T_sim=T)

# Dolo
soldolo = simul_change_dolo(model = QDolo, k0 = np.array([k0]),
                            exog0 = [R,tau,zeta,psi],
                            exog1 = [R,tau,zeta,psi_new],
                            t_change = t, T_sim = T)

# Plot the path of capital under both solutions
time = range(T)
plt.figure()
plt.plot(time, sol['k'], 'x', label = 'Qmod', alpha = 0.8)
plt.plot(time, soldolo['k'], '+', label = 'Dolo', alpha = 0.8)
plt.legend()
plt.title('Capital dynamics')
plt.ylabel('$k_t$ : capital')
plt.xlabel('$t$ : time')
# %% [markdown]
# ## 3. An unanticipated corporate tax-cut
# %% {"code_folding": [0]}
# Set the taxes of the 'high-tax' scenario
tau_high = 0.4
# Set time of the change
t = 0

# Qmod class

# Copy the initial model, set a higher psi and re-solve
Q_high_tau = deepcopy(Qmodel)
Q_high_tau.tau = tau_high
Q_high_tau.solve()

# Capital will start at it steady state in the
# high-tax scenario
k0 = Q_high_tau.kss

sol = structural_change(mod1 = Q_high_tau, mod2 = Qmodel,
                        k0 = k0, t_change = t,T_sim=T)

# Dolo
soldolo = simul_change_dolo(model = QDolo, k0 = np.array([k0]),
                            exog0 = [R,tau_high,zeta,psi],
                            exog1 = [R,tau,zeta,psi],
                            t_change = t, T_sim = T)

# Plot the path of capital under both solutions
time = range(T)
plt.figure()
plt.plot(time, sol['k'], 'x', label = 'Qmod', alpha = 0.8)
plt.plot(time, soldolo['k'], '+', label = 'Dolo', alpha = 0.8)
plt.legend()
plt.title('Capital dynamics')
plt.ylabel('$k_t$ : capital')
plt.xlabel('$t$ : time')
# %% [markdown]
# ## 4. A corporate tax cut announced at t=0 but taking effect at t=5
# %% {"code_folding": [0]}
# Modify the time of the change
t = 5

# Qmod class
sol = structural_change(mod1 = Q_high_tau, mod2 = Qmodel,
                        k0 = k0, t_change = t,T_sim=T)

# Dolo
soldolo = simul_change_dolo(model = QDolo, k0 = np.array([k0]),
                            exog0 = [R,tau_high,zeta,psi],
                            exog1 = [R,tau,zeta,psi],
                            t_change = t, T_sim = T)

# Plot the path of capital under both solutions
time = range(T)
plt.figure()
plt.plot(time, sol['k'], 'x', label = 'Qmod', alpha = 0.8)
plt.plot(time, soldolo['k'], '+', label = 'Dolo', alpha = 0.8)
plt.legend()
plt.title('Capital dynamics')
plt.ylabel('$k_t$ : capital')
plt.xlabel('$t$ : time')
# %% [markdown]
# ## 5. An unanticipated ITC increase
# %% {"code_folding": [0]}
# Set time of the change
t=0
# Set investment tax credit in the high case
itc_high = 0.2
# Set initial value of capital
k0 = Qmodel.kss

# Qmod class

# Copy the initial model, set a higher psi and re-solve
Q_high_itc = deepcopy(Qmodel)
Q_high_itc.zeta = itc_high
Q_high_itc.solve()

sol = structural_change(mod1 = Qmodel, mod2 = Q_high_itc,
                        k0 = k0, t_change = t,T_sim=T)

# Dolo
soldolo = simul_change_dolo(model = QDolo, k0 = np.array([k0]),
                            exog0 = [R,tau,zeta,psi],
                            exog1 = [R,tau,itc_high,psi],
                            t_change = t, T_sim = T)

# Plot the path of capital under both solutions
time = range(T)
plt.figure()
plt.plot(time, sol['k'], 'x', label = 'Qmod', alpha = 0.8)
plt.plot(time, soldolo['k'], '+', label = 'Dolo', alpha = 0.8)
plt.legend()
plt.title('Capital dynamics')
plt.ylabel('$k_t$ : capital')
plt.xlabel('$t$ : time')
# %% [markdown]
# ## 6. An ITC increase announced at t=0 but taking effect at t=5
# %% {"code_folding": [0]}
# Modify time of the change
t = 5

# Qmod class
sol = structural_change(mod1 = Qmodel, mod2 = Q_high_itc,
                        k0 = k0, t_change = t,T_sim=T)

# Dolo
soldolo = simul_change_dolo(model = QDolo, k0 = np.array([k0]),
                            exog0 = [R,tau,zeta,psi],
                            exog1 = [R,tau,itc_high,psi],
                            t_change = t+1, T_sim = T)

# Plot the path of capital under both solutions
time = range(T)
plt.figure()
plt.plot(time, sol['k'], 'x', label = 'Qmod', alpha = 0.8)
plt.plot(time, soldolo['k'], '+', label = 'Dolo', alpha = 0.8)
plt.legend()
plt.title('Capital dynamics')
plt.ylabel('$k_t$ : capital')
plt.xlabel('$t$ : time')
