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
# # Examples of structural change in the Abel-Hayashi "Q" investment model
#
# This notebook illustrates the dynamic behavior of capital and its marginal
# value in the Abel-Hayashi model of investment when structural changes happen.
#
# I simulate the changes discussed in Prof. Christopher D. Carroll's graduate
# Macroeconomics [lecture notes](http://www.econ2.jhu.edu/people/ccarroll/public/lecturenotes/Investment/qModel/):
# productivity, corporate tax rate, and investment tax credit changes.
#
# For each change I display the behavior of the model in two different
# contexts:
# * The change takes place at $t=0$ without notice.
# * The change is announced at $t=0$ but takes place at $t=5$.

# %% {"code_folding": []}
# Preamble
import numpy as np
import matplotlib.pyplot as plt

from copy import deepcopy

from dolo import *
import dolo.algos.perfect_foresight as pf
import dolo.algos.value_iteration as vi

import pandas as pd

# Since the Qmod class is in other folder we need to
# change the path.
import sys
sys.path.append('../')
from Qmod.Q_investment import Qmod

# %% [markdown]
# I first import a function that computes and presents optimal dynamics in face of
# structural changes in the Qmod implementation.
# %% {"code_folding": [0]}
from Qmod.Q_investment import structural_change


# %% [markdown]
# I now define a function to handle parameter changes in the Dolo implementation

# %% {"code_folding": [0]}
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
# ## Examples:
#
# ## 1. An unanticipated increase in productivity
# %% {"code_folding": [0]}
# Total simulation time
T = 20
# Time the change occurs
t = 0
# Initial level of capital
k0 = Qmodel.kss

# Productivity in the "new" state
psi_new = 1.3

## Qmod class

# Copy the initial model, set the higher psi and re-solve
Q_high_psi = deepcopy(Qmodel)
Q_high_psi.psi = psi_new
Q_high_psi.solve()

sol = structural_change(mod1 = Qmodel, mod2 = Q_high_psi,
                        k0 = k0, t_change = t,T_sim=T)

## Dolo

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
