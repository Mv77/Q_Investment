# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.2'
#       jupytext_version: 1.2.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # YOUR NAME
#
# This notebook is a template showing you how to use some tools that will be useful for problem set 5.
#
# The tools are:
# 1. Qmod: an implementation of the q-Model as a class in python. This is used to obtain phase diagrams.
# 2. Dolo: a general purpose tool used to represent and solve economic models. This is used to compute optimal responses.
#
# The use of both tools will make us have to handle two representations of our problems, one as a Qmod object, and one as a Dolo model.
#
# Start by loading the tools we will be using.

# %% {"code_folding": []}
# Preamble: import the packages we will be using

# Usual packages
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from dolo import *
import dolo.algos.perfect_foresight as pf
import pandas as pd

# Import the Qmod python code from external file

# Since the Qmod class is in other folder we need to
# change the path.
import sys
sys.path.append('../')
from Qmod.Q_investment import Qmod

# %% [markdown]
# I then define a function that takes two Qmod objects and plots their phase diagrams in the same figure.

# %% {"code_folding": []}
def phase_diagrams(mod1,mod2,k_min,k_max,npoints = 300):
    """
    Draws the phase diagram of the Qmodel under two different sets of
    parameter values in the same figure, and returns it.

    Parameters:
        - mod1          : Qmod object representing the first set of parameter values.
        - mod1          : Qmod object representing the second set of parameter values.
        - [k_min,k_max] : limits for the value of capital in the phase diagrams.
        - npoints       : number of points in the capital grid to be used for phase
                          diagram plots.
    """

    # Create a figure
    fig, ax = plt.subplots()

    # Plot the loci of the pre and post-change models.
    k_range = np.linspace(k_min,k_max,npoints)
    mods = [mod1,mod2]
    colors = ['r','b']
    labels = ['Mod. 1','Mod. 2']
    for i in range(2):

        # Plot k0 locus
        ax.plot(k_range,mods[i].P*np.ones(npoints),
                 linestyle = '--', color = colors[i],label = labels[i] + ' loci')
        # Plot lambda0 locus
        ax.plot(k_range,[mods[i].lambda0locus(x) for x in k_range],
                 linestyle = '--', color = colors[i])
        # Plot steady state
        ax.plot(mods[i].kss,mods[i].P,marker = '*', color = colors[i])
        
        # Plot stable arm
        stab_arm = [mods[i].findLambda(k0 = x, k1 = mods[i].k1Func(x)) for x in k_range]
        ax.plot(k_range, stab_arm, linestyle = '-', color = colors[i], label = labels[i] + ' stable arm.')

    return(ax)
# %% [markdown]
# Now I create a base model parametrization.

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

# %% [markdown]
# And create the objects representing the model. First in Qmod:

# %%
## Qmod python class

# Create the object.
Qmodel = Qmod(beta, tau, alpha, omega, zeta, delta, psi)
# Solve to find the steady state and policy rule.
Qmodel.solve()

# %% [markdown]
# Then in Dolo.
#
# There is an important difference. The implementation in Dolo treats the interest rate, technological factor, tax rate, and investment tax credit as not parameters but exogenous variables subject to change.

# %%
## Dolo

# First import the external file with the model description.
QDolo = yaml_import("../Dolo/Q_model.yaml")

# Then replace the default parameters with our desired ones.
QDolo.set_calibration(alpha = alpha, delta = delta, omega = omega)

#############
# IMPORTANT #
#############

# We do not pass R, psi, tau, or itc because they are handled not as parameters
# but exogenous variables that can change over time. (see below)

# %% [markdown]
# ## Dynamic response example: changes in productivity.
#
# I now give an example of how to use Dolo to solve for the optimal behavior of the firm.
# The basic idea is:
# 1. We will create a path for the exogenous variables of the model.
# 2. Dolo assumes that this path is announced to the firm at the first time period, and it computes the optimal dynamics according to these paths.
# %%
# Set up the basic features of the simulation

# Total simulation time
T = 100
# Initial level of capital
k0 = Qmodel.kss

# %% {"code_folding": []}
# Design the shock:

# for this example, I am assuming that productivity increases and then
# goes back to the original level.

psi_high = 1.2
Psi_sequence = np.array([psi]*3+
                        [psi_high]*50+
                        [psi]*(T-3-50))
# Check the pattern of the shock
plt.figure()
plt.plot(Psi_sequence)
plt.ylabel('$\psi$')

# Dolo receives a DataFrame with the full future paths for ALL exogenous
# variables. So we create one:
Exog = pd.DataFrame({'R':[R]*T,
                     'tau':[tau]*T,
                     'itc_1':[zeta]*T,
                     'psi':Psi_sequence})

# Examine the first few entries.
Exog.head()
# Note all other variables are left constant.

# %%
# Now use the "perfect foresight" dolo solver
response = pf.deterministic_solve(model = QDolo, # Model we are using (in dolo)
                                  shocks = Exog, # Paths for exog. variables 
                                  T=T,           # Total simulation time
                                  s1 = [k0],     # Initial state
                                  verbose=True)

# Response is a DataFrame with the paths of every variable over time.
# It adds information we don't need on the first row. So we delete it
response = response[1:]

# Inspect the first few elements.
response.head()

# %% [markdown]
# # IMPORTANT
#
# Because of the way the model is implemented, it does not keep track of $\lambda_t$ or $ITC_t$, but  $\lambda_{t+1}$ and $ITC_{t+1}$ instead. 
#
# Thus, lambda_1 in row 1 corresponds to $\lambda_2$. Same with the ITC.

# %% [markdown]
# # Plots
#
# Now we can use Qmod to plot the phase diagrams and add the optimal dynamics that we just found.

# %%
# Draw the two phase diagrams and save them in an object
# to add the plots of dynamics later.

# Copy the initial model, set the higher psi and re-solve
Q_high_psi = deepcopy(Qmodel)
Q_high_psi.psi = psi_high
Q_high_psi.solve()

# Now we draw the phase diagrams of our base model "Qmodel"
# and the new one "Q_high_psi", and store the plot in
# object "ax"
ax = phase_diagrams(mod1 = Qmodel, mod2 = Q_high_psi, k_min = 2, k_max = 8)

# Now we can add the behavior of lambda and k to the diagram.
ax.plot(response.k, response.lambda_1, '.k',label = 'Opt. Response')
plt.legend()
plt.xlabel('k')
plt.ylabel('$\\lambda$')

# New figure for capital dynamics
plt.figure()
plt.plot(response.k,'.')
plt.ylabel('$k$')
plt.xlabel('Time')
