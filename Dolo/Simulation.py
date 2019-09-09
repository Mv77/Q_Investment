# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.2'
#       jupytext_version: 1.2.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
# Setup
import matplotlib as plt
import numpy as np
from dolo import *
import dolo.algos.perfect_foresight as pf
import dolo.algos.value_iteration as vi
import pandas as pd

# %%
# Load and calibrate the model model
model = yaml_import("Q_model.yaml")

alpha = 0.33
delta = 0.05
omega = 2

model.set_calibration(alpha = alpha, delta = delta, omega = omega)

# %%
# Interest rate simulation

# Create empty dataframe for exog. variables
exog = pd.DataFrame(columns = ['R','tau'])

# Generate an interest rate process
exog.R = np.concatenate((np.repeat(1.03,50),
                    np.repeat(1.05,10),
                    np.repeat(1.01,20)))

# Leave tau at 0
exog.tau = 0

# Simpulate the optimal response
dr = pf.deterministic_solve(model = model,shocks = exog,verbose=True)

# Plot the results
ex = 'R'
vars = ['k','i']

for var in vars:
    
    fig, ax1 = plt.pyplot.subplots()

    color = 'tab:red'
    ax1.set_xlabel('Time')
    ax1.set_ylabel(ex, color=color)
    ax1.plot(dr[ex], color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:blue'
    ax2.set_ylabel(var, color=color)  # we already handled the x-label with ax1
    ax2.plot(dr[var], color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.pyplot.grid(axis='both')
    plt.pyplot.show()

# %%
# Interest rate simulation

# Create empty dataframe for exog. variables
exog = pd.DataFrame(columns = ['R','tau'])

# Generate a future tax cut dynamic
exog.tau = np.concatenate((np.repeat(0.2,50),
                           np.repeat(0,50)))

# Leave R at 1.02
exog.R = 1.02

# Simpulate the optimal response
dr = pf.deterministic_solve(model = model,shocks = exog,verbose=True)

# Plot the results
ex = 'tau'
vars = ['k','i']

for var in vars:
    
    fig, ax1 = plt.pyplot.subplots()

    color = 'tab:red'
    ax1.set_xlabel('Time')
    ax1.set_ylabel(ex, color=color)
    ax1.plot(dr[ex], color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:blue'
    ax2.set_ylabel(var, color=color)  # we already handled the x-label with ax1
    ax2.plot(dr[var], color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.pyplot.show()
