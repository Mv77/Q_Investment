# -*- coding: utf-8 -*-
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

# %%
# Setup
import matplotlib as plt
import numpy as np
from dolo import *
import dolo.algos.perfect_foresight as pf
import dolo.algos.value_iteration as vi
import pandas as pd

# Define a function to handle plots
def plotQmodel(model, exog, returnDF = False):
    
    # Simpulate the optimal response
    dr = pf.deterministic_solve(model = model,shocks = exog,verbose=True)
    
    # Plot exogenous variables
    fig, axes = plt.pyplot.subplots(1,3, figsize = (10,3))
    axes = axes.flatten()
    ex = ['R','tau','itc_1']
    
    for i in range(len(ex)):
        ax = axes[i]
        ax.plot(dr[ex[i]],'.')
        ax.set_xlabel('Time')
        ax.set_ylabel(ex[i])
    fig.tight_layout()
    
    # Plot optimal response variables
    fig, axes = plt.pyplot.subplots(2,2, figsize = (10,6))
    axes = axes.flatten()
    opt = ['k','i','lambda_1','q_1']
    
    for i in range(len(opt)):
        ax = axes[i]
        ax.plot(dr[opt[i]],'.')
        ax.set_xlabel('Time')
        ax.set_ylabel(opt[i])
    fig.tight_layout()
    
    if returnDF:
        return(dr)


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
exog = pd.DataFrame(columns = ['R','tau','itc_1'])

# Generate an interest rate process
exog.R = np.concatenate((np.repeat(1.03,50),
                    np.repeat(1.05,10),
                    np.repeat(1.01,20)))

# Leave tau at 0
exog.tau = 0
# Leave itc at 0
exog.itc_1 = 0

# Solve for the optimal response and plot the results  
plotQmodel(model,exog)

# %%
# Tax rate simulation

# Create empty dataframe for exog. variables
exog = pd.DataFrame(columns = ['R','tau','itc_1'])

# Generate a future tax cut dynamic
exog.tau = np.concatenate((np.repeat(0.2,50),
                           np.repeat(0,50)))

# Leave R at 1.02
exog.R = 1.02
# Leave itc at 0
exog.itc_1 = 0

# Solve for the optimal response and plot the results  
plotQmodel(model,exog)

# %%
# ITC simulation

# Create empty dataframe for exog. variables
exog = pd.DataFrame(columns = ['R','tau','itc_1'])

# Generate a future itc increase dynamic
exog.itc_1 = np.concatenate((np.repeat(0,50),
                           np.repeat(0.25,50)))

# Leave R at 1.02
exog.R = 1.02
# Leave tau at 0
exog.tau = 0

# Solve for the optimal response and plot the results  
plotQmodel(model,exog)

