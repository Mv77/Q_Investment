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
# # Examples of using the Qmod class.
#
# ## [Mateo Velásquez-Giraldo](https://github.com/Mv77)

# %% {"code_folding": []}
# Preamble
import numpy as np
import matplotlib.pyplot as plt

from Q_investment import Qmod
# %% [markdown]
# # Examples

# %% [markdown]
# ## 1. Model solution and policy rule.

# %%
# Create model object
Qexample = Qmod()
# Solve to find the policy rule (k[t+1] in terms of k[t])
Qexample.solve()

# Plot policy rule
k = np.linspace(1,3*Qexample.kss,20)

plt.figure()
plt.plot(k,[Qexample.k1Func(x) for x in k], label = "Optimal capital")
plt.plot(k,k, linestyle = '--', color = 'k', label = "45° line")
plt.plot(Qexample.kss,Qexample.kss,'*r', label = "Steady state")
plt.title('Policy Rule')
plt.xlabel('k(t)')
plt.ylabel('k(t+1)')
plt.legend()
plt.show()
# %% [markdown]
# ## 2. Simulation of capital dynamics.

# %%
# Find capital dynamics from a given starting capital
k0 = 2*Qexample.kss
t = 50

# Simulate capital trajectory
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
# ## 3. The impact of ajdustment costs

# %%
# Create and solve two instances, one with high and one with low adjustment costs omega
Qlow  = Qmod(omega =  0.1)
Qhigh = Qmod(omega =  0.9)

Qlow.solve()
Qhigh.solve()

# Simulate adjustment from an initial capital level
k0 = 2*Qhigh.kss
t = 50
k_low = Qlow.simulate(k0,t)
k_high = Qhigh.simulate(k0,t)

# Plot
plt.figure()
plt.plot(k_low, label = 'Low $\\omega$')
plt.plot(k_high, label = 'High $\\omega$')
plt.axhline(y = Qhigh.kss,linestyle = '--',color = 'k', label = 'Steady state ${k}$')
plt.title('Capital')
plt.xlabel('Time')
plt.legend()
plt.show()
# %% [markdown]
# ## 4. Phase diagram.

# %%
# Create and solve model object
Qexample = Qmod()
Qexample.solve()
# Generate its phase diagram
Qexample.phase_diagram(stableArm = True)

# %% [markdown]
# Why is the $\dot{\lambda}=0$ locus truncated?
#
# With constant prices, there may be instances where $\lambda_t$ can not be equal to $\lambda_{t+1}$. Notice first that $\lambda_t$ is a function of $\lambda_{t+1}$ (current marginal value of capital is a function of its expected marginal value tomorrow).
#
# If, for instance, $k_t$ is low, the marginal productivity of capital will be high, and this can push $\lambda_t$ above $\lambda_{t+1}$, as is the case in the following diagram, which plots $\lambda_t$ computed from the envelope condition at a fixed $k$ and varying $\lambda_{t+1}$.

# %%
Qexample.plotEnvelopeCond(k=2)

# %% [markdown]
# Note that the envelope condition never crosses the $\lambda_t = \lambda_{t+1}$ line. Thus, there is no $\dot{\lambda}=0$ locus at $k=2$.
