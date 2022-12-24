# Mattia Fussi - 562860
# Intelligent Systems for Pattern Recognition
# 
# Midterm 2 - Assignment 1

import numpy as np
from hmmlearn import hmm
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.dates as mdates
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def plot_seq(seq, samples, length, Ymin = 0, Ymax = 10, title = '', ylab1 = '', ylab2 = ''):
    
    fig = plt.figure()
    plt.plot(length, samples, color = 'tab:blue', label = 'True data', linewidth = 0.7)
    plt.plot(length, seq, color = 'tab:orange', label = 'Sampled data', linewidth = 0.7)
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.xticks(rotation = 30)
    plt.ylabel(ylab1)

    return fig

def plot_hs(seq, samp, timestamp, title = '', ylab = ''):
    
    colors = []

    for i in range(len(seq)):
        if seq[i] == 0:
            colors.append('tab:blue')
        elif seq[i] == 1:
              colors.append('tab:green')
        elif seq[i] == 2:
             colors.append('tab:orange')
        elif seq[i] == 3:
            colors.append('tab:red')
        elif seq[i] == 4:
            colors.append('tab:purple')
        else:
            colors.append('tab:pink')
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.grid()
    plt.title(title)
    ax.set_xticklabels(labels = timestamp, rotation = 45)
    ax.xaxis_date()
    ax.set_ylabel(ylab)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m'))
    ax.axis(xmin = '2016-04-12' ,xmax =  '2016-05-26')
    ax.scatter(timestamp, samp, color = colors, s = 1)

    return fig
    
df = pd.read_csv("~/Documenti/unipi/AA2/S2/ispr/mid2/energydata_complete.csv", header = 0).set_index('date')
val_len = len(df.loc["2016-04-12 00:00:00":"2016-05-26 23:50:00", "Appliances"])

# Appliances dataset
A = []

t_appl = df.loc["2016-01-12 00:00:00":"2016-04-11 23:50:00", "Appliances"]
v_appl = df.loc["2016-04-12 00:00:00":"2016-05-26 23:50:00", "Appliances"]

A_train = np.reshape(list(t_appl), (-1,1))
A_val = np.reshape(list(v_appl), (-1,1))

# Instantiates models and fit with different numbers of hidden states
print('Fitting HMM on Appliances data...')
model_a2 = hmm.GaussianHMM(n_components = 2, covariance_type = "full", n_iter = 1000, tol = 0.0001).fit(A_train)
model_a3 = hmm.GaussianHMM(n_components = 3, covariance_type = "full", n_iter = 1000, tol = 0.0001).fit(A_train)
model_a4 = hmm.GaussianHMM(n_components = 4, covariance_type = "full", n_iter = 1000, tol = 0.0001).fit(A_train)

# Predict sequence of hidden states with Viterbi for first month of data
print('Predicting hidden states on Appliances...')
hstates_a2 = model_a2.predict(A_val)
hstates_a3 = model_a3.predict(A_val)
hstates_a4 = model_a4.predict(A_val)

# Lights dataset
L = []

t_lig =  df.loc["2016-01-12 00:00:00":"2016-04-11 23:50:00", 'lights']
v_lig =  df.loc["2016-04-12 00:00:00":"2016-05-26 23:50:00", 'lights']

L_train = np.reshape(list(t_lig), (-1,1))
L_val = np.reshape(list(v_lig), (-1,1))

print('Fitting HMM on Lights data...')
model_l2 = hmm.GaussianHMM(n_components = 2, covariance_type = "full", n_iter = 1000, tol = 0.0001).fit(L_train)
model_l3 = hmm.GaussianHMM(n_components = 3, covariance_type = "full", n_iter = 1000, tol = 0.0001).fit(L_train)
model_l4 = hmm.GaussianHMM(n_components = 4, covariance_type = "full", n_iter = 1000, tol = 0.0001).fit(L_train)

print('Predicting hidden states on Lights...')
hstates_l2 = model_l2.predict(L_val)
hstates_l3 = model_l3.predict(L_val)
hstates_l4 = model_l4.predict(L_val)

# ------------------------------------------------------------------
# Visualization
times = pd.date_range('2016-04-12', periods = val_len, freq = '10min')
    
fig = plot_hs(hstates_a2, v_appl, times, title = '2HS most likely prediction', ylab = 'Appliances')
fig.savefig('hs_a2.png', dpi = 200)
fig = plot_hs(hstates_a3, v_appl, times, title = '3HS most likely prediction', ylab = 'Appliances')
fig.savefig('hs_a3.png', dpi = 200)
fig = plot_hs(hstates_a4, v_appl, times, title = '4HS most likely prediction', ylab = 'Appliances')
fig.savefig('hs_a4.png', dpi = 200)

fig = plot_hs(hstates_l2, v_lig, times, title = '2HS most likely prediction', ylab = 'Lights')
fig.savefig('hs_l2.png', dpi = 200)
fig = plot_hs(hstates_l3, v_lig, times, title = '3HS most likely prediction', ylab = 'Lights')
fig.savefig('hs_l3.png', dpi = 200)
fig = plot_hs(hstates_l4, v_lig, times, title = '4HS most likely prediction', ylab = 'Lights')
fig.savefig('hs_l4.png', dpi = 200)    

# ------------------------------------------------------------------
# Sample from the fitted distributions
n = 24 * 6 * 1 # 24h * 6obs * 1d
times = pd.date_range('2016-04-12', periods = n, freq = '10min')

# Appliances
X_a2, Z_a2 = model_a2.sample(n)
X_a3, Z_a3 = model_a3.sample(n)
X_a4, Z_a4 = model_a4.sample(n)

fig = plot_seq(X_a2, A_val[:n], times, Ymin = -10, Ymax = 800, title = 'Sampled data (2HS) vs True data', ylab1 = 'Appliances', ylab2 = 'Sampled data')
fig.savefig('smpl_a2.png', dpi = 200)
fig = plot_seq(X_a3, A_val[:n], times, Ymin = -10, Ymax = 800, title = 'Sampled data (3HS) vs True data', ylab1 = 'Appliances', ylab2 = 'Sampled data')
fig.savefig('smpl_a3.png', dpi = 200)
fig = plot_seq(X_a4, A_val[:n], times, Ymin = -10, Ymax = 800, title = 'Sampled data (4HS) vs True data', ylab1 = 'Appliances', ylab2 = 'Sampled data')
fig.savefig('smpl_a4.png', dpi = 200)

# Lights
X_l2, Z_l2 = model_l2.sample(n)
X_l3, Z_l3 = model_l3.sample(n)
X_l4, Z_l4 = model_l4.sample(n)

fig = plot_seq(X_l2, L_val[:n], times, Ymin = -10, Ymax = 80, title = 'Sampled data (2HS) vs True data', ylab1 = 'Lights', ylab2 = 'Sampled data')
fig.savefig('smpl_l2.png', dpi = 200)
fig = plot_seq(X_l3, L_val[:n], times, Ymin = -10, Ymax = 80, title = 'Sampled data (3HS) vs True data', ylab1 = 'Lights', ylab2 = 'Sampled data')
fig.savefig('smpl_l3.png', dpi = 200)
fig = plot_seq(X_l4, L_val[:n], times, Ymin = -10, Ymax = 80, title = 'Sampled data (4HS) vs True data', ylab1 = 'Lights', ylab2 = 'Sampled data')
fig.savefig('smpl_l4.png', dpi = 200)