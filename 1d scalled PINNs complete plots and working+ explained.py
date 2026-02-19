# -*- coding: utf-8 -*-
"""
Created on Sun Feb  1 02:26:33 2026

@author: Home
"""

import os
os.environ["DDE_BACKEND"] = "tensorflow"

#############time reversal 1d put pinns for paper1
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import deepxde as dde
from deepxde.backend import tf
import scipy.stats as si
import pandas as pd

def exact_solution(x, tao):
    """
    Returns the exact solution for a given x and t .

    Parameters
    ----------
    x : np.ndarray
    t : np.ndarray
    """
    d1 = ((np.log(x / K) + (r + 0.5 * sigma ** 2)
          * (tao))) / (sigma * np.sqrt(tao))
    d2 = (np.log(x / K) + (r - 0.5 * sigma ** 2)
          * (tao)) / (sigma * np.sqrt(tao))

    put = -x * si.norm.cdf(-d1, 0.0, 1.0) + K*np.exp(-r*(tao)) * si.norm.cdf(-d2, 0.0, 1.0)
    # print (t,call)
    return put



def main():
    
    def pde(x, y):
        """
        #BSPDE forward in time. ie tau is time to expiry
        Here x is a 2d array with x1 being scalled stock price and x2 being tau
        """
        S1, tau = x[:, 0:1], x[:, 1:2]

        V_S1 = dde.grad.jacobian(y, x, i=0, j=0)
        V_t  = dde.grad.jacobian(y, x, i=0, j=1)

        V_S1S1 = dde.grad.hessian(y, x, i=0, j=0)
        diffusion = 0.5 * (
            sigma**2 * S1**2 * V_S1S1
        )

        drift = r * (S1 * V_S1)
        rem= diffusion + drift - r * y # this is imp for deepxde syntax

        return V_t - rem


    def func(x):
        return np.maximum(K-x[:, 0:1], 0)
    """
    # smooth payoff was tried but for 1d case this did not make much improvement
    #because the error introduced by approximation was greater than the improvement.
    # verdict: dont use soft max
    def smooth_put(x):
        eps=0.01
        return 0.5 * (
            (K - x[:, 0:1]) +
            tf.sqrt((K - x[:, 0:1])**2 + eps**2)
        )
    def smooth_2(x):
        beta=50.0
        S=x[:,0:1]
        payoff=(1.0/beta)*np.log(beta*(K-S))
        return payoff
"""
    ###
    geom = dde.geometry.Interval(0.0, L)
    timedomain = dde.geometry.TimeDomain(0, T)
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)

    #BOUNDRIES
    bc_S0 = dde.DirichletBC(geomtime,lambda x: K * np.exp(-r * x[:, 1:2]),
        lambda x, on_boundary: on_boundary and np.isclose(x[0], 0.0))
    
    bc_SL = dde.DirichletBC(geomtime,lambda x: np.zeros((len(x), 1)),
        lambda x, on_boundary: on_boundary and np.isclose(x[0], L))
    
    ic = dde.IC( geomtime, func, lambda _, on_initial: on_initial)

    bcs=[ic,bc_S0,bc_SL]

    # Define the PDE problem and configurations of the network:
    data = dde.data.TimePDE(geomtime, pde, bcs, 
        num_domain=2000*col,
        num_boundary=200*col, 
        num_initial=100*col )

    net = dde.nn.FNN([2] + [30] * 3 + [1], "tanh", "Glorot normal")
    net.apply_output_transform(lambda x,y:tf.nn.softplus(y)) # to ensure positive output
    model = dde.Model(data, net)

    # Build and train the model:
    lenbc=len(bcs)-1
    model.compile("adam", lr=0.0001,loss_weights =[5,1]+[1]*lenbc)
    model.train(epochs=5000)
    model.compile("L-BFGS",loss_weights =[5,1]+[1]*lenbc)
    losshistory, train_state = model.train()

    # Plot/print the results
    dde.saveplot(losshistory, train_state, issave=True, isplot=True)

    #############comparison

    # COMPARING WITH EXACT SOLUTION  FOR MODEL EVALUATION

    N=1000
    timedomain1 = dde.geometry.TimeDomain(0, 1)
    geom = dde.geometry.Interval(x_min,x_max)
    geomtime1 = dde.geometry.GeometryXTime(geom, timedomain1)
    X=geomtime1.random_points(N)
    y_true = np.zeros((N))
    y_pred=np.zeros((N))

    # Obtain the value of the exact solution/MODEL PREDICTION for each generated point:
    for i in range(N):
        y_true[i] = exact_solution(X[i][0],X[i][1])
        y_pred[i]=model.predict([(X[i][0],X[i][1])])

    #######################################
    error = (y_true-y_pred)
    max_err_num = np.max(np.abs(error))
    print("Max error#####:=", max_err_num)

    plt.xlabel('(MC-PINN)/Strike')
    plt.ylabel('density')
    plt.title('error distribution')
    sns.histplot(error, kde=False, bins=50)
    plt.show()

    ###########################
    #droping points with exact solution zero caause that would creat error
    thresh=0.0 #1% of strike
    nonzero_indices = y_true>thresh
    num_kept=np.sum(nonzero_indices)
    print("kept points=",num_kept)
    
    # Filter y_true and y_pred using these indices
    y_true_filtered = y_true[nonzero_indices]
    y_pred_filtered = y_pred[nonzero_indices]


    #model parameters/ evaluation matrices   
    mse =dde.metrics.mean_squared_error(y_true_filtered, y_pred_filtered)
    print("MSE:", mse)
    print("RMSE=",np.sqrt(mse))
    e=dde.metrics.mean_absolute_percentage_error(y_true_filtered, y_pred_filtered)
    print('mean_absolute_percentage_error=',e)
    a=dde.metrics.absolute_percentage_error_std(y_true_filtered, y_pred_filtered)
    print("metrics.absolute_percentage_error_std=",a)
    c=dde.metrics.l2_relative_error(y_true_filtered, y_pred_filtered)
    print('l2_relative_error=',c)
    f=dde.metrics.nanl2_relative_error(y_true_filtered, y_pred_filtered)
    print('nanl2_relative_error=',f)
    from sklearn.metrics import r2_score
    print("r2=",r2_score(y_true_filtered, y_pred_filtered))


################### PLOTS at tau=1 #############################
    # Number of points in X-DIM:
    x_dim= 1000

    x = np.linspace(x_min, x_max, num=x_dim).reshape(x_dim, 1)
  
    v=np.zeros((x_dim))
    for i in range(x_dim):
        v[i]=exact_solution(x[i], 1)
        
    results = np.zeros((x_dim))
    
    x_test_new = np.column_stack((x, np.ones(x_dim)))
    results= model.predict(x_test_new)
        
    
    plt.xlabel('stock price')
    plt.ylabel('option value')
    plt.plot(x*K1,v*K1,label='exact solution')
    plt.plot(x*K1,results*K1, label='models prediction')
    plt.legend()
    plt.show()
    
    
    ##################################
    err=np.zeros((x_dim))
    for i in range (x_dim):
      err[i]=v[i]-results[i]
    print(np.shape(err))
    max_err = np.max(np.abs(err))
    print("Max pointwise error:", max_err*K1)

    
    plt.xlabel('stock')
    plt.ylabel('error')
    plt.title('pointwise error error')
    plt.plot(x*K1,err*K1,linestyle='--')
    plt.show()
    ##################################
    
    # ============================================================
    # 2D diagnostic: error at-the-money (S = K) over time
    # ============================================================
    n_t = 300
    tau_vals = np.linspace(0.1, T, n_t).reshape(-1, 1)
    S_vals = K * np.ones_like(tau_vals)
    
    X_atm = np.hstack((S_vals, tau_vals))
    
    # PINN prediction
    pinn_vals = model.predict(X_atm).ravel()
    
    # Exact solution (FIXED)
    exact_vals = exact_solution(
        S_vals,
        tau_vals
    ).ravel()
    
    # Errors
    abs_err = exact_vals - pinn_vals
    rel_err = abs_err / (exact_vals + 1e-8)
    
    # Plot
    plt.figure(figsize=(7, 4))
    plt.plot(tau_vals, abs_err * K1, lw=2)
    plt.xlabel("Time to maturity $\\tau$")
    plt.ylabel("error")
    plt.title("error of scalled PINNs at-the-money (S = K)")
    plt.show()
    
    plt.figure(figsize=(7, 4))
    plt.plot(tau_vals, rel_err * K1, lw=2)
    plt.xlabel("Time to maturity $\\tau$")
    plt.ylabel("relative error")
    plt.title("error of scalled PINNs at-the-money (S = K)")
    plt.show()

    
   
    
 



if __name__ == "__main__":

    # Problem parameters:
    sigma = 0.3
    r = 0.03
    K = 1
    K1=4 #scalling factor
    T=1
    #MODEL PARAMETERS FOR nn
    L = 2.5*K # max domain
    col=5 
    
    #Parameters for evaluation of model
    x_min, t_min = (0.01, 0.) 
    x_max, t_max = (2*K, T)
    

    main()
