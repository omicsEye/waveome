"""
Script to run a simulation looking at kernel recovery sliding between number 
of observations versus number of units.
"""

# Libraries
import gpflow
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import os 
import sys
import tensorflow as tf
from helper_functions import *
import time
import pickle
import itertools

# Options
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Functions
def sim_data(rate=12, num_units=30, fixed_num=False,
             include_output=False, kern_out=None,
             eps=0, set_seed=True):
    # Set random seed
    if set_seed:
        np.random.seed(9102)

    # Assign treatment group to each unit
    # treat_group = np.repeat([0,1], num_units/2)
    treat_group = np.random.binomial(n=1, p=0.5, size=num_units)

    # Sample number of observations for each unit
    if fixed_num:
        num_obs = np.repeat(rate, num_units)
    else:
        num_obs = np.random.poisson(lam=rate, size=num_units)

    # Sample from uniform distribution for observation times
    x = np.concatenate(
        [np.sort(np.round(np.random.uniform(low=0, high=12, size=i), 1)) for i in num_obs],
        axis=0
    )

    # Put unit ID and observation time together
    df = np.array([np.repeat(np.arange(num_units), num_obs),
                   np.repeat(treat_group, num_obs),
                   x]).T

    df = pd.DataFrame(
        df,
        columns=['id', 'treat', 'time']
    )

    # Standardize continuous variable
    df.time = (df.time - 6) / np.sqrt((1/12.)*(12**2))

    if include_output and kern_out is not None:

        for k in kern_out.keys():
            # Simulate output
            f_ = np.random.multivariate_normal(
                mean=np.zeros_like(df.iloc[:, 0]).flatten(),
                cov=kern_out[k]['model'](df) + 1e-6 * np.eye(df.shape[0]),
                size=1
            )[0]

            # Add noise
            df[k] = f_ + np.random.normal(loc=0, scale=np.sqrt(eps), size=len(f_))

    return df

def run_simulation(rate, epsilon, units, iters, kernel_list):
    
    print(rate, epsilon, units, iters)
    
    # Prep output dataset
    sim_out = pd.DataFrame()

    # Build temporary simulated data
    temp_df = sim_data(
        rate=rate,
        num_units=units,
        fixed_num=True,
        include_output=True,
        kern_out=kernel_dictionary,
        set_seed=False,
        eps=epsilon
    )

    # Run kernel search process
    for i in range(4):
        search_out = full_kernel_search(
            X=temp_df[['id', 'treat', 'time']],
            Y=temp_df.drop(columns=['id', 'treat', 'time']).iloc[:, i],
            kern_list=kernel_list,
            cat_vars=[0, 1],
            random_seed=9102
        )

        # Save resulting kernels and information
        cur_out = pd.DataFrame({
            'rate': [rate], #4*[r],
            'eps': [epsilon], #4*[eps],
            'units': [units], #4*[u],
            'iter': [iters], #4*[i],
            'output': ['y'+str(i+1)], #['y1', 'y2', 'y3', 'y4'],
            'kernel': [search_out['best_model']], #[x['best_model'] for x in search_out],
            'model': [search_out['models'][search_out['best_model']]]
            #[x['models'][x['best_model']] for x in search_out]
        })

        sim_out = pd.concat([sim_out, cur_out])
        # print(f"Finished rate {r} with {u} units and epsilon {eps}")
        # print(f"Took {round((time.time() - start_time)/60, 1)} minutes")

    return sim_out.reset_index(drop=True)


if __name__ == "__main__":
    
    # Get arg imputs
    task_id = int(sys.argv[1])

    # First kernel is just a simple time varying covariance structure + unit offset
    k1 = (gpflow.kernels.Matern12(variance=1.0,
                                  lengthscales=1.0,
                                  active_dims=[2]) +
          Categorical(variance=2.0,
                           active_dims=[0]))

    # Second kernel is time varying unit specific effect + periodic overall effect
    k2 = (gpflow.kernels.Matern12(variance=1.0,
                                  lengthscales=0.5,
                                  active_dims=[2]) *
          Categorical(active_dims=[0], variance=1.0) +
          gpflow.kernels.Periodic(
              base_kernel=gpflow.kernels.SquaredExponential(
                  variance=2.0, active_dims=[2]),
              period=0.5))

    # Third kernel is random unit specific effect + treatment effect
    k3 = (Categorical(active_dims=[0], variance=2.0) +
          Categorical(active_dims=[1], variance=1.0) *
          Lin(variance=1.0,
              active_dims=[2]))

    # Fourth kernel is nonlinear random treatment effect over time +
    # nonlinear individual effect over time
    k4 = (Categorical(active_dims=[1], variance=1.5) *
          Poly(degree=3,
               offset=0.1,
               variance=1.0,
               active_dims=[2]) +
          Categorical(active_dims=[0], variance=1.5) *
          gpflow.kernels.SquaredExponential(variance=1.0,
                                            lengthscales=0.5,
                                            active_dims=[2]))

    # Kernel dictionary
    kernel_dictionary = {
        'y1': {'model': k1},
        'y2': {'model': k2},
        'y3': {'model': k3},
        'y4': {'model': k4}
    }

    # Set options
    np.random.seed(9102+task_id)
    # Total number of observations 2^10 = 1024
    units = [2**x for x in range(11)]
    rates = units[::-1]
    epsilons = 11*[3.0]
    iters = 4
    nested_settings = [[(r, e, u, i) for i in range(iters)] 
                       for r, e, u in zip(rates, epsilons, units)]
    sim_settings = [item for sublist in nested_settings for item in sublist]
    # sim_settings = list(itertools.product(*[rates, epsilons, units, list(range(0, iters))]))
    np.random.shuffle(sim_settings)
    # print(sim_settings)
    kernel_list = [
        Lin(),
        Poly(),
        gpflow.kernels.SquaredExponential(),
        gpflow.kernels.Matern12(),
        gpflow.kernels.ArcCosine(),
        gpflow.kernels.Periodic(base_kernel=gpflow.kernels.SquaredExponential())
    ]

    # Run simulation
    start_time = time.time()
    with tqdm_joblib(tqdm(desc="Simulation", total=len(sim_settings))) as progress_bar:
        sim_out = Parallel(n_jobs=40, verbose=1)(
            delayed(run_simulation)(
                rate=r,
                epsilon=e,
                units=u,
                iters=i,
                kernel_list=kernel_list
            )
            for r, e, u, i in sim_settings
        )

    # Collapse output
    sim_results = pd.concat(sim_out)
    end_time = time.time()
    print("----%.2f seconds----"%(end_time - start_time))

    # Save output
    f = open("./unit_vs_rate_output/sim_results_unit_vs_rate"+str(task_id)+".pkl", "wb")
    pickle.dump(sim_results, f)
    f.close()
