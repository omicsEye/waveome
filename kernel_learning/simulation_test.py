# Libraries
import gpflow
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import os 
import tensorflow as tf
# tf.config.set_visible_devices([], 'GPU')
# tf.get_logger().setLevel('ERROR')
from gpflow.utilities import print_summary
from helper_functions import *
import time
import pickle
import itertools

# Options
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# os.environ['MKL_NUM_THREADS'] = '1'
# os.environ['OMP_NUM_THREADS'] = '1'
# os.environ['MKL_DYNAMIC'] = 'FALSE'

# Disable all GPUS
# tf.config.set_visible_devices([], 'GPU')
# visible_devices = tf.config.get_visible_devices()
# for device in visible_devices:
#     assert device.device_type != 'GPU'
print(tf.config.get_visible_devices())
# start_time = time.time()

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
    # df.time = (df.time - df.time.mean()) / df.time.std()

    if include_output and kern_out is not None:

        for k in kern_out.keys():
            # Simulate output
            f_ = np.random.multivariate_normal(
                mean=np.zeros_like(df.iloc[:, 0]).flatten(),
                cov=kern_out[k]['model'](df) + 1e-6 * np.eye(df.shape[0]),
                size=1
            )[0]

            # Add noise
            df[k] = f_ + np.random.normal(loc=0, scale=eps, size=len(f_))

    return df

def run_simulation(rate, epsilon, units, iters, kernel_list):
        
    # Prep output dataset
    sim_out = pd.DataFrame()

    # Build temporary simulated data
    temp_df = sim_data(
        rate=rate,
        num_units=units,
        include_output=True,
        kern_out=kernel_dictionary,
        set_seed=False,
        eps=epsilon
    )

    # Run kernel search process
    for i in range(4):
        search_out = split_kernel_search(
            X=temp_df[['id', 'treat', 'time']],
            Y=temp_df.drop(columns=['id', 'treat', 'time']).iloc[:, i],
            kern_list=kernel_list,
            cat_vars=[0, 1],
            unit_idx=0,
            max_depth=5,
            early_stopping=True,
            prune=True,
            keep_all=False,
            keep_only_best=True,
            lik='gaussian',
            random_seed=9102,
            metric_diff=0
            )
        


        # Save resulting kernels and information
        cur_out = pd.DataFrame({
            'rate': [rate], #4*[r],
            'eps': [epsilon], #4*[eps],
            'units': [units], #4*[u],
            'iter': [iters], #4*[i],
            'output': ['y'+str(i)], #['y1', 'y2', 'y3', 'y4'],
            'kernel': [search_out['best_model']], #[x['best_model'] for x in search_out],
            'model': [search_out['models'][search_out['best_model']]]
            #[x['models'][x['best_model']] for x in search_out]
        })

        sim_out = pd.concat([sim_out, cur_out])
        # print(f"Finished rate {r} with {u} units and epsilon {eps}")
        # print(f"Took {round((time.time() - start_time)/60, 1)} minutes")

    return sim_out.reset_index(drop=True)


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
          period=3.0))

# Third kernel is random unit specific effect + treatment effect
k3 = (Categorical(active_dims=[0], variance=2.0) +
      Categorical(active_dims=[1], variance=1.0) *
      gpflow.kernels.Linear(variance=1.0,
                            active_dims=[2]))

# Fourth kernel is nonlinear random treatment effect over time +
# nonlinear individual effect over time
k4 = (Categorical(active_dims=[0], variance=0.5) +
      Categorical(active_dims=[1], variance=1.0) *
      gpflow.kernels.Polynomial(degree=1,
                                offset=3.,
                                variance=1.0,
                                active_dims=[2]) +
      Categorical(active_dims=[0], variance=1.0) *
      gpflow.kernels.SquaredExponential(variance=2.0,
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
np.random.seed(9102)
rates = [5, 10, 20, 50, 100] # [3, 9] #[2, 4, 12] #[5, 10, 20]
units = [10, 50, 100] #[10, 30, 50]
epsilons = [0.1, 1.0, 10] # SNR [30, 3, 0.3]
iters = 100
sim_settings = list(itertools.product(*[rates, epsilons, units, list(range(0, iters))]))
np.random.shuffle(sim_settings)
#print(sim_settings)
kernel_list = [
    gpflow.kernels.SquaredExponential(),
    gpflow.kernels.Matern12(),
    gpflow.kernels.Linear(),
    gpflow.kernels.Polynomial(),
    gpflow.kernels.Periodic(base_kernel=gpflow.kernels.SquaredExponential())
]

# # THIS SECTION IS TESTING IF GPU ON/OFF IS WORKING
# df = sim_data(
#     num_units=100,
#     rate=20,
#     include_output=True,
#     kern_out=kernel_dictionary,
#     set_seed=False
# )

# foo = kernel_test(
#     X=df[['id', 'treat', 'time']],
#     Y=df['y1'],
#     k=gpflow.kernels.SquaredExponential(active_dims=[2]) + Categorical(active_dims=[0])
# )
# print(foo)
# import sys
# sys.exit('Finished')
# Rate = 9, unit = 10, 5 jobs with 5 tasks, CPU = 2.55min, GPU = locks on last task

# Run simulation
start_time = time.time()
with tqdm_joblib(tqdm(desc="Simulation", total=len(sim_settings))) as progress_bar:
    sim_out = Parallel(n_jobs=-1, verbose=1)(
        delayed(run_simulation)(
            rate=r,
            epsilon=e,
            units=u,
            iters=i,
            kernel_list=kernel_list
        )
        for r, e, u, i in sim_settings
    )
# from dask.distributed import Client
# client = Client(processes=False)
# import joblib
# with joblib.parallel_backend('dask', n_jobs=5):
#     sim_out = joblib.Parallel(verbose=1)(
#         joblib.delayed(run_simulation)(
#             rate=r, epsilon=e, 
#             units=u, iters=i, 
#             kernel_list=kernel_list)
#     for r, e, u, i in sim_settings
#     )

#start_time = time.time()
#sim_results = run_simulation(rates, epsilons, units, iters, kernel_list, start_time)

# Collapse output
sim_results = pd.concat(sim_out)
end_time = time.time()
print("----%.2f seconds----"%(end_time - start_time))

# Save output
f = open("sim_results.pkl", "wb")
pickle.dump(sim_results, f)
f.close()
