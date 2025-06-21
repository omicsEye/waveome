# Libraries
import copy
import pickle
import re
import sys
import time
import warnings

import gpflow
import numpy as np
import pandas as pd
import ray
import statsmodels as sm
import tensorflow as tf
from joblib import Parallel, delayed
from scipy.linalg import LinAlgError
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler
from sklearn.utils._testing import ignore_warnings
from statsmodels.formula.api import glm, mixedlm
from statsmodels.gam.api import BSplines, GLMGam
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from tqdm import tqdm

import waveome
from waveome.kernels import Categorical
from waveome.model_search import GPSearch
from waveome.utilities import tqdm_joblib

# Options
warnings.simplefilter('ignore', ConvergenceWarning)
warnings.simplefilter('ignore', RuntimeWarning)
warnings.simplefilter('ignore', sm.tools.sm_exceptions.ValueWarning)
# gpflow.config.set_default_float(np.float32)

# Objects
## Specify kernels
# First kernel is just a simple time varying covariance structure + unit offset
k1 = (gpflow.kernels.Matern12(variance=1.0,
                              lengthscales=1.0,
                              active_dims=[2]) +
      Categorical(variance=2.0,
                  active_dims=[0]))

# Second kernel is random unit specfic effect + time varying unit specific effect + 
# periodic overall effect
k2 = (#Categorical(variance=0.5,
      #            active_dims=[0]) + 
      gpflow.kernels.Matern12(variance=1., #0.5 
                                        lengthscales=0.5, 
                                        active_dims=[2]) * 
     Categorical(active_dims=[0]) + 
     gpflow.kernels.Periodic(
         base_kernel=gpflow.kernels.SquaredExponential(
             variance=2., active_dims=[2]), # variance=2.5
         period=3.0))

# Third kernel is random unit specific effect + treatment effect
k3 = (Categorical(active_dims=[0], variance=0.5) + 
      Categorical(active_dims=[1]) * 
      gpflow.kernels.Linear(variance=0.1, 
                            active_dims=[2]))

# Fourth kernel is random unit effect + nonlinear random treatment effect over time + 
# nonlinear individual effect over time
k4 = (#Categorical(active_dims=[0]) + 
      Categorical(active_dims=[1], variance=0.001) * 
      gpflow.kernels.Polynomial(degree=2, 
                                offset=3., #0.05, 
                                variance=1., #0.05, 
                                active_dims=[2]) + 
#       gpflow.kernels.Linear(active_dims=[2]) + 
      Categorical(active_dims=[0]) *
      # gpflow.kernels.Matern12(variance=1.0, 
      #                         lengthscales=3.0, 
      #                         active_dims=[2]) + 
      gpflow.kernels.SquaredExponential(variance=2.,
                                        lengthscales=0.5,
                                        active_dims=[2]))


# Kernel dictonary
kern_out = {'y1': {'model': k1},
            'y2': {'model': k2},
            'y3': {'model': k3},
            'y4': {'model': k4}}

# Functions
def sim_data(
    rate=12,
    num_units=50,
    fixed_num=False, 
    include_output=True,
    kern_out=None,
    eps=0,
    alpha=1.,
    random_seed=None,
):
    """ Simulate data for evalulation with known GP process.

    """
    
    # Set random seed
    if random_seed is not None:
        np.random.seed(random_seed)

    # Assign treatment group to each unit
    #treat_group = np.repeat([0,1], num_units/2)
    prob_treat = np.random.beta(a=1, b=1)
    treat_group = np.random.binomial(n=1, p=prob_treat, size=num_units)

    # Sample number of observations for each unit
    if fixed_num:
        num_obs = np.repeat(rate, num_units)
    else:
        num_obs = np.random.poisson(lam=rate, size=num_units)

    # Sample from uniform distribution for observation times
    x = np.concatenate(
        [np.sort(np.round(np.random.uniform(low=0, high=12, size=i),1)) for i in num_obs],
        axis=0
    )

    # Standardize
    x = (x - x.mean()) / x.std()

    # Put unit ID and observation time together
    df = np.array(
        [np.repeat(np.arange(num_units), num_obs),
        np.repeat(treat_group, num_obs),
        x],
        dtype=gpflow.default_float()
    ).T
    
    df = pd.DataFrame(
        df,
        columns = ['id', 'treat', 'time']
    )

    likelihood_params = {}
        
    if include_output and kern_out is not None:

        for k in kern_out.keys():
        
            # Simulate output
            try:
                f_ = np.random.multivariate_normal(
                    mean=np.zeros_like(df.iloc[:,0]).flatten(),
                    cov=(
                        kern_out[k]['model'](df)
                        +1e-6*np.eye(df.shape[0], dtype=gpflow.default_float())
                    ),
                    size=1
                )[0]
            except:
                return None, None
            
            # Add noise
            # df[k] = f_ + np.random.normal(loc=0, scale=eps, size=len(f_))
            f_noise_ = f_ + np.random.normal(loc=0, scale=eps, size=len(f_))

            # Transform latent value
            mu_ = np.exp(f_noise_)

            # # Sample dispersion parameter
            # alpha_ = np.random.exponential(scale=1.)
            alpha_ = alpha

            # Transform params to variance
            sigma2_ = mu_ + alpha_ * mu_**2
            p_ = (mu_ / sigma2_)
            n_ = (mu_**2) / (sigma2_ - mu_)

            # Store parameters from likelihood
            likelihood_params[k] = {
                "p": p_,
                "n": n_,
                "alpha": alpha_
            }

            # Now get observed values
            df[k] = np.random.negative_binomial(
                n=n_.flatten(),
                p=p_.flatten(),
                size=len(f_)
            ).astype(float)

    return df, likelihood_params

def retrieve_features_in_models(model_object):
    out = {}

    # Get additive kernel specific feature names 
    for k, v in model_object.models.items():

        # # Search model
        # if model_object.model_selection_type == "stepwise":
        #     out[k] = (
        #         [
        #             "*".join(np.array(model_object.feat_names)[np.array(y, dtype=np.int32)]) for y in 
        #                 [re.findall(r"\d+", x) for x in 
        #                     v["best_model"].split("+")
        #                 ]
        #         ]
        #     )
        # # Penalization model
        # else:
        out[k] = (
            [
                "*".join(np.array(model_object.feat_names)[np.array(y, dtype=np.int32)]) for y in 
                    [re.findall(r"\[(\d+)\]", x) for x in 
                        v.kernel_name.split("+")
                    ]
            ]
        )
    
    return out

def fit_mixed_models(lm_train_df, out_var="y1"):
    model_name_list = ["m_full", "m_time", "m_treat", "m_intercept", "m_fe"]
    formula_list = 5*[f"{out_var} ~ time + treat"]
    re_list = ["1 + time + treat", "1 + time", "1 + treat", "1", ""]

    out_dict = {}

    for m, f, r in zip(model_name_list, formula_list, re_list):
        try:
            if m != "m_fe":
                out_dict[m] = mixedlm(
                    formula=f,
                    re_formula=r,
                    data=lm_train_df,
                    groups="id"
                ).fit()
            else:
                out_dict[m] = glm(
                    formula=f,
                    data=lm_train_df
                ).fit()
        except: # (np.linalg.LinAlgError, ValueError) as e:
            out_dict[m] = None

    return out_dict

def test_ll(larger_model, smaller_model):
    from scipy.stats import chi2
    if larger_model is None and smaller_model is None:
        return np.nan
    elif larger_model is None:
        return 1.
    elif smaller_model is None:
        return 0.
    else:
        if smaller_model.llf > larger_model.llf:
            return 1.
        df = len(larger_model.params) - len(smaller_model.params)
        val = -2 * (smaller_model.llf - larger_model.llf)
        return chi2.pdf(x=val, df=df)

def mean_squared_error(y_true, y_pred, y_int=None):
    # Fill in missing predictions with training mean
    y_pred[np.isnan(y_pred)] = y_int
    return np.mean((y_true - y_pred) ** 2)

def return_linear_terms(m_full, m_time, m_treat, m_intercept, m_fe, df, cutoff=0.05):
    # Get best random effects portion from single effect
    if m_time is None and m_treat is None:
        best_single = None
        best_re_vals = []
    elif m_time is None:
        best_single = m_treat
        best_re_vals = ["id", "id*treat"]
    elif m_treat is None:
        best_single = m_time
        best_re_vals = ["id", "id*time"]
    elif m_time.llf > m_treat.llf:
        best_single = m_time
        best_re_vals = ["id", "id*time"]
    else:
        best_single = m_treat
        best_re_vals = ["id", "id*treat"]

    # Now search through nested models
    if test_ll(m_full, best_single) < 0.05:
        b_mod = m_full
        re_vals = ["id", "id*time", "id*treat"]
    elif test_ll(best_single, m_intercept) < 0.05:
        b_mod = best_single
        re_vals = best_re_vals
    elif test_ll(m_intercept, m_fe) < 0.05:
        b_mod = m_intercept
        re_vals = ["id"]
    else:
        b_mod = m_fe
        re_vals = []

    if b_mod is None:
        return None, []

    # Now get significant fixed effects from best model
    wt = b_mod.wald_test_terms(scalar=True)
    fe_vals = wt.table.query("pvalue < 0.05").index.tolist()
    fe_vals = [x for x in fe_vals if "Intercept" not in x]

    # Refit best model (check random effects then simplify formula)
    if len(best_re_vals) > 1:
        re_form = "1+" + "+".join([x.replace("id*", "") for x in best_re_vals if x != "id"])
    elif len(best_re_vals) == 1:
        re_form = "1"
    else:
        re_form = ""

    # New formula
    if len(fe_vals) > 0:
        exog_form = '+'.join([x for x in b_mod.wald_test_terms(scalar=True).table.query("pvalue < 0.05").index if x != 'Intercept' ])
    else:
        exog_form = "1"
    fe_form = f"{b_mod.model.endog_names} ~ {exog_form}"
    
    try:
        if isinstance(b_mod, sm.genmod.generalized_linear_model.GLMResultsWrapper):
            b_mod_out = glm(
                formula=fe_form,
                data=df
            ).fit()
        else:
            b_mod_out = mixedlm(
                formula=fe_form,
                re_formula=re_form,
                data=df,
                groups="id"
            ).fit()
    except:
        b_mod_out = b_mod

    return b_mod_out, (fe_vals + re_vals)


def get_mixed_preds(mod, x_input):

    # Get fixed effect component first
    preds = mod.predict(x_input)
    try:
        if hasattr(mod, "random_effects"):
            re_shape = list(mod.random_effects.values())[0].values.shape
            # Get random effect values for each group
            b_is = np.array([
                mod.random_effects[x[1]["id"]].values.astype(float)
                if x[1]["id"] in mod.random_effects else np.zeros(shape=re_shape)
                for x in x_input.iterrows()
            ])
            # Get observed value for each random effect
            z_is = np.array([
                x[1][mod.random_effects[x[1]["id"]].index].values.astype(float)
                if x[1]["id"] in mod.random_effects else np.zeros(shape=re_shape)
                for x in x_input.iterrows()
            ])
            # Overwrite ID with an indiciator
            np.put(a=z_is, ind=0, v=1)
            # Sum up all random effect contributions
            re_preds = np.sum(z_is * b_is, axis=1)
            preds += re_preds

        return preds
    except:
        return preds

def calc_kl_all(n, p, x, y, est_m, m_type, log_y=False, y_pred = None, y_std = None):
    from scipy.special import kl_div
    from scipy.stats import nbinom, norm

    # Get prob of observations under true distribution
    p_x = nbinom(n=n, p=p).pmf(y)

    # Transform outcome if on log scale
    if log_y:
        y_trans = np.log(y + 1)
    else:
        y_trans = y

    if m_type == "gp":
        q_x = np.exp(est_m.predict_log_density(
            (
                tf.convert_to_tensor(x),
                tf.convert_to_tensor(y_trans.values[:, None])
            )
        ))
        q_x[np.isnan(q_x)] = 0
        q_x[np.isinf(q_x)] = 1
    elif m_type == "mixed":
        if y_pred is None:
            mus = get_mixed_preds(est_m, x)
            if hasattr(est_m.model, "get_scale"):
                scale = np.sqrt(est_m.model.get_scale(
                    fe_params=est_m.fe_params,
                    cov_re=est_m.cov_re,
                    vcomp=est_m.vcomp
                ))
            else:
                scale = np.sqrt(est_m.model.scale)
        else:
            mus = y_pred
            scale = y_std
        scale += 1e-6
        q_x = norm.pdf(x=y_trans, loc=mus, scale=scale)

    elif m_type == "glm":
        if y_pred is None:
            nb_preds = est_m.predict(x)
        else:
            nb_preds = y_pred

        try:
            nb_sig2 = nb_preds + est_m.k_constant * nb_preds ** 2 +  1e-6
        except:
            nb_sig2 = nb_preds + 1.0 * nb_preds ** 2 +  1e-6
        nb_p = nb_preds / nb_sig2 + 1e-6
        nb_n = nb_preds ** 2 / (nb_sig2 - nb_preds) + 1e-6
        q_x = nbinom(n=nb_n, p=nb_p).pmf(y_trans)
    elif m_type == "gam":
        # gam_preds = est_m.predict(
        #     exog = x[["id", "treat"]],
        #     exog_smooth = x["time"]
        # )
        gam_preds = y_pred
        k_const = 1 if est_m is None else est_m.k_constant
        nb_sig2 = gam_preds + k_const * gam_preds ** 2 + 1e-6
        nb_p = gam_preds / nb_sig2 + 1e-6
        nb_n = gam_preds ** 2 / (nb_sig2 - gam_preds) + 1e-6
        q_x = nbinom(n=nb_n, p=nb_p).pmf(y_trans)
    elif m_type == "lasso":
        # mus = est_m.predict(x)
        mus = y_pred
        scale = np.std(
            y_trans - mus
        ) + 1e-6
        q_x = norm.pdf(x=y_trans, loc=mus, scale=scale)
    else:
        ValueError("Unknown m_type requested.")

    # Truncate probabilities
    p_x[p_x <= 0] = 0.001
    p_x[p_x >= 1] = 0.999
    q_x[q_x <= 0] = 0.001
    q_x[q_x >= 1] = 0.999
    # q_x[np.isnan(q_x)] = np.nanmax(q_x)

    # return np.sum(kl_div(p_x, q_x))
    return np.mean(np.log(p_x / q_x))

# Wrap everything in a large function
@ignore_warnings(category=ConvergenceWarning)
@ray.remote
def run_simulation(
    input_df=None,
    kern_out=None,
    rate=50,
    num_units=10,
    epsilon=0,
    alpha=1,
    random_seed=0
):
    
    np.random.seed(random_seed)
    
    # Create output dataframe
    output_df = input_df.copy()
    output_df["features"] = ""
    output_df[["train_kl", "holdout_kl", "train_mse", "holdout_mse"]] = np.nan

    # Output names
    input_names = ["id", "treat", "time"]
    output_names = ["y1", "y2", "y3", "y4"]

    # First simulate some data
    print("Simulating data")
    sim_df, sim_dict = sim_data(
        kern_out=kern_out,
        random_seed=random_seed,
        num_units=num_units,
        rate=rate,
        eps=epsilon,
        alpha=alpha
    )

    # End iteration if simulated data fails
    if sim_df is None:
        print("Simulating data failed!")
        return pd.DataFrame()

    # Then split into training and holdout
    train_idx = sim_df.sample(frac=0.8, random_state=random_seed).index.to_numpy()
    train_df = sim_df.iloc[train_idx]
    holdout_df = sim_df.iloc[sim_df.index.difference(train_idx)]
    #print(f"{train_df.shape=}, {holdout_df.shape=}")

    train_dict = {}
    holdout_dict = {}
    for k, v in sim_dict.items():
        train_dict[k] = {}
        holdout_dict[k] = {}

        for k2, v2 in v.items():
            if k2 != "alpha":
                train_dict[k][k2] = v2[train_idx]
                holdout_dict[k][k2] = v2[holdout_df.index.to_numpy()]
            else:
                train_dict[k][k2] = v2
                holdout_dict[k][k2] = v2

    # Also make transformations for Gaussian output models
    lm_train_df = (
        train_df
        .assign(
            y1=lambda x: np.log(x["y1"] + 1),
            y2=lambda x: np.log(x["y2"] + 1),
            y3=lambda x: np.log(x["y3"] + 1),
            y4=lambda x: np.log(x["y4"] + 1),
        ).astype({"id": str})
    )

    lm_holdout_df = (
        holdout_df
        .assign(
            y1=lambda x: np.log(x["y1"] + 1),
            y2=lambda x: np.log(x["y2"] + 1),
            y3=lambda x: np.log(x["y3"] + 1),
            y4=lambda x: np.log(x["y4"] + 1),
        ).astype({"id": str})
    )

    # Get the full list of IDs
    id_list = sorted(set(sim_df.id)) #.astype(str)))

    # Track timing for each model
    start_time = time.time()

    ###########################
    # Perform GP penalization #
    ###########################
    print("GP penalization")
    gps_sim_pen = GPSearch(
        X=train_df[input_names],
        Y=train_df[output_names],
        unit_col="id",
        categorical_vars=["treat"],
        outcome_likelihood="negativebinomial",
        Y_transform=None
    )
    gps_sim_pen.penalized_optimization(random_seed=random_seed)

    # Which features were chosen?
    gpp_feats = retrieve_features_in_models(gps_sim_pen)
    for k, v in gpp_feats.items():
        output_df.loc[
            (output_df["model"] == "waveome_penalized") & (output_df["output"] == k),
            "features"
        ] = ",".join(v) if isinstance(v, list) else v
    
    # KL-divergence and MSE for both training and holdout set
    for o in output_names:
        gpp_train_kl = calc_kl_all(
            n=train_dict[o]["n"],
            p=train_dict[o]["p"],
            x=train_df[input_names],
            y=train_df[o],
            est_m=gps_sim_pen.models[o],
            m_type="gp"
        )
        gpp_train_mse = mean_squared_error(
            y_true=train_df[o],
            y_pred=gps_sim_pen.models[o].predict_y(
                tf.convert_to_tensor(train_df[input_names])
            )[0].numpy().flatten(),
            y_int=train_df[o].mean()
        )
        output_df.loc[
            (output_df["model"] == "waveome_penalized") & (output_df["output"] == o),
            ["train_kl", "train_mse"]
        ] = gpp_train_kl, gpp_train_mse
        gpp_holdout_kl = calc_kl_all(
            n=holdout_dict[o]["n"],
            p=holdout_dict[o]["p"],
            x=holdout_df[input_names],
            y=holdout_df[o],
            est_m=gps_sim_pen.models[o],
            m_type="gp"
        )
        gpp_holdout_mse = mean_squared_error(
            y_true=holdout_df[o],
            y_pred=gps_sim_pen.models[o].predict_y(
                tf.convert_to_tensor(holdout_df[input_names])
            )[0].numpy().flatten(),
            y_int=train_df[o].mean()

        )
        output_df.loc[
            (output_df["model"] == "waveome_penalized") & (output_df["output"] == o),
            ["holdout_kl", "holdout_mse"]
        ] = gpp_holdout_kl, gpp_holdout_mse

    end_time = time.time()
    print(f"GP Penalized time: {end_time - start_time}")
    start_time = end_time

    ##########################
    # Then perform GP search #
    ##########################
    print("GP search")
    gps_sim_search = GPSearch(
        X=train_df[input_names],
        Y=train_df[output_names],
        unit_col="id",
        categorical_vars=["treat"],
        outcome_likelihood="negativebinomial",
        Y_transform=None
    )
    gps_sim_search.run_search(random_seed=random_seed)

    # Which features were chosen?
    gps_feats = retrieve_features_in_models(gps_sim_search)
    for k, v in gps_feats.items():
        output_df.loc[
            (output_df["model"] == "waveome_search") & (output_df["output"] == k),
            "features"
        ] = ",".join(v) if isinstance(v, list) else v

    # KL-divergence and MSE for both training and holdout set
    for o in output_names:
        gps_train_kl = calc_kl_all(
            n=train_dict[o]["n"],
            p=train_dict[o]["p"],
            x=train_df[input_names],
            y=train_df[o],
            est_m=gps_sim_search.models[o],
            # est_m=gps_sim_search.models[o]["models"][gps_sim_search.models[o]["best_model"]]["model"],
            m_type="gp"
        )
        gps_train_mse = mean_squared_error(
            y_true=train_df[o],
            # y_pred=gps_sim_search.models[o]["models"][gps_sim_search.models[o]["best_model"]]["model"].predict_y(
            #     tf.convert_to_tensor(train_df[input_names])
            # )[0].numpy().flatten()
            y_pred=gps_sim_search.models[o].predict_y(
                tf.convert_to_tensor(train_df[input_names])
            )[0].numpy().flatten()
        )
        output_df.loc[
            (output_df["model"] == "waveome_search") & (output_df["output"] == o),
            ["train_kl", "train_mse"]
        ] = gps_train_kl, gps_train_mse
        gps_holdout_kl = calc_kl_all(
            n=holdout_dict[o]["n"],
            p=holdout_dict[o]["p"],
            x=holdout_df[input_names],
            y=holdout_df[o],
            est_m=gps_sim_search.models[o],
            # est_m=gps_sim_search.models[o]["models"][gps_sim_search.models[o]["best_model"]]["model"],
            m_type="gp"
        )
        gps_holdout_mse = mean_squared_error(
            y_true=holdout_df[o], 
            y_pred=gps_sim_search.models[o].predict_y(
                tf.convert_to_tensor(holdout_df[input_names])
            )[0].numpy().flatten()
        )
        output_df.loc[
            (output_df["model"] == "waveome_search") & (output_df["output"] == o),
            ["holdout_kl", "holdout_mse"]
        ] = gps_holdout_kl, gps_holdout_mse

    end_time = time.time()
    print(f"GP search time: {end_time - start_time}")
    start_time = end_time

    ###############################
    # Then perform mixed modeling #
    ###############################
    print("Mixed LM")
    # Fit models
    for o in output_names:

        # Fit models
        mm_dict = fit_mixed_models(lm_train_df, out_var=o)

        # Which features were chosen from the best model?
        try:
            best_mm, mm_feats = return_linear_terms(**mm_dict, df=lm_train_df)
        except:
            best_mm = None
            mm_feats = []
        output_df.loc[
            (output_df["model"] == "mixed_lm") & (output_df["output"] == o),
                "features"
            ] = ",".join(mm_feats) if isinstance(mm_feats, list) else mm_feats
        
        # KL-divergence and MSE for both training and holdout set
        mm_train_kl, mm_train_mse = np.nan, np.nan
        mm_holdout_kl, mm_holdout_mse = np.nan, np.nan
        try:
            mm_train_hat = np.exp(get_mixed_preds(best_mm, lm_train_df))
        except AttributeError:
            mm_train_hat = np.repeat(
                train_df[o].mean(),
                repeats=train_df.shape[0]
            )
        mm_train_kl = calc_kl_all(
            n=train_dict[o]["n"],
            p=train_dict[o]["p"],
            x=lm_train_df[input_names],
            y=train_df[o].values,
            est_m=best_mm,
            m_type="mixed",
            log_y=True,
            y_pred=mm_train_hat,
            y_std=train_df[o].std()
        )
        mm_train_mse = mean_squared_error(
            y_true=train_df[o],
            y_pred=mm_train_hat
        )
        output_df.loc[
            (output_df["model"] == "mixed_lm") & (output_df["output"] == o),
            ["train_kl", "train_mse"]
        ] = mm_train_kl, mm_train_mse

        try:
            mm_holdout_hat = np.exp(get_mixed_preds(best_mm, lm_holdout_df))
        except AttributeError:
            mm_holdout_hat = np.repeat(
                train_df[o].mean(),
                repeats=holdout_df.shape[0]
            )
        mm_holdout_kl = calc_kl_all(
            n=holdout_dict[o]["n"],
            p=holdout_dict[o]["p"],
            x=lm_holdout_df[input_names],
            y=holdout_df[o].values,
            est_m=best_mm,
            m_type="mixed",
            log_y=True,
            y_pred=mm_holdout_hat,
            y_std=train_df[o].std()
        )
        mm_holdout_mse = mean_squared_error(
            y_true=holdout_df[o],
            y_pred=mm_holdout_hat
        )
        output_df.loc[
            (output_df["model"] == "mixed_lm") & (output_df["output"] == o),
            ["holdout_kl", "holdout_mse"]
        ] = mm_holdout_kl, mm_holdout_mse

        # if o == "y4":
        #     return {
        #         "model": best_mm,
        #         "dict_params": holdout_dict[o],
        #         "x": lm_holdout_df[input_names],
        #         "y": holdout_df[o]
        #     }

    end_time = time.time()
    print(f"Mixed LM time: {end_time - start_time}")
    start_time = end_time

    #######################
    # Then perform NB GLM #
    #######################
    print("NB GLM")
    # Go through each output
    for o in output_names:
        best_nb = None
        best_nb_ll = -np.inf
        best_alpha = None
        # Find best alpha value based on log-likelihood
        for a in np.arange(start=1, stop=11):

            m_nb_glm = glm(
                formula = f"{o} ~ C(id, levels=id_list) + time + treat + C(id, levels=id_list) * time + C(id, levels=id_list) * treat + time * treat",
                data = train_df,
                family = sm.genmod.families.NegativeBinomial(alpha=a)
            )

            try:
                m_nb_glm = m_nb_glm.fit(cov_type="cluster", cov_kwds={"groups": train_df.id})
            except:
                try:
                    m_nb_glm = m_nb_glm.fit(method="lbfgs")
                except:
                    continue

            if m_nb_glm.llf > best_nb_ll:
                best_nb_ll = m_nb_glm.llf
                best_nb = copy.copy(m_nb_glm)
                best_alpha = a

        if best_nb is not None:
            # Find significant features from model
            # Make sure we can do a wald test - not possible if covariance matrix ill-defined
            try:
                wt_nb = best_nb.wald_test_terms(scalar=True)
                wt_nb = set([
                    x.replace("C(", "").replace(", levels=id_list)", "").replace(", levels=id", "").replace(":", "*")
                    for x in wt_nb.table.query("pvalue < 0.05").index.tolist()
                ])
                if "Intercept" in wt_nb:
                    wt_nb.remove("Intercept")
                wt_nb = list(wt_nb)

                # Reft model to only significant features
                # New formula
                if len(wt_nb) > 0:
                    exog_form = '+'.join(
                        [x for x in best_nb.wald_test_terms(scalar=True).table.query("pvalue < 0.05").index if x != 'Intercept' ])
                else:
                    exog_form = "1"
                fe_form = f"{o} ~ {exog_form}"
                best_nb_updated = glm(
                    formula = fe_form,
                    data = train_df,
                    family = sm.genmod.families.NegativeBinomial(alpha=best_alpha)
                )
                try:
                    best_nb_updated = best_nb_updated.fit(cov_type="cluster", cov_kwds={"groups": train_df.id})
                except:
                    try:
                        best_nb_updated = best_nb_updated.fit()
                    except:
                        best_nb_updated = best_nb
            except:
                best_nb_updated = best_nb
                wt_nb = []
            output_df.loc[
                (output_df["model"] == "glm") & (output_df["output"] == o),
                    "features"
                ] = ",".join(wt_nb) if isinstance(wt_nb, list) else wt_nb
        
        # Calculate and store training KL and MSE
        nb_train_kl, nb_train_mse = np.nan, np.nan
        nb_holdout_kl, nb_holdout_mse = np.nan, np.nan
        try:
            nb_train_hat = best_nb_updated.predict(
                train_df
            )
        except:
            nb_train_hat = np.repeat(
                train_df[o].mean(),
                repeats=train_df.shape[0]
            )

        nb_train_kl = calc_kl_all(
            n=train_dict[o]["n"],
            p=train_dict[o]["p"],
            x=train_df[["id", "treat", "time"]],
            y=train_df[o],
            est_m=None,
            m_type="glm",
            y_pred=nb_train_hat
        )
        nb_train_mse = mean_squared_error(
            y_true=train_df[o],
            y_pred=nb_train_hat
        )
        output_df.loc[
            (output_df["model"] == "glm") & (output_df["output"] == o),
            ["train_kl", "train_mse"]
        ] = nb_train_kl, nb_train_mse

        # Calculate and store holdout KL and MSE
        try:
            nb_holdout_hat = best_nb_updated.predict(
                holdout_df
            )
        except:
            nb_holdout_hat = np.repeat(
                train_df[o].mean(),
                repeats=holdout_df.shape[0]
            )
        nb_holdout_kl = calc_kl_all(
            n=holdout_dict[o]["n"],
            p=holdout_dict[o]["p"],
            x=holdout_df[["id", "treat", "time"]],
            y=holdout_df[o],
            est_m=None,
            m_type="glm",
            y_pred=nb_holdout_hat
        )
        nb_holdout_mse = mean_squared_error(
            y_true=holdout_df[o],
            y_pred=nb_holdout_hat
        )
        output_df.loc[
            (output_df["model"] == "glm") & (output_df["output"] == o),
            ["holdout_kl", "holdout_mse"]
        ] = nb_holdout_kl, nb_holdout_mse

    end_time = time.time()
    print(f"NB-GLM time: {end_time - start_time}")
    start_time = end_time

    ####################
    # Then perform GAM #
    ####################
    print("GAM")
    # Truncate time in holdout set for gam, else knot error
    gam_holdout_df = (
        holdout_df
        .assign(
            time = lambda x: x["time"].clip(
                lower=train_df["time"].min(),
                upper=train_df["time"].max()
            )
        )
    )


    for o in output_names:

        # Create spline object
        bs = BSplines(train_df["time"], df=12, degree=3)

        best_gam = None
        best_gam_ll = -np.inf
        best_gam_alpha = None
        for a in np.arange(start=1, stop=11):

            gam_bs = GLMGam.from_formula(
                f'{o} ~ C(id, levels=id_list) + treat',
                data=train_df,
                smoother=bs,
                family=sm.genmod.families.NegativeBinomial(alpha=a)
            )

            # Set penalization amound
            try:
                gam_bs.alpha = gam_bs.select_penweight_kfold()[0]
            except:
                continue

            # Fit model
            try:
                fit_gam = gam_bs.fit()
            except:
                continue

            if fit_gam.llf > best_gam_ll:
                best_gam_ll = fit_gam.llf
                best_gam = copy.copy(fit_gam)
                best_gam_alpha = a

        if best_gam is not None:
            # Which terms are significant?
            wt_gam = best_gam.wald_test_terms(scalar=True)
            gam_feats = set([ x.split("_")[0].split("[")[0].replace("C(", "").replace(", levels=id_list)", "").replace(", levels=id", "") for x in wt_gam.table.query("pvalue < 0.05").index.tolist()])
            if "Intercept" in gam_feats:
                gam_feats.remove("Intercept")
            gam_feats = list(gam_feats)

            output_df.loc[
                (output_df["model"] == "gam") & (output_df["output"] == o),
                    "features"
            ] = ",".join(gam_feats) if isinstance(gam_feats, list) else gam_feats

        gam_train_kl, gam_train_mse = np.nan, np.nan
        gam_holdout_kl, gam_holdout_mse = np.nan, np.nan

        # Calculate and store training KL and MSE
        try:
            gam_train_hat = best_gam.predict(
                exog = train_df[["id", "treat"]],
                exog_smooth = train_df["time"]
            )
        except AttributeError:
            gam_train_hat = np.repeat(
                train_df[o].mean(),
                repeats=train_df.shape[0]
            )
        gam_train_kl = calc_kl_all(
            n=train_dict[o]["n"],
            p=train_dict[o]["p"],
            x=train_df[["id", "treat", "time"]],
            y=train_df[o],
            est_m=None,
            m_type="gam",
            y_pred=gam_train_hat
        )
        gam_train_mse = mean_squared_error(
            y_true=train_df[o],
            y_pred=gam_train_hat
        )
        output_df.loc[
            (output_df["model"] == "gam") & (output_df["output"] == o),
            ["train_kl", "train_mse"]
        ] = gam_train_kl, gam_train_mse

        # Calculate and store holdout KL and MSE
        try:
            gam_holdout_hat = best_gam.predict(
                exog = gam_holdout_df[["id", "treat"]],
                exog_smooth = gam_holdout_df["time"]
            )
        except AttributeError:
            gam_holdout_hat = np.repeat(
                train_df[o].mean(),
                repeats=holdout_df.shape[0]
            )
        gam_holdout_kl = calc_kl_all(
            n=holdout_dict[o]["n"],
            p=holdout_dict[o]["p"],
            x=gam_holdout_df[["id", "treat", "time"]],
            y=holdout_df[o],
            est_m=best_gam,
            m_type="gam",
            y_pred=gam_holdout_hat
        )
        gam_holdout_mse = mean_squared_error(
            y_true=holdout_df[o],
            y_pred=gam_holdout_hat
        )
        output_df.loc[
            (output_df["model"] == "gam") & (output_df["output"] == o),
            ["holdout_kl", "holdout_mse"]
        ] = gam_holdout_kl, gam_holdout_mse
    
    end_time = time.time()
    print(f"GAM time: {end_time - start_time}")
    start_time = end_time

    ######################
    # Then perform LASSO #
    ######################
    print("LASSO")
    # Preprocessing
    ohe = OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False)
    pf = PolynomialFeatures(include_bias=False, interaction_only=True)
    ss = StandardScaler()
    lm_train_X_dummies = pd.DataFrame(
        ohe.fit_transform(lm_train_df[["id"]]),
        columns=ohe.get_feature_names_out()
    ).assign(
        treat = lm_train_df["treat"].values,
        time = lm_train_df["time"].values
    )
    lm_train_X_dummies_int = pf.fit_transform(lm_train_X_dummies)
    lm_train_X_dummies_int_scaled = ss.fit_transform(lm_train_X_dummies_int)

    # Get rid of id x id interactions
    col_flags = np.array([x.count("id_") < 2 for x in pf.get_feature_names_out()])
    lm_train_X_dummies_int_scaled = lm_train_X_dummies_int_scaled[:, col_flags]

    # Do same for holdout set
    lm_holdout_X_dummies = pd.DataFrame(
        ohe.transform(lm_holdout_df[["id"]]),
        columns=ohe.get_feature_names_out()
    ).assign(
        treat = lm_holdout_df["treat"].values,
        time = lm_holdout_df["time"].values
    )
    lm_holdout_X_dummies_int = pf.transform(lm_holdout_X_dummies)
    lm_holdout_X_dummies_int_scaled = ss.transform(lm_holdout_X_dummies_int)
    lm_holdout_X_dummies_int_scaled = lm_holdout_X_dummies_int_scaled[:, col_flags]


    # Finally train model for each outcome
    for o in output_names:
        try:
            lcv = LassoCV()
            lcv.fit(
                lm_train_X_dummies_int_scaled,
                lm_train_df[o]
            )
        except:
            lcv = None
        
        if lcv is not None:
            # Get features that have at least some coefficient value
            non_zero_features = pf.get_feature_names_out()[col_flags][np.abs(lcv.coef_) > 0.01]
            # non_zero_features[np.array(["time" in x for x in non_zero_features])]

            # Transform output to be similar to other models
            lasso_feats = [re.sub(" ", "*", y) for y in set(
                    [re.sub("\_\d+\.0", "", x) for x in non_zero_features]
                )]
            output_df.loc[
                (output_df["model"] == "lasso") & (output_df["output"] == o),
                    "features"
            ] = ",".join(lasso_feats) if isinstance(lasso_feats, list) else lasso_feats

        # Calculate and store training KL and MSE
        lasso_train_kl, lasso_train_mse = np.nan, np.nan
        lasso_holdout_kl, lasso_holdout_mse = np.nan, np.nan

        # Calculate and store training KL and MSE
        try:
            lasso_train_hat = np.exp(
                lcv.predict(lm_train_X_dummies_int_scaled)
            )
        except AttributeError:
            lasso_train_hat = np.repeat(
                train_df[o].mean(),
                repeats=train_df.shape[0]
            )
        lasso_train_kl = calc_kl_all(
            n=train_dict[o]["n"],
            p=train_dict[o]["p"],
            x=lm_train_X_dummies_int_scaled,
            y=train_df[o],
            est_m=lcv,
            m_type="lasso",
            log_y=True,
            y_pred=np.log(lasso_train_hat)
        )
        lasso_train_mse = mean_squared_error(
            y_true=train_df[o],
            y_pred=lasso_train_hat
        )
        output_df.loc[
            (output_df["model"] == "lasso") & (output_df["output"] == o),
            ["train_kl", "train_mse"]
        ] = lasso_train_kl, lasso_train_mse

        # Calculate and store holdout KL and MSE
        try:
            lasso_holdout_hat = np.exp(
              lcv.predict(lm_holdout_X_dummies_int_scaled)
            )
        except AttributeError:
            lasso_holdout_hat = np.repeat(
                train_df[o].mean(),
                repeats=holdout_df.shape[0]
            )
        lasso_holdout_kl = calc_kl_all(
            n=holdout_dict[o]["n"],
            p=holdout_dict[o]["p"],
            x=lm_holdout_X_dummies_int_scaled,
            y=holdout_df[o],
            est_m=lcv,
            m_type="lasso",
            log_y=True,
            y_pred=np.log(lasso_holdout_hat)
        )
        lasso_holdout_mse = mean_squared_error(
            y_true=holdout_df[o],
            y_pred=lasso_holdout_hat
        )
        output_df.loc[
            (output_df["model"] == "lasso") & (output_df["output"] == o),
            ["holdout_kl", "holdout_mse"]
        ] = lasso_holdout_kl, lasso_holdout_mse

    end_time = time.time()
    print(f"LASSO time: {end_time - start_time}")
    start_time = end_time

    #######################
    # Then perform GP-ARD #
    #######################
    print("GP-ARD")
    for o in output_names:

        # Fit model
        gp_ard = gpflow.models.GPR(
            data=(
                tf.convert_to_tensor(lm_train_X_dummies_int_scaled, dtype=tf.dtypes.double),
                tf.convert_to_tensor(lm_train_df[o].values.reshape(-1, 1), dtype=tf.dtypes.double)
            ),
            kernel=gpflow.kernels.SquaredExponential(
                lengthscales=np.ones(lm_train_X_dummies_int_scaled.shape[1])
            ),
        )

        # Optimize hyperparameters
        try:
            gpflow.optimizers.Scipy().minimize(
                gp_ard.training_loss,
                gp_ard.trainable_variables
            )
        except:
            pass

        # Get features that have a low lengthscale value
        # (i.e. smaller than +/- 3 [or 6] given standardization)
        non_zero_features_ard = (
            pf.get_feature_names_out()
            [col_flags]
            [(gp_ard.kernel.lengthscales < 6) & (gp_ard.kernel.lengthscales > 0.01)]
        )
        ard_feats = [
            re.sub(" ", "*", y) for y in set(
                [re.sub("\_\d+\.0", "", x) for x in non_zero_features_ard]
            )]

        output_df.loc[
            (output_df["model"] == "ard") & (output_df["output"] == o),
                "features"
        ] = ",".join(ard_feats) if isinstance(ard_feats, list) else ard_feats


        ard_train_kl, ard_train_mse = np.nan, np.nan
        ard_holdout_kl, ard_holdout_mse = np.nan, np.nan

        # KL-divergence and MSE for training set
        try:
            ard_train_hat = np.exp(
                gp_ard.predict_y(lm_train_X_dummies_int_scaled)[0]
            ).flatten()
        except AttributeError:
            ard_train_hat = np.repeat(
                train_df[o].mean(),
                repeats=train_df.shape[0]
            )
        ard_train_hat[np.isnan(ard_train_hat)] = train_df[o].mean()
        ard_train_hat[np.isinf(ard_train_hat)] = train_df[o].mean()

        ard_train_kl = calc_kl_all(
            n=train_dict[o]["n"],
            p=train_dict[o]["p"],
            x=lm_train_X_dummies_int_scaled,
            y=train_df[o],
            est_m=gp_ard,
            m_type="gp",
            log_y=True
        )

        ard_train_mse = mean_squared_error(
            y_true=train_df[o],
            y_pred=ard_train_hat
        )
        output_df.loc[
            (output_df["model"] == "ard") & (output_df["output"] == o),
            ["train_kl", "train_mse"]
        ] = ard_train_kl, ard_train_mse

        # KL-divergence and MSE for holdout set
        try:
            ard_holdout_hat = np.exp(
                gp_ard.predict_y(lm_holdout_X_dummies_int_scaled)[0]
            ).flatten()
        except AttributeError:
            ard_holdout_hat = np.repeat(
                train_df[o].mean(),
                repeats=holdout_df.shape[0]
            )
        ard_holdout_hat[np.isnan(ard_holdout_hat)] = train_df[o].mean()
        ard_holdout_hat[np.isinf(ard_holdout_hat)] = train_df[o].mean()
        ard_holdout_kl = calc_kl_all(
            n=holdout_dict[o]["n"],
            p=holdout_dict[o]["p"],
            x=lm_holdout_X_dummies_int_scaled,
            y=holdout_df[o],
            est_m=gp_ard,
            m_type="gp",
            log_y=True
        )
        ard_holdout_mse = mean_squared_error(
            y_true=holdout_df[o],
            y_pred=ard_holdout_hat
        )
        output_df.loc[
            (output_df["model"] == "ard") & (output_df["output"] == o),
            ["holdout_kl", "holdout_mse"]
        ] = ard_holdout_kl, ard_holdout_mse

        # if o == "y4" and random_seed == 2:
        #     return {
        #         "model": gp_ard,
        #         "dict_params": train_dict[o],
        #         "x": lm_train_X_dummies_int_scaled,
        #         "y": train_df[o]
        #     }

    end_time = time.time()
    print(f"ARD time: {end_time - start_time}")
    start_time = end_time

    ##########################
    # Then perform GP-ARD-NB #
    ##########################
    print("NB-ARD")
    for o in output_names:

        # Fit model
        gp_ard_nb = gpflow.models.VGP(
            data=(
                tf.convert_to_tensor(lm_train_X_dummies_int_scaled, dtype=tf.dtypes.double),
                tf.convert_to_tensor(train_df[o].values.reshape(-1, 1), dtype=tf.dtypes.double)
            ),
            kernel=gpflow.kernels.SquaredExponential(
                lengthscales=np.ones(lm_train_X_dummies_int_scaled.shape[1])
            ),
            likelihood=waveome.likelihoods.NegativeBinomial()
        )

        # Optimize hyperparameters
        try:
            gpflow.optimizers.Scipy().minimize(
                gp_ard_nb.training_loss,
                gp_ard_nb.trainable_variables
            )
        except:
            pass

        # Get features that have a low lengthscale value
        # (i.e. smaller than +/- 3 or 6 given standardization)
        non_zero_features_ard_nb = (
            pf.get_feature_names_out()
            [col_flags]
            [(gp_ard_nb.kernel.lengthscales < 6) & (gp_ard_nb.kernel.lengthscales > 0.01)]
        )
        ard_nb_feats = [
            re.sub(" ", "*", y) for y in set(
                [re.sub("\_\d+\.0", "", x) for x in non_zero_features_ard_nb]
            )]

        output_df.loc[
            (output_df["model"] == "nb_ard") & (output_df["output"] == o),
                "features"
        ] = ",".join(ard_nb_feats) if isinstance(ard_nb_feats, list) else ard_nb_feats

        # KL-divergence and MSE for training set
        try:
            nb_ard_train_hat = np.exp(
                gp_ard_nb.predict_y(lm_train_X_dummies_int_scaled)[0]
            ).flatten()
        except AttributeError:
            nb_ard_train_hat = np.repeat(
                train_df[o].mean(),
                repeats=train_df.shape[0]
            )
        nb_ard_train_hat[np.isnan(nb_ard_train_hat)] = train_df[o].mean()
        nb_ard_train_hat[np.isinf(nb_ard_train_hat)] = train_df[o].mean()
        ard_nb_train_kl = calc_kl_all(
            n=train_dict[o]["n"],
            p=train_dict[o]["p"],
            x=lm_train_X_dummies_int_scaled,
            y=train_df[o],
            est_m=gp_ard_nb,
            m_type="gp",
        )
        ard_nb_train_mse = mean_squared_error(
            y_true=train_df[o],
            y_pred=nb_ard_train_hat
        )
        output_df.loc[
            (output_df["model"] == "nb_ard") & (output_df["output"] == o),
            ["train_kl", "train_mse"]
        ] = ard_nb_train_kl, ard_nb_train_mse

        # KL-divergence and MSE for holdout set
        try:
            nb_ard_holdout_hat = np.exp(
                gp_ard_nb.predict_y(lm_holdout_X_dummies_int_scaled)[0]
            ).flatten()
        except AttributeError:
            nb_ard_holdout_hat = np.repeat(
                train_df[o].mean(),
                repeats=holdout_df.shape[0]
            )
        nb_ard_holdout_hat[np.isnan(nb_ard_holdout_hat)] = train_df[o].mean()
        nb_ard_holdout_hat[np.isinf(nb_ard_holdout_hat)] = train_df[o].mean()
        ard_nb_holdout_kl = calc_kl_all(
            n=holdout_dict[o]["n"],
            p=holdout_dict[o]["p"],
            x=lm_holdout_X_dummies_int_scaled,
            y=holdout_df[o],
            est_m=gp_ard_nb,
            m_type="gp"
        )
        ard_nb_holdout_mse = mean_squared_error(
            y_true=holdout_df[o],
            y_pred=nb_ard_holdout_hat
        )
        output_df.loc[
            (output_df["model"] == "nb_ard") & (output_df["output"] == o),
            ["holdout_kl", "holdout_mse"]
        ] = ard_nb_holdout_kl, ard_nb_holdout_mse

        # if o == "y4":
        #     return {
        #         "model": gp_ard_nb,
        #         "dict_params": holdout_dict[o],
        #         "x": lm_holdout_X_dummies_int_scaled,
        #         "y": holdout_df[o],
        #         "holdout_hat": nb_ard_holdout_hat
        #     }

    end_time = time.time()
    print(f"NB-ARD time: {end_time - start_time}")
    start_time = end_time

    return output_df

# Get array task input
task_id = int(sys.argv[1])

# Set simulation parameter values
run_id = 5 * (task_id-1) + np.arange(5)
rate_list = [2, 4, 8, 16]
units_list = [10, 50, 100, 500]
epsilon_list = [0, 1, 10]
alpha_list = [1, 10, 100]
model_list = [
    "waveome_penalized", "waveome_search",
    "mixed_lm", "glm",
    "gam", "lasso",
    "ard", "nb_ard"
]
output_list = ["y1", "y2", "y3", "y4"]

# Cross-join to get all possible combinations
sim_output = pd.DataFrame(
    np.concatenate(
        [
            x.flatten()[:, None]
            for x in np.meshgrid(
                rate_list,
                units_list,
                epsilon_list,
                alpha_list,
                run_id,
                model_list,
                output_list
            )
        ]
        , axis=1
    ),
    columns=[
        "rate", "units", 
        "epsilon", "alpha", "run_id", 
        "model", "output"
    ],
)
sim_output = sim_output.astype({
    "rate": float,
    "units": int,
    "epsilon": float,
    "alpha": float,
    "run_id": int,
    "model": str,
    "output": str
})

# Initialize the cluster
ray.init(num_cpus=8)

sim_output_list = []
num_iters = int(sim_output.shape[0] / 32)
np.random.seed(task_id)
rand_order_iters = np.random.choice(range(num_iters), size=num_iters, replace=False)
start_time = time.time()

for i in rand_order_iters:
    start_idx = 32*i
    end_idx = 32*(i+1)
    # print(f"Starting simulation {sim_output.run_id.values[start_idx]}")
    sim_output_list.append(
        run_simulation.remote(
            input_df=sim_output.iloc[start_idx:end_idx, :],
            kern_out=kern_out,
            rate=sim_output.rate.values[start_idx],
            num_units=sim_output.units.values[start_idx],
            epsilon=sim_output.epsilon.values[start_idx],
            alpha=sim_output.alpha.values[start_idx],
            random_seed=sim_output.run_id.values[start_idx]
        )
    )

# Retrieve results
sim_output = ray.get(sim_output_list)
# # Run simulation (randomize iters)
# np.random.seed(task_id)
# with tqdm_joblib(tqdm(desc="Simulation", total=num_iters)) as progress_bar:
#     sim_output = Parallel(n_jobs=32, verbose=1)(
#         delayed(run_simulation)(
#             input_df=sim_output.iloc[(32*i):(32*(i+1)), :],
#             kern_out=kern_out,
#             rate=sim_output.rate.values[(32*i)],
#             num_units=sim_output.units.values[(32*i)],
#             epsilon=sim_output.epsilon.values[(32*i)],
#             alpha=sim_output.alpha.values[(32*i)],
#             random_seed=sim_output.run_id.values[(32*i)]
#         )
#         for i in np.random.choice(range(num_iters), size=num_iters, replace=False)
#     )

# Concatenate all of the runs
sim_results = pd.concat(sim_output, axis=0)

end_time = time.time()
print("----%.2f seconds----"%(end_time - start_time))

# Save output as pickle file
with open(f"./sim_waveome_output/sim_waveome_results_{task_id}.pickle", "wb") as handle:
    pickle.dump(sim_results, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Clear Ray
# ray.shutdown()
