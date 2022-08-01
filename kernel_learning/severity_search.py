# Libraries
import pandas as pd
import numpy as np
import gpflow 
import pickle
from helper_functions import *
import time

# Options
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

# Data
# Read in metabolomics data
mbx = pd.read_csv('../examples/iHMP/data/iHMP_labeled_metabolomics.csv')
# Subset metabolites to labeled sets

# Get metabolite list
mbx_list = mbx.Metabolite
# Reshape metabolites for merge
mbx = mbx.iloc[:,7:].transpose().rename(columns = mbx_list)
## Only keep metabolites that have at least 20% non-missing values
# mbx_list = mbx_list[(mbx.notna().mean() >= 0.2).values]
mbx = mbx[mbx_list]

# Read in metadata for timing of samples
meta = pd.read_csv('../examples/iHMP/data/iHMP_metadata.csv')
# Subset metadata to appropriate samples and columns of interest
meta = meta[meta['External ID'].isin(mbx.index)][['External ID', 'Participant ID', 
                                                  'date_of_receipt', 'diagnosis',
                                                  'hbi', 'sccai', 'race', 'sex',
                                                  #'Age at diagnosis', 'consent_age',
                                                  #'site_name'
                                                 ]]
meta.set_index('External ID', inplace=True)

# Combine severity scores
meta['severity'] = np.where(meta['diagnosis'] == 'CD', meta.hbi, meta.sccai)
meta['severity'] = np.where(meta['diagnosis'] == 'nonIBD', 0, meta['severity'])

# Make sure that intensities are numeric and log transform them
log_mbx = np.log(mbx.astype('double'))

# Calculate the means and standard deviations of each column
log_means = log_mbx.mean()
log_stds = log_mbx.std()

# Transform dataset
log_mbx = (log_mbx - log_means)/log_stds

# Merge metabolomics info to metadata
df = (meta.merge(mbx, #log_mbx, 
                 left_index = True, 
           right_index = True).
      drop_duplicates().
      rename(columns = {'Participant ID': 'id',
                        'date_of_receipt': 'date'#,
#                         0: 'intensity'
                       }))

# Fix the date column
df.date = pd.to_datetime(df.date)

# Now scale the dates compared to the earliest date
min_date = df.date.min()
df['days_from_start'] = (df.date - min_date).dt.days

# Find the max severity day for each ID
max_date = df[['id', 'severity', 'date']].\
    dropna(subset=['severity']).\
    sort_values(['severity', 'date'], ascending=[True, False]).\
    groupby('id').\
    tail(1).\
    drop(columns=['severity']).\
    rename(columns={'date': 'max_severity_date'})

# Merge to get the median date for controls
max_date = max_date.merge(
    right=df.query("diagnosis == 'nonIBD'")[['id', 'date']].\
        groupby('id', as_index=False).\
        apply(lambda x: x.iloc[int(np.floor((len(x)+1)//2))]).\
        rename(columns={'date': 'med_severity_date'}),
    on='id',
    how='left'
)
# Use median date if available, otherwise use computed value
max_date['max_severity_date'] = max_date.med_severity_date.combine_first(max_date.max_severity_date)
# Drop unnecessary column
max_date.drop(columns=['med_severity_date'], inplace=True)

# Merge this back to df
df = pd.merge(df, max_date, on='id')
df['days_from_max_severity'] = (df.date - df.max_severity_date).dt.days
df = df.drop(columns=['max_severity_date'])

# Drop columns that aren't needed (drop id for the moment)
df = df.drop(columns=['date'])#, 'id'])

# Drop duplicate metabolites
df = df.loc[:,~df.columns.duplicated()]

# Only keep CD observations because they have hbi
# df = df.query("diagnosis == 'CD'").drop(columns = ['diagnosis'])

# Only keep non-missing severity for the moment
# df = df[df.hbi.notna()]

# Fill in severity scores for individuals, last carry forward 
df['severity'] = df[['id', 'severity']].\
    groupby('id').\
    fillna(method='ffill')
df = df[df.severity.notna()]
df = df.drop(columns=['hbi', 'sccai'])

# Drop UC individuals for now
df = df.query("diagnosis != 'UC'")
df = df.query("diagnosis == 'CD'")
df = df.drop(columns=['diagnosis'])

# Store individual information look up vectors
# Get numerics for each categorical value as well as the lookup index
df['id'], id_map = pd.factorize(df['id'])
# df['diagnosis'], diagnosis_map = pd.factorize(df['diagnosis'])
df['race'], race_map = pd.factorize(df['race'])
df['sex'], sex_map = pd.factorize(df['sex'])
# n_id = df.id.nunique()
# id_list = df.id.unique()
# id_vals = df.id.values
# id_idx = np.array([np.where(id_list == x)[0][0] for x in id_vals])
# df['id'] = id_idx

# Only keep metabolites 
df_original = df.copy()

# Standardize severity and days for convergence properties
df.days_from_start = (df.days_from_start - df.days_from_start.mean())/df.days_from_start.std()
df.days_from_max_severity = (df.days_from_max_severity - df.days_from_max_severity.mean())/df.days_from_max_severity.std()
# df.severity = (df.severity - df.severity.mean())/df.severity.std()

# Fill in zeros and transform input metabolite data
df_transform = df_original.copy()
df_transform.iloc[:, 5:-2] = (np.log(df.iloc[:, 5:-2].fillna(0) + 1) - np.log(df.iloc[:, 5:-2].fillna(0) + 1).mean())/np.log(df.iloc[:, 5:-2].fillna(0) + 1).std()
df_transform['days_from_start'] = df['days_from_start']

kernel_list = [gpflow.kernels.SquaredExponential(),
               gpflow.kernels.Matern12(),
               Lin(),
               Poly(),
               gpflow.kernels.ArcCosine(),
               gpflow.kernels.Periodic(base_kernel=gpflow.kernels.SquaredExponential())]

start_time = time.time()

severity_model = full_kernel_search(
    X=df_transform.drop(['severity', 'days_from_max_severity'], axis=1),
    Y=df.severity,
    kern_list=kernel_list,
    cat_vars=[0, 1, 2, 3],
    max_depth=5,
    early_stopping=True,
    prune=True,
    keep_all=False,
    # lik = 'gaussian',
    lik='poisson',
    verbose=True,
    random_seed=9102
)

end_time = time.time()
print("----%.2f seconds----"%(end_time - start_time))

# Save output
f = open("severity_model.pkl","wb")
pickle.dump(severity_model, f)
f.close()
