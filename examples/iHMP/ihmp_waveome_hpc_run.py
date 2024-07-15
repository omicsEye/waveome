# Libraries
import pickle

import numpy as np
import pandas as pd

from waveome.model_search import GPSearch

# Data Read
mbx = pd.read_csv(
    "../iHMP/data/iHMP_labeled_metabolomics.csv", low_memory=False
)
mtd = pd.read_csv("../iHMP/data/iHMP_metadata.csv", low_memory=False)

# Print out sizes

# Clean up
# Save separate lookup table
mbx_lookup = mbx[["HMDB (*Representative ID)", "Metabolite", "Compound"]]
mbx = (
    mbx
    # Drop all column identifiers that aren't necessary for search
    .drop(columns=mbx.columns[:6])
    # Prepare to transpose dataframe
    .set_index("Compound")
    # Make sure we have observation (row) by metabolite (column)
    .transpose()
    # Pull off index for matching
    .reset_index(names="External ID")
    # Impute zero for missing intensity value
    .fillna(0.0)
    .set_index("External ID")
    .sort_index()
)

pd.set_option("display.max_columns", 500)
"""
Metadata column selection:
- External ID (sample ID)
- Participant ID (person ID)
- date_of_receipt
- site_name (location)
- Age at diagnosis 
- consent_age
- diagnosis
- General wellbeing
- Abdominal pain
- Number of liquid or very soft stools in the past 24 hours:
- Arthralgia (joint pain)
- hbi
- Hispanic or Latino Origin
- BMI
- Height
- Weight.1
- Cancer columns
- Subject was diagnosed within the last 6 months
- Age at diagnosis (A)
- race
- Blood in the stool
- General well being over the past 24 hours
- sccai
- sex
- smoking status
"""

column_list = [
    "External ID",
    "Participant ID",
    "site_name",
    "Age at diagnosis",
    "consent_age",
    "diagnosis",
    "General wellbeing",
    "hbi",
    "Hispanic or Latino Origin",
    "date_of_receipt",
    "race",
    "General well being over the past 24 hours",
    "sccai",
    "sex",
]

# Now prepare metadata set
mtd_sub = (
    mtd.loc[mtd["data_type"].values == "metabolomics", column_list]
    .rename(
        columns={
            "Participant ID": "participant_id",
            "Age at diagnosis": "age_at_diagnosis",
            "Hispanic or Latino Origin": "hispanic",
        }
    )
    .assign(disease_years=lambda x: x["consent_age"] - x["age_at_diagnosis"])
    .assign(
        study_days=lambda x: (
            pd.to_datetime(x["date_of_receipt"])
            - pd.to_datetime(x["date_of_receipt"]).min()
        ).dt.days  # .astype('timedelta64[D]')
    )
    .assign(age=lambda x: x["consent_age"] + (x["study_days"] / 365.0))
    .assign(
        general_wellbeing=lambda x: x["General wellbeing"].combine_first(
            x["General well being over the past 24 hours"]
        )
    )
    .assign(
        severity=lambda x: np.where(
            x["diagnosis"] == "CD", x["hbi"], x["sccai"]
        )
    )
    .assign(active_disease=lambda x: np.where(x["severity"] >= 5, "1", "0"))
    .assign()
    .drop(
        columns=[
            "General wellbeing",
            "General well being over the past 24 hours",
            "hbi",
            "sccai",
            "date_of_receipt",
        ]
    )
    .set_index("External ID")
    .sort_index()
)

# Now we want to run the model search process
# Get missing indicators
x_miss_idx = (
    mtd_sub[
        [
            "participant_id",
            "site_name",
            "age_at_diagnosis",
            "age",
            "diagnosis",
            "race",
            "sex",
            "general_wellbeing",
            "active_disease",
        ]
    ]
    .isna()
    .sum(axis=1)
    > 0
)

# Load up information
gps = GPSearch(
    X=mtd_sub.loc[~x_miss_idx, :][
        [
            "participant_id",
            "site_name",
            "age_at_diagnosis",
            "age",
            "diagnosis",
            "race",
            "sex",
            "general_wellbeing",
            "active_disease",
        ]
    ],
    Y=mbx.loc[~x_miss_idx, :],
    unit_col="participant_id",
    categorical_vars=[
        "site_name",
        "diagnosis",
        "race",
        "sex",
        "general_wellbeing",
        "active_disease",
    ],
    outcome_likelihood="negativebinomial",
)

# Run search
gps.run_search(random_seed=9102, num_jobs=40)

# Save output in pickle file
with open("ihmp_waveome_output.pickle", "wb") as handle:
    pickle.dump(gps, handle, protocol=pickle.HIGHEST_PROTOCOL)
