#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: manoli
"""

# NOTE: This file uses the ToMSyn ToM VBM as an example, but the process is the same for the Richardson ToM and ToMSyn action prediction analyses,
# except in the Richardson analyses we only run the first model in this script (ToM, sex, age, TCV), and in the ToMSyn action prediction analysis ap_score is 
# the main predictor and ToM_score a covariate.

# =============================================================================
# IMPORTS
# =============================================================================

import os                                             
import pandas as pd                                    
import numpy as np                                    
import nibabel as nib                                  
from nilearn.glm.second_level import SecondLevelModel  
from nilearn.glm import threshold_stats_img
from nilearn import plotting               
import SUITPy.flatmap as flatmap                      
import matplotlib.pyplot as plt                        

# =============================================================================
# SETUP
# =============================================================================

# Root data directory 
data_dir = '/data/cereb_tom_anat'

# Load behavioral data
all_children = pd.read_csv(os.path.join(data_dir, "tomsyn", "tomsyn_behav.csv"))

# Load SUIT-space images
children_T1 = os.path.join(data_dir, "tomsyn", "T1w")

sub_list = os.path.join(children_T1, "tomsyn_sub_list.txt")
with open(sub_list, 'r') as file:
    subjects = [line.strip() for line in file.readlines()]

cereb_img_list = []

for sub in subjects:
    sub_dir = os.path.join(children_T1, sub)
    file_path = os.path.join(sub_dir, f'wd{sub}_T1w_seg1.nii') # SUIT-space modulated GM
    # Check if file exists
    if os.path.exists(file_path):
        cereb_img_list.append(file_path)
    else:
        print(f'File does not exist: {file_path}')

# Sort images so that the covariate values are assigned to the correct subject
cereb_img_sorted = sorted(cereb_img_list)

# Total number of subjects 
n_subjects = len(cereb_img_sorted)


# =============================================================================
# CALCULATE TOTAL CEREBELLUM VOLUME
# =============================================================================

# Function to extract numeric subject ID for sorting
gm_img_list = []
wm_img_list = []

for sub in subjects:
    sub_dir = os.path.join(children_T1, sub)
    gm_path = os.path.join(sub_dir, f'wd{sub}_T1w_seg1.nii')   # GM
    wm_path = os.path.join(sub_dir, f'wd{sub}_T1w_seg2.nii')   # WM

    if os.path.exists(gm_path) and os.path.exists(wm_path):
        gm_img_list.append(gm_path)
        wm_img_list.append(wm_path)
    else:
        print(f"Missing GM or WM image for subject {sub}")

# Sort
gm_img_sorted = sorted(gm_img_list)
wm_img_sorted = sorted(wm_img_list)

# Compute TCV
# TCV = Total Cerebellar Volume, calculated here as the sum of grey matter + white matter tissue volumes
tcv_rows = []
threshold = 0.0001  # Exclude background zeros

for gm_path, wm_path in zip(gm_img_sorted, wm_img_sorted):
    gm_img = nib.load(gm_path)
    wm_img = nib.load(wm_path)

    gm_data = gm_img.get_fdata(dtype=np.float32)
    wm_data = wm_img.get_fdata(dtype=np.float32)

    # Threshold very small values to ignore background
    gm_data = np.where(gm_data > threshold, gm_data, 0)
    wm_data = np.where(wm_data > threshold, wm_data, 0)

    # Voxel volume (mm^3) from the image header's voxel dimensions,
    # needed to convert summed tissue probabilities into a physical volume
    voxel_vol = np.prod(gm_img.header.get_zooms()[:3])
    tcv_mm3 = (gm_data + wm_data).sum() * voxel_vol

    # Subject ID parsed from the filename (portion before the first underscore)
    subject_id = os.path.basename(gm_path).split('_')[0]
    tcv_rows.append({'subject': subject_id, 'tcv_mm3': tcv_mm3})

# Create dataframe and mean-center
tcv_df = pd.DataFrame(tcv_rows)
tcv_df['tcv_c'] = tcv_df['tcv_mm3'] - tcv_df['tcv_mm3'].mean()

print(tcv_df.head())


# =============================================================================
# SPECIFY COVARIATES
# =============================================================================

# Continuous predictors are mean-centered 

# ToM
tom_score = all_children["ToM_score"].astype(float)
tom_score = tom_score - tom_score.mean()

# Sex (binary)
sex = all_children["sex"]

# Age
age = all_children["age"]
age = age - age.mean()

# TCV (already mean-centered)
tcv_c = tcv_df['tcv_c'].values

# Executcve function
ef_score = all_children["EF_Score"]
ef_score = ef_score - ef_score.mean()

# Language
lang_score = all_children["SETK_MW_T"]
lang_score = lang_score - lang_score.mean()

# General cognitcve abilities
g = all_children["ZK_ABC_sum"]
g = g - g.mean()

# Action prediction
ap_score = all_children["ap_score"]
ap_score = ap_score - ap_score.mean()

# Intercept
intercept = np.ones(len(cereb_img_sorted))


# =============================================================================
# SECOND-LEVEL MODELS
# =============================================================================

# Run two models, one with sex, age, and TCV as covariates, and a full model with
# action prediction and general cognitive and linguistic abilities.

out_dir = os.path.join(data_dir, "results/tom/tomsyn/vbm")
os.makedirs(out_dir, exist_ok=True)


model_specs = [
    {
        # Specify sex, age, TCV model
        "columns": ["ToM", "sex", "age", "tcv", "intercept"],
        "values": [tom_score, sex, age, tcv_c, intercept],
        # Model outputs
        "unthresh_file": os.path.join(out_dir, "tomsyn_vbm_covs_SexAgeTCV_unthresh.nii.gz"),
        "fdr_file": os.path.join(out_dir, "tomsyn_vbm_covs_SexAgeTCV_fdr.nii.gz"),
        "flatmap_file": os.path.join(out_dir, "tomsyn_vbm_covs_SexAgeTCV_fdr.png"),
    },
    {
        # Specify full model
        "columns": ["ToM", "sex", "age", "tcv", "ef_score", "lang_score", "ap_score", "g", "intercept"],
        "values": [tom_score, sex, age, tcv_c, ef_score, lang_score, ap_score, g, intercept],
         # Model outputs
        "unthresh_file": os.path.join(out_dir, "tomsyn_vbm_all_covs_unthresh.nii.gz"),
        "fdr_file": os.path.join(out_dir, "tomsyn_vbm_all_covs_fdr.nii.gz"),
        "flatmap_file": os.path.join(out_dir, "tomsyn_vbm_all_covs_fdr.png"),
    },
]

for spec in model_specs:

    # ---- Design matrix ----
    design_matrix_covs = pd.DataFrame(
        np.vstack(spec["values"]).T,
        columns=spec["columns"])

    # Visualize the design matrix to sanity-check covariate values/ordering
    fig, ax = plt.subplots(1, 1, figsize=(4, 8))
    plotting.plot_design_matrix(design_matrix_covs, ax=ax)
    ax.set_ylabel("subjects")
    plt.show()

    # ---- 2nd Level GLM ----
    second_level_model = SecondLevelModel(smoothing_fwhm=5.0).fit(
        cereb_img_sorted, design_matrix=design_matrix_covs)

    z_map = second_level_model.compute_contrast(
        second_level_contrast='ToM',  # Use the column name from design matrix
        output_type="z_score")

    # ---- FDR Correction ----
    thresholded_map, threshold = threshold_stats_img(
        z_map, alpha=.001, height_control='fdr', two_sided=True)

    # ---- Save images ----
    nib.save(z_map, spec["unthresh_file"])
    nib.save(thresholded_map, spec["fdr_file"])

    # ---- Plot on flatmap ----
    fdr_data = flatmap.vol_to_surf(thresholded_map)
    print('Output is a np.array of size:', fdr_data.shape)

    flatmap.plot(data=fdr_data, cmap='autumn',
        threshold = [0, 3],
        new_figure=True,
        colorbar=True,
        render='matplotlib')

    # Save flatmap
    plt.savefig(spec["flatmap_file"])

    plt.show()