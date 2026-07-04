#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: manoli
"""

# NOTE: This file uses the ToMSyn ToM covariance as an example, but the process is the same for the Richardson ToM and ToMSyn action prediction analyses,
# except in the Richardson analyses we only run the first model in this script (ToM, sex, age, TCV), and in the ToMSyn action prediction analysis ap_score is 
# the main predictor and ToM_score a covariate.

# =============================================================================
# IMPORTS
# =============================================================================

import os
import numpy as np
import pandas as pd
import nibabel as nib
from nilearn.image import mean_img, resample_img, math_img
from nilearn.glm.second_level import SecondLevelModel
from nilearn.glm.thresholding import threshold_stats_img
from nilearn.plotting import plot_roi, plot_stat_map, plot_glass_brain
from nilearn.input_data import NiftiMasker


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
    
# Cerebellar GM images (SUIT space)
cereb_img_list = []
for sub in subjects:
    sub_dir = os.path.join(children_T1, sub, 'suit')
    file_path = os.path.join(sub_dir, f'wd{sub}_T1w_seg1.nii')  # SUIT-space modulated cerebellum GM
    if os.path.exists(file_path):
        cereb_img_list.append(file_path)
    else:
        print(f'File does not exist: {file_path}')

cereb_img_sorted = sorted(cereb_img_list)
n_subjects = len(cereb_img_sorted)
assert n_subjects > 0, "No SUIT cerebellar GM images found."

# Cortical GM images (MNI space)
cort_img_list = []
for sub in subjects:
    sub_dir = os.path.join(children_T1, sub, 'mri')
    file_path = os.path.join(sub_dir, f'mwp1{sub}_T1w.nii')  # MNI-space modulated cortex GM
    if os.path.exists(file_path):
        cort_img_list.append(file_path)
    else:
        print(f'File does not exist: {file_path}')

cort_img_sorted = sorted(cort_img_list)
n_subjects = len(cort_img_sorted)
assert n_subjects > 0, "No CAT12 cortical GM images found."


# =============================================================================
# TIV CALCULATION (WHOLE-BRAIN)
# =============================================================================


# Function to extract numeric subject ID for sorting
gm_img_list = []
wm_img_list = []
csf_img_list = []

for sub in subjects:
    sub_dir = os.path.join(children_T1, sub, 'mri')
    gm_path = os.path.join(sub_dir, f'mwp1{sub}_T1w.nii') # GM
    wm_path = os.path.join(sub_dir, f'mwp2{sub}_T1w.nii') # WM
    csf_path = os.path.join(sub_dir, f'mwp3{sub}_T1w.nii') # CSF

    if os.path.exists(gm_path) and os.path.exists(wm_path) and os.path.exists(csf_path):
        gm_img_list.append(gm_path)
        wm_img_list.append(wm_path)
        csf_img_list.append(csf_path)
    else:
        print(f"Missing GM or WM or CSF image for subject {sub}")

# Sort
gm_img_sorted = sorted(gm_img_list)
wm_img_sorted = sorted(wm_img_list)
csf_img_sorted = sorted(csf_img_list)

# Compute TIV
# TIV = Total Intracranial Volume, calculated here as the sum of CSF + grey matter + white matter tissue volumes
tiv_rows = []
threshold = 0.0001  # Exclude background zeros 

for gm_path, wm_path, csf_path in zip(gm_img_sorted, wm_img_sorted, csf_img_sorted):
    gm_img = nib.load(gm_path)
    wm_img = nib.load(wm_path)
    csf_img = nib.load(csf_path)

    gm_data = gm_img.get_fdata(dtype=np.float32)
    wm_data = wm_img.get_fdata(dtype=np.float32)
    csf_data = csf_img.get_fdata(dtype=np.float32)

    # Threshold very small values to ignore background
    gm_data = np.where(gm_data > threshold, gm_data, 0)
    wm_data = np.where(wm_data > threshold, wm_data, 0)
    csf_data = np.where(csf_data > threshold, csf_data, 0)

    # Voxel volume (mm^3) from the image header's voxel dimensions,
    # needed to convert summed tissue probabilities into a physical volume
    voxel_vol = np.prod(gm_img.header.get_zooms()[:3])
    tiv_mm3 = (gm_data + wm_data + csf_data).sum() * voxel_vol

    # Subject ID parsed from the filename (portion before the first underscore)
    subject_id = os.path.basename(gm_path).split('_')[0]
    tiv_rows.append({'subject': subject_id, 'TIV_mm3': tiv_mm3})

# Create dataframe and mean-center
tiv_df = pd.DataFrame(tiv_rows)
tiv_df['TIV_c'] = tiv_df['TIV_mm3'] - tiv_df['TIV_mm3'].mean()

print(tiv_df.head())


# =============================================================================
# ROI SETUP
# =============================================================================


# ROI extraction helpers
def make_sphere_roi(center_mm, radius_mm, ref_img):
    affine = ref_img.affine
    shape = ref_img.shape[:3]
    # Create voxel coordinate grid
    i, j, k = np.meshgrid(np.arange(shape[0]), 
                          np.arange(shape[1]), 
                          np.arange(shape[2]), 
                          indexing='ij')
    voxel_coords = np.column_stack([i.ravel(), j.ravel(), k.ravel()])
    coords_mm = nib.affines.apply_affine(affine, voxel_coords)
    dist = np.linalg.norm(coords_mm - center_mm, axis=1)
    mask = (dist <= radius_mm).astype(np.uint8).reshape(shape)
    return nib.Nifti1Image(mask, affine)

def binarize(img, thr=0.5):
    dat = img.get_fdata()
    out = (dat >= thr).astype(np.uint8)
    return nib.Nifti1Image(out, img.affine, img.header)

# Cerebellum coordinates (SUIT space, mm)
roi_coords_cereb = {
    "lcrus2": (-14, -84, -39), # Tomsyn VBM
    "rcrus2": (24, -80, -39) # Richardson VBM
}
cereb_radius = 5 # mm 


# =============================================================================
# DEFINE CEREBELLUM ROIS
# =============================================================================


# Load reference image for cerebellum
ref_img_cereb = nib.load(cereb_img_sorted[0])

# Define masks
cereb_masks = {}
for roi_name, coord in roi_coords_cereb.items():
    roi_mask_img_cereb = make_sphere_roi(coord, cereb_radius, ref_img_cereb)
    cereb_masks[roi_name] = roi_mask_img_cereb
    # QC plot
    plot_roi(roi_mask_img_cereb, bg_img=ref_img_cereb,
             title=f"Cerebellum ROI check: {roi_name}",
             display_mode="ortho", cut_coords=coord)


# Build masker to extract data from cerebellum GM maps

cereb_maskers = {
    name: NiftiMasker(mask_img=mask, detrend=False, standardize=False)
    for name, mask in cereb_masks.items()
}


# =============================================================================
# EXTRACT ROI DATA
# =============================================================================

out_dir = os.path.join(data_dir, "results/tom/tomsyn/covariance")
os.makedirs(out_dir, exist_ok=True)

rows = []
for sid, sp, cp in zip(subjects, cereb_img_sorted, cort_img_sorted):
    row = {"subject": sid}

    # Cerebellum data extraction
    suit_img = nib.load(sp)  
    for name, masker in cereb_maskers.items():
        vals = masker.fit_transform(suit_img)   # shape: (1, n_voxels_in_mask)
        row[name] = float(vals.mean())  # mean GM within SUIT mask


    rows.append(row)

roi_df = pd.DataFrame(rows)

# Add covariates
data_beh = roi_df.merge(all_children[["subject", "ToM_score", "sex", "age",
                                 "EF_Score", "SETK_MW_T", "ZK_ABC_sum", "g", "ap_score"]], 
                    on="subject", how="left")

tiv_df['subject'] = tiv_df['subject'].str.replace(r'^mwp1', '', regex=True)
covar_data = data_beh.merge(tiv_df[["subject", "TIV_c"]], on="subject", how="left") # Mean-centered TIV

# Mean center continuous variables
covar_data["ToM_c"] = covar_data["ToM_score"] - covar_data["ToM_score"].mean()
covar_data["age_c"] = covar_data["age"] - covar_data["age"].mean()
covar_data["ef_c"] = covar_data["EF_Score"] - covar_data["EF_Score"].mean()
covar_data["lang_c"] = covar_data["SETK_MW_T"] - covar_data["SETK_MW_T"].mean()
covar_data["g_c"] = covar_data["ZK_ABC_sum"] - covar_data["ZK_ABC_sum"].mean()
covar_data["ap_c"] = covar_data["ap_score"] - covar_data["ap_score"].mean()

covar_data_csv = os.path.join(out_dir, 'tomsyn_covar_data.csv')
covar_data.to_csv(covar_data_csv, index=False)

print(covar_data.head())


# =============================================================================
# BUILD CORTEX MASK
# =============================================================================

# Build a cortex mask to only run interaction models in the cortex

ref_img = nib.load(cort_img_list[0])

cort_imgs_sorted = []
for p in cort_img_sorted:
    img = nib.load(p)
    cort_imgs_sorted.append(img)

gm_mean = mean_img(cort_imgs_sorted)
gm_mask = math_img("img > 0.1", img=gm_mean)  # Slightly conservative mask 

# Only include cortex
atl = os.path.join(data_dir, 'Schaefer2018_100Parcels_7Networks_order_FSLMNI152_2mm.nii.gz')
cort_atlas = nib.load(atl)

# Resample cortex mask to exact GM mask grid (nearest for labels/masks)
atl_on_gm = resample_img(cort_atlas, target_affine=gm_mask.affine, target_shape=gm_mask.shape,
    interpolation="nearest")

# Binarize
atl_mask = math_img("img > 0", img=atl_on_gm)

# Create mask
cortex_mask = math_img("(gm > 0) & (cort > 0)", gm=gm_mask, cort=atl_mask)
plot_stat_map(cortex_mask, bg_img=ref_img, display_mode="ortho",
              title="Cortex mask = GM>0.2 minus cerebellum")


# =============================================================================
# INTERACTION MODELS
# =============================================================================


# Create dataframe for design matrix

dm_base = pd.DataFrame({"subject": subjects}).merge(
    covar_data, on="subject", how="left")

# Specify cerebellar ROIs
cereb_roi_names = ['rcrus2', 'lcrus2']

# Specify models
model_specs = [
    ("base",       []),  # Cereb_c, ToM_c, Intercept only
    ("SexAgeTIV", ["sex", "TIV_c", "age_c"]),
    ("all",   ["sex", "TIV_c", "ef_c", "lang_c", "ap_c", "g_c", "age_c"]),
]

# Fit interaction models
summary_rows = []

for cereb in cereb_roi_names:
    if cereb not in dm_base.columns:
        print(f"[SKIP] '{cereb}' not found in data.")
        continue

    # Start from base each time
    dm = dm_base.copy()
    # Center cerebellar predictor
    dm[f"{cereb}_c"] = dm[cereb] - dm[cereb].mean()
    # Interaction
    dm["Int"] = dm[f"{cereb}_c"] * dm["ToM_c"]
    # Intercept
    dm["Intercept"] = 1.0

    for model_name, terms in model_specs:
        # Base columns
        X = dm[["Intercept", f"{cereb}_c", "ToM_c", "Int"]].copy()

        # Add covariates 
        for cov in terms:
            if cov in dm.columns:
                X[cov] = dm[cov]
            else:
                print(f"[WARN] covariate '{cov}' not found in dm; skipping.")

        # Fit second-level model on cortex only
        slm = SecondLevelModel(mask_img=cortex_mask, smoothing_fwhm=5, minimize_memory=False)
        slm = slm.fit(cort_imgs_sorted, design_matrix=X)

        # Interaction contrast
        z_map = slm.compute_contrast("Int", output_type="z_score")
        thr_map, thr = threshold_stats_img(
            z_map, alpha=0.001, height_control="fdr", two_sided=True
        )

        # Save
        base = f"tomsyn_covar_{model_name}_{cereb}"
        z_path   = os.path.join(out_dir, f"{base}_unthresh.nii.gz")
        thr_path = os.path.join(out_dir, f"{base}_thresh_fdr.nii.gz")
        nib.save(z_map, z_path)
        nib.save(thr_map, thr_path)

        # Glass brain plot of the interaction 
        glass_path = os.path.join(
            out_dir, f"tomsyn_tom_covar_{model_name}_{cereb}_fdr.png"
        )
        display = plot_glass_brain(
            thr_map,
            cmap='hot',
            colorbar=True,
            display_mode="lyr",
            vmin=0,
            vmax=5
        )
        display.savefig(glass_path, dpi=600)
        display.close()

        summary_rows.append({
            "model": model_name,
            "cereb": cereb,
            "n_subj": len(X),
            "cols": ",".join(X.columns),
            "z_map": z_path,
            "z_map_thr": thr_path,
            "glass_brain": glass_path,
            "alpha_fdr": 0.05,
            "threshold_used": float(thr) if thr is not None else None,
        })

# Save summary
summary_df = pd.DataFrame(summary_rows)
summary_csv = os.path.join(out_dir, "tomsyn_expl_voxwise_covar_summary.csv")
summary_df.to_csv(summary_csv, index=False)
print(f"[OK] Saved voxelwise interaction maps & summary: {summary_csv}")