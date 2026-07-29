#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: manoli
"""

# NOTE: This script follows the ToMNAP action prediction VBM analysis. The model includes
# action prediction, sex, age, total cerebellar volume, and an intercept.

# =============================================================================
# IMPORTS
# =============================================================================

import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import SUITPy.flatmap as flatmap
from nilearn.glm import threshold_stats_img
from nilearn.glm.second_level import SecondLevelModel
from nilearn.image import concat_imgs, index_img, math_img
from scipy.ndimage import label


# =============================================================================
# SETUP
# =============================================================================

# Root data directory
data_dir = '/data/cereb_tom_anat'

# Load behavioral data
all_children = pd.read_csv(os.path.join(data_dir, "tomnap", "tomnap_behav.csv"))

# Load SUIT-space images
children_T1 = os.path.join(data_dir, "tomnap", "T1w")
sub_list = os.path.join(children_T1, "tomnap_sub_list.txt")
with open(sub_list, 'r') as file:
    subjects = [line.strip() for line in file.readlines()]

cereb_img_list = []
for sub in subjects:
    sub_dir = os.path.join(children_T1, sub, 'suit')
    file_path = os.path.join(sub_dir, f'wd{sub}_T1w_seg1.nii')
    if os.path.exists(file_path):
        cereb_img_list.append(file_path)
    else:
        print(f'File does not exist: {file_path}')

# Sort images so that the covariate values are assigned to the correct subject
cereb_img_sorted = sorted(cereb_img_list)

# Total number of subjects
n_subjects = len(cereb_img_sorted)

# Resampling settings
n_boot = 2000
random_seed = 2026
smoothing_fwhm = 5.0
fdr_alpha = .05

# Output directories and original thresholded result
out_dir = os.path.join(data_dir, "results/nap/tomnap/vbm")
loo_dir = os.path.join(out_dir, "loo")
bootstrap_dir = os.path.join(out_dir, "bootstrap")
original_map_path = os.path.join(out_dir, "tomnap_vbm_covs_SexAgeTCV_fdr.nii.gz")
figure_path = os.path.join(out_dir, "tomnap_vbm_covs_SexAgeTCV_resampling_stability.png")
os.makedirs(loo_dir, exist_ok=True)
os.makedirs(bootstrap_dir, exist_ok=True)


# =============================================================================
# CALCULATE TOTAL CEREBELLUM VOLUME
# =============================================================================

# Collect gray- and white-matter images
gm_img_list = []
wm_img_list = []
for sub in subjects:
    sub_dir = os.path.join(children_T1, sub, 'suit')
    gm_path = os.path.join(sub_dir, f'wd{sub}_T1w_seg1.nii')
    wm_path = os.path.join(sub_dir, f'wd{sub}_T1w_seg2.nii')
    if os.path.exists(gm_path) and os.path.exists(wm_path):
        gm_img_list.append(gm_path)
        wm_img_list.append(wm_path)
    else:
        print(f'Missing GM or WM image for subject {sub}')

# Sort images so that gray- and white-matter maps remain aligned
gm_img_sorted = sorted(gm_img_list)
wm_img_sorted = sorted(wm_img_list)

# TCV is the sum of gray- and white-matter tissue volumes
tcv_rows = []
threshold = 0.0001
for gm_path, wm_path in zip(gm_img_sorted, wm_img_sorted):
    gm_img = nib.load(gm_path)
    wm_img = nib.load(wm_path)
    gm_data = np.where(gm_img.get_fdata(dtype=np.float32) > threshold, gm_img.get_fdata(dtype=np.float32), 0)
    wm_data = np.where(wm_img.get_fdata(dtype=np.float32) > threshold, wm_img.get_fdata(dtype=np.float32), 0)
    voxel_vol = np.prod(gm_img.header.get_zooms()[:3])
    tcv_mm3 = (gm_data + wm_data).sum() * voxel_vol
    subject_id = os.path.basename(gm_path).split('_')[0]
    tcv_rows.append({'subject': subject_id, 'tcv_mm3': tcv_mm3})

# Create dataframe and mean-center TCV
tcv_df = pd.DataFrame(tcv_rows)
tcv_df['tcv_c'] = tcv_df['tcv_mm3'] - tcv_df['tcv_mm3'].mean()


# =============================================================================
# SPECIFY COVARIATES
# =============================================================================

# Continuous predictors are mean-centered

# NAP
ap_score = all_children["ap_score"].astype(float)
ap_score = ap_score - ap_score.mean()

# Sex
sex = all_children["sex"]

# Age
age = all_children["age"]
age = age - age.mean()

# TCV
tcv_c = tcv_df['tcv_c'].values

# Intercept
intercept = np.ones(n_subjects)

# Design matrix used in the original ToMNAP ToM VBM
design_matrix_covs = pd.DataFrame(np.vstack([ap_score, sex, age, tcv_c, intercept]).T, columns=["NAP", "sex", "age", "tcv", "intercept"])

# Concatenate subject-level gray-matter maps once for resampling
cereb_4d = concat_imgs(cereb_img_sorted)

# Fit the full model once to obtain a common analysis mask
second_level_model = SecondLevelModel(smoothing_fwhm=smoothing_fwhm).fit(cereb_4d, design_matrix=design_matrix_covs)
analysis_mask = second_level_model.masker_.mask_img_

# =============================================================================
# LEAVE-ONE-OUT ANALYSIS
# =============================================================================

# Refit the model after omitting each participant once
loo_z_maps = []
loo_thresholded_maps = []
loo_thresholds = []
for i, sub in enumerate(subjects):
    keep = np.arange(n_subjects) != i
    loo_model = SecondLevelModel(smoothing_fwhm=smoothing_fwhm, mask_img=analysis_mask).fit(index_img(cereb_4d, keep), design_matrix=design_matrix_covs.loc[keep].reset_index(drop=True))
    loo_z_map = loo_model.compute_contrast(second_level_contrast='NAP', output_type='z_score')
    loo_thresholded_map, loo_threshold = threshold_stats_img(loo_z_map, alpha=fdr_alpha, height_control='fdr', two_sided=True)
    loo_z_maps.append(loo_z_map)
    loo_thresholded_maps.append(loo_thresholded_map)
    loo_thresholds.append(loo_threshold)
    nib.save(loo_z_map, os.path.join(loo_dir, f'loo_without_{sub}_unthresh.nii.gz'))
    nib.save(loo_thresholded_map, os.path.join(loo_dir, f'loo_without_{sub}_fdr.nii.gz'))
    print(f'{i + 1}/{n_subjects}: omitted {sub}')

# Summarize voxel-wise effect size, sign consistency, and FDR inclusion
loo_z_data = concat_imgs(loo_z_maps).get_fdata()
loo_thresholded_data = concat_imgs(loo_thresholded_maps).get_fdata()
reference_img = loo_z_maps[0]
loo_maps = {
    'loo_mean_z.nii.gz': np.mean(loo_z_data, axis=3),
    'loo_sd_z.nii.gz': np.std(loo_z_data, axis=3),
    'loo_positive_frequency.nii.gz': np.mean(loo_z_data > 0, axis=3),
    'loo_negative_frequency.nii.gz': np.mean(loo_z_data < 0, axis=3),
    'loo_fdr_inclusion_frequency.nii.gz': np.mean(loo_thresholded_data != 0, axis=3)
}
for filename, data in loo_maps.items():
    nib.save(nib.Nifti1Image(data, reference_img.affine, reference_img.header), os.path.join(loo_dir, filename))

pd.DataFrame({'omitted_subject': subjects, 'fdr_threshold': loo_thresholds}).to_csv(os.path.join(loo_dir, 'loo_thresholds.csv'), index=False)


# =============================================================================
# LEAVE-ONE-OUT CLUSTER STABILITY
# =============================================================================

# Restrict stability summaries to positive voxels in the original FDR map
original_img = nib.load(original_map_path)
original_data = original_img.get_fdata()
original_positive_mask = original_data > 0
positive_frequency_data = nib.load(os.path.join(loo_dir, 'loo_positive_frequency.nii.gz')).get_fdata()
fdr_frequency_data = nib.load(os.path.join(loo_dir, 'loo_fdr_inclusion_frequency.nii.gz')).get_fdata()
original_frequencies = positive_frequency_data[original_positive_mask]

print('\nall original positive voxels')
print('mean positive frequency:', original_frequencies.mean())
print('median positive frequency:', np.median(original_frequencies))
print('minimum positive frequency:', original_frequencies.min())
print(f'proportion positive in all {n_subjects} LOO analyses:', np.mean(original_frequencies == 1))

# Identify connected clusters and summarize stability within each cluster
cluster_labels, n_clusters = label(original_positive_mask, structure=np.ones((3, 3, 3)))
cluster_results = []
for cluster_id in range(1, n_clusters + 1):
    cluster_mask = cluster_labels == cluster_id
    frequencies = positive_frequency_data[cluster_mask]
    peak_voxel = np.unravel_index(np.argmax(np.where(cluster_mask, original_data, -np.inf)), original_data.shape)
    peak_mni = nib.affines.apply_affine(original_img.affine, peak_voxel)
    cluster_results.append({'cluster': cluster_id, 'n_voxels': cluster_mask.sum(), 'peak_x': peak_mni[0], 'peak_y': peak_mni[1], 'peak_z': peak_mni[2], 'mean_positive_frequency': frequencies.mean(), 'median_positive_frequency': np.median(frequencies), 'minimum_positive_frequency': frequencies.min(), 'proportion_positive_all_loo': np.mean(frequencies == 1), 'proportion_positive_at_least_95pct': np.mean(frequencies >= .95)})

cluster_results = pd.DataFrame(cluster_results).sort_values('n_voxels', ascending=False).reset_index(drop=True)
cluster_results.to_csv(os.path.join(loo_dir, 'loo_stability_within_original_clusters.csv'), index=False)
print('\nloo stability within original clusters')
print(cluster_results.to_string(index=False))
print('\nmean FDR-inclusion frequency:', fdr_frequency_data[original_positive_mask].mean())


# =============================================================================
# BOOTSTRAP ANALYSIS 
# =============================================================================

rng = np.random.default_rng(random_seed)
img_shape = cereb_4d.shape[:3]
sum_z = np.zeros(img_shape)
sum_z_squared = np.zeros(img_shape)
positive_count = np.zeros(img_shape, dtype=np.int32)
fdr_count = np.zeros(img_shape, dtype=np.int32)
bootstrap_thresholds = []

# Resample participants with replacement and accumulate voxel-wise stability measures
for bootstrap in range(n_boot):
    boot_idx = rng.choice(n_subjects, size=n_subjects, replace=True)
    boot_model = SecondLevelModel(smoothing_fwhm=smoothing_fwhm, mask_img=analysis_mask).fit(index_img(cereb_4d, boot_idx), design_matrix=design_matrix_covs.iloc[boot_idx].reset_index(drop=True))
    boot_z_map = boot_model.compute_contrast(second_level_contrast='NAP', output_type='z_score')
    boot_thresholded_map, boot_threshold = threshold_stats_img(boot_z_map, alpha=fdr_alpha, height_control='fdr', two_sided=True)
    z_data = boot_z_map.get_fdata()
    sum_z += z_data
    sum_z_squared += z_data ** 2
    positive_count += z_data > 0
    fdr_count += boot_thresholded_map.get_fdata() > 0 
    bootstrap_thresholds.append(boot_threshold)
    if (bootstrap + 1) % 100 == 0:
        print(f'{bootstrap + 1}/{n_boot} bootstrap samples completed')

# Save voxel-wise bootstrap summaries
bootstrap_mean_z = sum_z / n_boot
bootstrap_sd_z = np.sqrt(sum_z_squared / n_boot - bootstrap_mean_z ** 2)
bootstrap_positive_frequency = positive_count / n_boot
bootstrap_fdr_frequency = fdr_count / n_boot
bootstrap_maps = {
    'bootstrap_mean_z.nii.gz': bootstrap_mean_z,
    'bootstrap_sd_z.nii.gz': bootstrap_sd_z,
    'bootstrap_positive_frequency.nii.gz': bootstrap_positive_frequency,
    'bootstrap_fdr_inclusion_frequency.nii.gz': bootstrap_fdr_frequency
}
for filename, data in bootstrap_maps.items():
    nib.save(nib.Nifti1Image(data, reference_img.affine, reference_img.header), os.path.join(bootstrap_dir, filename))

pd.DataFrame({'bootstrap': np.arange(1, n_boot + 1), 'fdr_threshold': bootstrap_thresholds}).to_csv(os.path.join(bootstrap_dir, 'bootstrap_thresholds.csv'), index=False)

# Summarize bootstrap stability within positive voxels from the original result
cluster_positive_frequency = bootstrap_positive_frequency[original_positive_mask]
cluster_fdr_frequency = bootstrap_fdr_frequency[original_positive_mask]
print('\nbootstrap positive-frequency stability')
print('mean:', cluster_positive_frequency.mean())
print('median:', np.median(cluster_positive_frequency))
print('minimum:', cluster_positive_frequency.min())
print('proportion positive in at least 95% of bootstraps:', np.mean(cluster_positive_frequency >= .95))
print('proportion positive in all bootstraps:', np.mean(cluster_positive_frequency == 1))
print('\nbootstrap FDR-inclusion stability')
print('mean:', cluster_fdr_frequency.mean())
print('median:', np.median(cluster_fdr_frequency))
print('minimum:', cluster_fdr_frequency.min())


# =============================================================================
# VISUALIZE RESAMPLING STABILITY 
# =============================================================================

# Project LOO and bootstrap positive-frequency maps onto the SUIT flatmap
loo_positive_img = nib.load(os.path.join(loo_dir, 'loo_positive_frequency.nii.gz'))
bootstrap_positive_img = nib.load(os.path.join(bootstrap_dir, 'bootstrap_positive_frequency.nii.gz'))
loo_masked_img = math_img('np.where(original > 0, stability, np.nan)', original=original_img, stability=loo_positive_img)
bootstrap_masked_img = math_img('np.where(original > 0, stability, np.nan)', original=original_img, stability=bootstrap_positive_img)
loo_surface = flatmap.vol_to_surf(loo_masked_img)
bootstrap_surface = flatmap.vol_to_surf(bootstrap_masked_img)

# Plot both stability maps with a shared color bar
fig = plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
flatmap.plot(data=loo_surface, cmap='autumn', cscale=[.50, 1], new_figure=False, colorbar=False, render='matplotlib')
plt.title('a. leave-one-out stability', fontsize=13)
plt.subplot(1, 2, 2)
flatmap.plot(data=bootstrap_surface, cmap='autumn', cscale=[.50, 1], new_figure=False, colorbar=False, render='matplotlib')
plt.title('b. bootstrap stability', fontsize=13)
color_map = mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=.50, vmax=1), cmap='autumn')
color_map.set_array([])
cbar = fig.colorbar(color_map, ax=fig.axes, orientation='horizontal', fraction=.05, pad=.08, aspect=35)
cbar.set_label('proportion of resampled analyses with a positive association', fontsize=11)
plt.savefig(figure_path, dpi=300, bbox_inches='tight')
plt.show()
