#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: manoli
"""

# =============================================================================
# SETUP
# =============================================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nibabel as nib
import matplotlib.cm as cm
import SUITPy.flatmap as flatmap
from nilearn.plotting import plot_glass_brain
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

# Root data directory 
data_dir = '/data/cereb_tom_anat'


# =============================================================================
# FLATMAP PLOTTING SETTINGS
# =============================================================================

bg_color = (1, 1, 1, 0)  # Transparent white at the lower end of all colormaps

grays_mild = LinearSegmentedColormap.from_list(
    'grays_mild',
    [(1, 1, 1, 1), (0.4, 0.4, 0.4, 1)])  # Lower value for grayer bg

parcel_base_color = list(cm.get_cmap('tab10')(9))  # Custom color

res_color = list(cm.get_cmap('Set1')(5))  # Custom color
res_color[3] = 0.4  # Opacity
res_cmap = LinearSegmentedColormap.from_list('pos', [bg_color, res_color])


# =============================================================================
# GLASS BRAIN PLOTTING SETTINGS
# =============================================================================

net_color = cm.get_cmap('tab10')(9)
cmap_net = ListedColormap([net_color])

glass_res_color = cm.get_cmap('tab10')(1)
cmap_res = ListedColormap([glass_res_color])


# =============================================================================
# FLATMAPS VBM TOM
# =============================================================================

# Load ToM parcellation from King et al. 
tom_path = os.path.join(data_dir, "atlases/con-MDTB06TheoryOfMind_space-SUIT.nii")
tom_img = nib.load(tom_path)
tom_surf = np.asarray(flatmap.vol_to_surf(tom_img)).squeeze()

# Keep and binarize big positive clusters
tom_surf[~np.isfinite(tom_surf)] = np.nan
tom_surf[tom_surf <= 0.06] = np.nan
tom_surf = np.isfinite(tom_surf).astype(float)

# Parcel overlay color is constant across both datasets below, so build it once
parcel_color = parcel_base_color.copy()
parcel_color[3] = 0.4  # Opacity
parcel_cmap = LinearSegmentedColormap.from_list('parcel', [bg_color, parcel_color])

dataset_specs = [
    {
        "stat_file": os.path.join(data_dir, "results/tom/tomsyn/vbm/tomsyn_vbm_covs_SexAgeTCV_fdr.nii.gz"),
        "overlay_file": os.path.join(data_dir, "results/tom/tomsyn/vbm/tomsyn_vbm_king_overlay.png"),
    },
    {
        "stat_file": os.path.join(data_dir, "results/tom/richardson/vbm/richardson_vbm_covs_SexAgeTCV_fdr.nii.gz"),
        "overlay_file": os.path.join(data_dir, "results/tom/richardson/vbm/richardson_vbm_king_overlay.png"),
    },
]

for spec in dataset_specs:

    # ---- Load VBM results ----
    stat_img = nib.load(spec["stat_file"])
    stat_surf = np.asarray(flatmap.vol_to_surf(stat_img)).squeeze()

    # Keep and binarize positive clusters
    pos_only = stat_surf.copy()
    pos_only[~np.isfinite(pos_only)] = np.nan
    pos_only[pos_only <= 0] = np.nan
    pos_bin = np.isfinite(pos_only).astype(float)

    # ---- Plot ----
    fig, ax = plt.subplots()

    # Background flatmap
    bg = np.ones_like(tom_surf, dtype=float)
    flatmap.plot(bg, colorbar=False, new_figure=False, bordersize=0.8)

    # King parcellation overlay
    flatmap.plot(tom_surf.astype(float),
                 cmap=parcel_cmap,
                 colorbar=False,
                 new_figure=False,
                 bordersize=0,
                 backgroundcolor=(0, 0, 0, 0),
                 underlay=np.zeros_like(tom_surf))

    # VBM results overlay
    flatmap.plot(pos_bin,
                 cmap=res_cmap,
                 colorbar=False,
                 new_figure=False,
                 bordersize=0,
                 backgroundcolor=(0, 0, 0, 0),
                 underlay=np.zeros_like(pos_bin))

    # ---- Save ----
    fig.savefig(
        spec["overlay_file"],
        dpi=300,
        bbox_inches="tight",
        facecolor="white"
    )

    plt.show()


# =============================================================================
# FLATMAPS VBM ACTION PREDICTION
# =============================================================================

# ---- VBM results ----
stat_path = os.path.join(data_dir, "results/ap/tomsyn/vbm/tomsyn_vbm_covs_SexAgeTCV_fdr.nii.gz")
stat_img = nib.load(stat_path)
stat_surf = np.asarray(flatmap.vol_to_surf(stat_img)).squeeze()

# Keep and binarize positive clusters
pos_only = stat_surf.copy()
pos_only[~np.isfinite(pos_only)] = np.nan
pos_only[pos_only <= 0] = np.nan
pos_bin = np.isfinite(pos_only).astype(float)


# SALIENCE NETWORK

# Load atlas NIfTI
atl_path = os.path.join(data_dir, "atlases/atl-Buckner7_space-SUIT_dseg.nii")
atl_img = nib.load(atl_path)
atl_c = atl_img.get_fdata()

# Load label names
lab_num = pd.read_csv(os.path.join(data_dir, 'atlases/buckner_labels.txt'), header=None)
lab_num.columns = ['parcel_name']
lab_num['parcel'] = np.arange(1, len(lab_num) + 1)

# Find parcel ID
target_names = ['network 4']
targets = lab_num[lab_num['parcel_name'].isin(target_names)]
print(targets)  # Sanity check
target_labels = targets['parcel'].tolist()

# Mask and binarize target parcel
sal_vol = np.isin(atl_c, target_labels).astype(np.float32)  # 1 in parcels, 0 elsewhere
sal_img = nib.Nifti1Image(sal_vol, atl_img.affine)
sal_surf = np.asarray(flatmap.vol_to_surf(sal_img)).squeeze()
sal_bin = (sal_surf > 0).astype(float)


# KING ACTION OBSERVATION

# Load contrast from King et al.
ao_path = os.path.join(data_dir, "atlases/con-MDTB07ActionObservation_space-SUIT.nii")
ao_img = nib.load(ao_path)
ao_surf = np.asarray(flatmap.vol_to_surf(ao_img)).squeeze()

# Keep and binarize big positive clusters
ao_surf[~np.isfinite(ao_surf)] = np.nan
ao_surf[ao_surf <= 0.01] = np.nan
ao_bin = np.isfinite(ao_surf).astype(float)


# ---- Plot and save each overlay ----
overlay_specs = [
    {
        "overlay_surf": sal_bin,
        "parcel_opacity": 0.2,
        "overlay_file": os.path.join(data_dir, "results/ap/tomsyn/vbm/tomsyn_vbm_sal_overlay.png"),
    },
    {
        "overlay_surf": ao_bin,
        "parcel_opacity": 0.4,
        "overlay_file": os.path.join(data_dir, "results/ap/tomsyn/vbm/tomsyn_vbm_king_overlay.png"),
    },
]

for spec in overlay_specs:

    fig, ax = plt.subplots()

    # Background flatmap
    bg = np.ones_like(spec["overlay_surf"], dtype=float)
    flatmap.plot(bg, cmap=grays_mild, colorbar=False, new_figure=False, bordersize=0.8)

    # Atlas overlay (opacity varies per item, so cmap is rebuilt here)
    parcel_color = parcel_base_color.copy()
    parcel_color[3] = spec["parcel_opacity"]  # Opacity
    parcel_cmap = LinearSegmentedColormap.from_list('parcel', [bg_color, parcel_color])
    flatmap.plot(spec["overlay_surf"].astype(float),
                 cmap=parcel_cmap,
                 colorbar=False,
                 new_figure=False,
                 bordersize=0,
                 backgroundcolor=(0, 0, 0, 0),
                 underlay=np.zeros_like(spec["overlay_surf"]))

    # VBM results overlay
    flatmap.plot(pos_bin,
                 cmap=res_cmap,
                 colorbar=False,
                 new_figure=False,
                 bordersize=0,
                 backgroundcolor=(0, 0, 0, 0),
                 underlay=np.zeros_like(pos_bin))

    fig.savefig(spec["overlay_file"], dpi=300, bbox_inches="tight", facecolor="white")

    plt.show()


# =============================================================================
# GLASS BRAINS COVARIANCE TOM
# =============================================================================

# Load ToM meta-analysis image from Schurz et al.
tom_path = os.path.join(data_dir, "atlases/Cl1_thresh.nii.gz")
tom_img = nib.load(tom_path)
tom = tom_img.get_fdata()

# Threshold image
bin_data = np.where(tom > 3.5, 1.0, 0.0)

bin_img = nib.Nifti1Image(
    bin_data.astype(np.float32),
    tom_img.affine,
    tom_img.header
)

# Load and plot covariance results
dataset_specs = [
    {
        "stat_file": os.path.join(data_dir, "results/tom/tomsyn/covariance/tomsyn_covar_SexAgeTIV_lcrus2_fdr.nii.gz"),
        "overlay_file": os.path.join(data_dir, "results/tom/tomsyn/covariance/tomsyn_covar_lcrus2_ToMMeta_SexAgeTIV_lcrus2.png"),
    },
    {
        "stat_file": os.path.join(data_dir, "results/tom/richardson/covariance/richardson_covar_SexAgeTIV_lcrus2_fdr.nii.gz"),
        "overlay_file": os.path.join(data_dir, "results/tom/richardson/covariance/richardson_covar_lcrus2_ToMMeta_SexAgeTIV_lcrus2.png"),
    },
]

for spec in dataset_specs:

    # ---- Load and binarize stat image ----
    stat_img = nib.load(spec["stat_file"])
    stat_data = stat_img.get_fdata()
    mask = (stat_data > 0).astype(np.float32)

    mask_img = nib.Nifti1Image(
        mask,
        stat_img.affine,
        stat_img.header
    )

    # ---- Plot on glass brain ----
    display = plot_glass_brain(
        None,
        display_mode="lyr"
    )

    display.add_overlay(
        bin_img,
        cmap=cmap_net,
        threshold=0.5,
        alpha=0.4
    )

    display.add_overlay(
        mask_img,
        cmap=cmap_res,
        threshold=0.5,
        alpha=1.0
    )

    # ---- Save figure ----
    display.savefig(spec["overlay_file"], dpi=300)


# =============================================================================
# GLASS BRAINS COVARIANCE ACTION PREDICTION
# =============================================================================

## TOMSYN

rois = ['rviia', 'lviib', 'rviib']

# Load salience network
sal_labels = [6, 7, 8, 9, 10, 32, 33, 34, 35, 36, 37]

atlas_path = os.path.join(data_dir, "atlases/Yeo2011_7Networks_N1000.split_components.FSL_MNI152_2mm.nii.gz")
atlas_img = nib.load(atlas_path)
atlas = atlas_img.get_fdata()

sal_img = nib.Nifti1Image(
    np.isin(atlas, sal_labels).astype(np.float32),
    atlas_img.affine,
    atlas_img.header
)

for roi in rois:

    # ---- Load and binarize stat image ----
    stat_img_path = os.path.join(data_dir, f'results/ap/tomsyn/covariance/tomsyn_covar_SexAgeTIV_{roi}_fdr.nii.gz')
    stat_img = nib.load(stat_img_path)

    stat_data = stat_img.get_fdata()
    mask = (stat_data > 0).astype(np.float32)

    mask_img = nib.Nifti1Image(
        mask,
        stat_img.affine,
        stat_img.header
    )

    # ---- Plot salience network underneath and stat mask on top ----
    display = plot_glass_brain(
        None,
        display_mode="lyr"
    )

    display.add_overlay(
        sal_img,
        cmap=cmap_net,
        threshold=0.5,
        alpha=0.4
    )

    display.add_overlay(
        mask_img,
        cmap=cmap_res,
        threshold=0.5,
        alpha=1.0
    )

    display.savefig(
        os.path.join(data_dir, f"results/ap/tomsyn/covariance/tomsyn_covar_sal_SexAgeTIV_{roi}.png"),
        dpi=300
    )