# The cerebellum supports two systems for understanding others in early childhood 

#### This repository contains analysis scripts and cerebellar normative models presented in this manuscript: 
TBD

### Instructions
If you'd like to reproduce the analyses in the manuscript, you can run the following scripts:

#### A. Segmentation
Scripts in **scripts/segmentation** generate tissue segmentations for the cerebellum and the cerebral cortex.
- **cereb_seg.m**: Runs SUIT to segment the cerebellum. Cerebellum isolation masks should be visually inspected and manually corrected when necessary via ITK-SNAP: https://www.itksnap.org/pmwiki/pmwiki.php
- **cortex_seg.m**: Runs CAT12 to segment the cerebral cortex. QC outputs should be visually inspected.

_Expected outputs_: Normalized, modulated GM images of the cerebellum and the cerebral cortex.

_Expected runtime_: ~15 minutes per script per subject (manual corrections: ~20 minutes per subject).

#### B. Analyses
Scripts in **scripts/analyses** perform VBM and cerebello-cerebral covariance analyses.
- **vbm_models.py**: Runs group-level VBM on cerebellar modulated GM to identify volumetric changes as a function of social abilities.

<img width="475" height="517" alt="Screenshot 2026-07-04 at 21 56 29" src="https://github.com/user-attachments/assets/163ab17b-e60a-43a0-b785-01c91f052505" />

<img width="669" height="306" alt="Screenshot 2026-07-04 at 21 56 33" src="https://github.com/user-attachments/assets/967d8406-0ce6-4859-aaf7-b0eeca0e390e" />

- **covar_models.py**: Runs group-level covariance between cerebellar ROIs and cereblar GM as a function of social abilities.

<img width="617" height="321" alt="Screenshot 2026-07-04 at 21 57 25" src="https://github.com/user-attachments/assets/8759c6c5-a648-4948-8c43-9e6bde4e31b1" />

<img width="526" height="329" alt="Screenshot 2026-07-04 at 21 57 38" src="https://github.com/user-attachments/assets/8e34e9cb-4a91-45b0-b044-4430050134fd" />

- **overlay_plots.py**: Plots VBM and covariance results overlaid on adult functional atlases.

_Expected outputs_: FDR-corrected VBM/covariance z-maps and plots.

_Expected runtime_: ~10 minutes per script.

### Data
All materials and data from the ToMSyn dataset are stored in a local repository at the Max Planck Institute for Human Cognitive and Brain Sciences. Fully anonymized data are available upon reasonable request, subject to the data protection regulations and ethical approvals governing the study. The ToM replication dataset (Richardson et al., 2018) is publicly available through OpenNeuro (https://openneuro.org/datasets/ds000228/versions/1.1.0). 

Adult maps used to contextualize the developmental findings are publicly available: MDTB cerebellar atlas (King et al., 2019; https://github.com/DiedrichsenLab/cerebellar_atlases/tree/master/King_2019), cerebellar resting-state functional networks (Buckner et al., 2011; https://github.com/DiedrichsenLab/cerebellar_atlases/tree/master/Buckner_2011), ToM meta-analysis maps (Schurz et al., 2021; https://osf.io/pav27/files/mrb35), and cerebral resting-state functional networks (Yeo et al., 2011; https://github.com/ThomasYeoLab/CBIG/tree/master/stable_projects/brain_parcellation/Yeo2011_fcMRI_clustering/1000subjects_reference/Yeo_JNeurophysiol11_SplitLabels).

### Requirements and installation
All Python scripts were executed using Python 3.11.8 (https://www.python.org/downloads/) (please see associated scripts for required Python libraries). Image segmentation and normalization was run in SUIT for the cerebellum (v. 3.5; https://github.com/jdiedrichsen/suit/releases/tag/3.5) and CAT12 for the cerebral cortex (https://github.com/ChristianGaser/cat12/releases/tag/26.0.rc3), within SPM12 (https://www.fil.ion.ucl.ac.uk/spm/software/spm12/) in MATLAB R2025b (https://www.mathworks.com/help/install/ug/install-products-with-internet-connection.html).

The scripts can be run on a standard desktop computer, however, we recommend running these analyses (especially segmentation algorithms) on a cluster to parallelize computations.



