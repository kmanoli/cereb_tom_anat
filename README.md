# The cerebellum supports two systems for understanding others in early childhood 

#### This repository contains analysis scripts and cerebellar normative models presented in this manuscript: 
https://www.biorxiv.org/content/10.64898/2026.08.03.742430v1

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

<img width="439" height="485" alt="Screenshot 2026-07-29 at 21 21 34" src="https://github.com/user-attachments/assets/ee20fc83-b1ee-43bc-b431-392edb32abd2" />

<img width="632" height="299" alt="Screenshot 2026-07-29 at 21 22 04" src="https://github.com/user-attachments/assets/55a04a6d-afc0-4036-9c18-8dde1453e26a" />

- **covar_models.py**: Runs group-level covariance between cerebellar ROIs and cereblar GM as a function of social abilities.

<img width="718" height="367" alt="Screenshot 2026-07-29 at 21 22 25" src="https://github.com/user-attachments/assets/26334812-ca91-4a29-be57-784073c92121" />

<img width="716" height="442" alt="Screenshot 2026-07-29 at 21 22 40" src="https://github.com/user-attachments/assets/a79463f5-f35e-4c90-953b-dc71662476fa" />

- **overlay_plots.py**: Plots VBM and covariance results overlaid on adult functional atlases.

- **loo_bootstrap_stability.py**: Runs leave-one-out and bootstrap resampling to test the stability of the nonverbal action prediction VBM maps.

_Expected outputs_: FDR-corrected VBM/covariance z-maps and plots.

_Expected runtime_: ~10 minutes per script. The LOO and bootstrap stability analyses may take around 5 hours, depending on the computational architecture.

### Data
All materials and data from the ToMNAP dataset are stored in a local repository at the Max Planck Institute for Human Cognitive and Brain Sciences. Fully anonymized data are available upon reasonable request, subject to the data protection regulations and ethical approvals governing the study. The ToM replication dataset (Richardson et al., 2018) is publicly available through OpenNeuro (https://openneuro.org/datasets/ds000228/versions/1.1.0). 

Adult maps used to contextualize the developmental findings are publicly available: MDTB cerebellar atlas (King et al., 2019; https://github.com/DiedrichsenLab/cerebellar_atlases/tree/master/King_2019), cerebellar resting-state functional networks (Buckner et al., 2011; https://github.com/DiedrichsenLab/cerebellar_atlases/tree/master/Buckner_2011), ToM meta-analysis maps (Schurz et al., 2021; https://osf.io/pav27/files/mrb35), and cerebral resting-state functional networks (Yeo et al., 2011; https://github.com/ThomasYeoLab/CBIG/tree/master/stable_projects/brain_parcellation/Yeo2011_fcMRI_clustering/1000subjects_reference/Yeo_JNeurophysiol11_SplitLabels).

### Requirements and installation
All Python scripts were executed using Python 3.11.8 (https://www.python.org/downloads/) (please see associated scripts for required Python libraries). Image segmentation and normalization was run in SUIT for the cerebellum (v. 3.5; https://github.com/jdiedrichsen/suit/releases/tag/3.5) and CAT12 for the cerebral cortex (https://github.com/ChristianGaser/cat12/releases/tag/26.0.rc3), within SPM12 (https://www.fil.ion.ucl.ac.uk/spm/software/spm12/) in MATLAB R2025b (https://www.mathworks.com/help/install/ug/install-products-with-internet-connection.html).

The scripts can be run on a standard desktop computer, however, we recommend running these analyses (especially segmentation algorithms) on a cluster to parallelize computations.



