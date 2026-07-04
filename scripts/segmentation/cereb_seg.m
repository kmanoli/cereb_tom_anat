% SUIT segmentation: produce modulated normalized cerebellar GM
% NOTE: This file uses the ToMSyn T1 images as an example, but the process is the same for the Richardson images.

% -----------------------
% SETUP
% -----------------------

clearvars;

baseDir   = '/data/cereb_tom_anat/tomsyn/T1w';
idList    = fullfile(baseDir, 'tomsyn_sub_list.txt');
inPattern = '%s/%s_T1w.nii';  % each ID has {ID}/{ID}_T1w.nii

spm('defaults','fmri');
spm_jobman('initcfg');

% Read subject IDs
fid = fopen(idList,'r'); assert(fid>0, 'Could not open %s', idList);
ids = textscan(fid,'%s'); fclose(fid);
ids = ids{1};

% Build T1 paths and keep only those that exist
T1paths = cell(size(ids));
keep = false(size(ids));
for i = 1:numel(ids)
    t1 = sprintf(inPattern, fullfile(baseDir, ids{i}), ids{i});
    if exist(t1,'file')
        T1paths{i} = t1;
        keep(i) = true;
    else
        warning('Missing T1: %s', t1);
    end
end
ids = ids(keep);
T1paths = T1paths(keep);
assert(~isempty(T1paths), 'No valid T1 images found.');

% -----------------------
%% ISOLATION/SEGMENTATION
% -----------------------
for i = 1:numel(ids)
    t1 = T1paths{i};

    % Isolate and segment the cerebellum
    suit_isolate_seg({t1});

    % Isolation mask of the cerebellum (output file ending in '_pcereb.nii')
    % should be inspected visually and hand-corrected (e.g., in ITK-SNAP)
    % before continuing to normalization/reslicing below.
end

% -----------------------
%% BUILD NORMALIZATION + RESLICING JOBS
% -----------------------

for i = 1:numel(ids)
    [subjDir, base] = fileparts(T1paths{i});  % base e.g. 'sub01_T1w'

    seg1     = fullfile(subjDir, [base '_seg1.nii']);       % GM
    seg2     = fullfile(subjDir, [base '_seg2.nii']);       % WM
    maskCorr = fullfile(subjDir, ['c_' base '_pcereb_corr.nii']);
    affine   = fullfile(subjDir, ['Affine_' base '_seg1.mat']);
    flowfld  = fullfile(subjDir, ['u_a_' base '_seg1.nii']);

    % Normalization job
    jobND.subjND(i).gray      = {seg1};
    jobND.subjND(i).white     = {seg2};
    jobND.subjND(i).isolation = {maskCorr};

    % Reslicing job (gray matter map)
    jobR.subj(i).affineTr  = {affine};
    jobR.subj(i).flowfield = {flowfld};
    jobR.subj(i).resample  = {seg1};  
    jobR.subj(i).mask      = {maskCorr};
end
jobR.jactransf = 1; % Jacobian modulation

% -----------------------
%% NORMALIZATION
% -----------------------
% Normalize the isolated cerebellum to SUIT atlas
suit_normalize_dartel(jobND);

% -----------------------
%% RESLICING
% -----------------------
% Reslice cerebellum gray matter map to SUIT space with Jacobian
suit_reslice_dartel(jobR);