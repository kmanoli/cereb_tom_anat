% CAT12 segmentation: produce modulated normalized cortical GM/WM/CSF
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

% -----------------------

% Read subject IDs
fid = fopen(idList,'r'); assert(fid>0, 'Could not open %s', idList);
ids = textscan(fid,'%s'); fclose(fid);
ids = ids{1};

% Build T1 paths
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

origDir = pwd;

% -----------------------
% Run CAT12 per subject
% -----------------------
for i = 1:numel(ids)
    subj    = ids{i};
    t1      = T1paths{i};
    subjDir = fileparts(t1);
    fprintf('\n=== [%d/%d] %s ===\n', i, numel(ids), subj);

    % CAT12 batch
    matlabbatch = {};
    matlabbatch{1}.spm.tools.cat.estwrite.data = {[t1 ',1']};

    % Output modulated GM/WM/CSF in MNI
    matlabbatch{1}.spm.tools.cat.estwrite.output.GM.native      = 0;
    matlabbatch{1}.spm.tools.cat.estwrite.output.GM.warped      = 0;
    matlabbatch{1}.spm.tools.cat.estwrite.output.GM.mod         = 1; % mwp1 (GM)

    matlabbatch{1}.spm.tools.cat.estwrite.output.WM.native      = 0;
    matlabbatch{1}.spm.tools.cat.estwrite.output.WM.warped      = 0;
    matlabbatch{1}.spm.tools.cat.estwrite.output.WM.mod         = 1; % mwp2 (WM)

    matlabbatch{1}.spm.tools.cat.estwrite.output.CSF.native     = 0;
    matlabbatch{1}.spm.tools.cat.estwrite.output.CSF.warped     = 0;
    matlabbatch{1}.spm.tools.cat.estwrite.output.CSF.mod        = 1; % mwp3 (CSF)

    % QC native bias-corrected T1 only
    matlabbatch{1}.spm.tools.cat.estwrite.output.bias.warped    = 0;
    matlabbatch{1}.spm.tools.cat.estwrite.output.bias.native    = 1;
    matlabbatch{1}.spm.tools.cat.estwrite.output.surface        = 0;
    matlabbatch{1}.spm.tools.cat.estwrite.output.ROImenu.noROI  = 1;
    matlabbatch{1}.spm.tools.cat.estwrite.output.warps          = [1 1];

    % Default settings
    matlabbatch{1}.spm.tools.cat.estwrite.opts.tpm              = {fullfile(spm('dir'),'tpm','TPM.nii')};
    matlabbatch{1}.spm.tools.cat.estwrite.opts.biasstr          = 0.5;
    matlabbatch{1}.spm.tools.cat.estwrite.opts.samp             = 3;

    % Run CAT12 in subject's own folder
    cd(subjDir);
    fprintf('>>> CAT12 running on %s\n', subj);
    spm_jobman('run', matlabbatch);
    cd(origDir);
end

fprintf('\nAll subjects processed.\n');
fprintf('Per-subject outputs in each subject''s own folder under: %s/<ID>/\n', baseDir);
fprintf(' Key files per subject:\n');
fprintf('  - mwp1*.nii : modulated GM in MNI (unsmoothed)\n');
fprintf('  - mwp2*.nii : modulated WM in MNI (unsmoothed)\n');
fprintf('  - mwp3*.nii : modulated CSF in MNI (unsmoothed)\n');