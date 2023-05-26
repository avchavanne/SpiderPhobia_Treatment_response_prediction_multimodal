%%Preprocessing script 
%containing realignment, unwarping, normalize, smoothing and 3dto4d. Takes
%raw 3d data.

%%%%%%%

subject_list = importdata('D:\Path\To\Subject\List.txt');


%% Make and run spm jobs for each subject
spm('defaults', 'FMRI');
spm_jobman('initcfg');
spm_path = 'C:\Path\To\SPM12\';
raw_data_dir = 'D:\Path\To\Global\Raw\Data\Dir\';% contains individual subject directories containing subject raw 3D data
n_functional_volumes = []; %how many functional volumes in the scan (e.g. 300)

jobs = [];
for crun = 1:length(subject_list)
    
    subject = subject_list(crun);
    subject_dir = strcat(raw_data_dir, char(subject), '\');
    [volumes_paths,subject_filenames,unwarped_volumes_paths] = concatenate_fvols(subject); 
   
    subject_job = SPM_prepro_job(subject, volumes_paths, unwarped_volumes_paths, spm_path);
    jobs = [jobs subject_job]
      
end
spm_jobman('run', jobs);

%% Function to grab functional volumes, provided directory
function [volumes_paths, subject_filenames] = concatenate_fvols(subject,subject_directory)
subject_filenames = {};
volumes_paths = {};
unwarped_volumes_paths = {};
for i = 1:n_functional_volumes %grab the subject-specific full filenames of each raw functional volume
   subject_files = dir(strcat(subject_directory,'*.nii')); %add specific name element of functional volumes
   subject_filenames{end+1,1} = subject_files(i).name;
end
volumes_paths = strcat(subject_directory, char(subject_filenames));
unwarped_volumes_paths = strcat(subject_directory,'\u', char(subject_filenames));
end
%% Preprocessing function
function job = SPM_prepro_job(subject, volumes, unwarped_volumes, spm_path)

volumes = cellstr(volumes);
unwarped_volumes = cellstr(unwarped_volumes);
first_unwarped_volume = cellstr(unwarped_volumes{1})

matlabbatch{1}.spm.spatial.realignunwarp.data.scans = volumes;
matlabbatch{1}.spm.spatial.realignunwarp.data.pmscan = '';
matlabbatch{1}.spm.spatial.realignunwarp.eoptions.quality = 0.9;
matlabbatch{1}.spm.spatial.realignunwarp.eoptions.sep = 4;
matlabbatch{1}.spm.spatial.realignunwarp.eoptions.fwhm = 5;
matlabbatch{1}.spm.spatial.realignunwarp.eoptions.rtm = 0;
matlabbatch{1}.spm.spatial.realignunwarp.eoptions.einterp = 2;
matlabbatch{1}.spm.spatial.realignunwarp.eoptions.ewrap = [0 0 0];
matlabbatch{1}.spm.spatial.realignunwarp.eoptions.weight = '';
matlabbatch{1}.spm.spatial.realignunwarp.uweoptions.basfcn = [12 12];
matlabbatch{1}.spm.spatial.realignunwarp.uweoptions.regorder = 1;
matlabbatch{1}.spm.spatial.realignunwarp.uweoptions.lambda = 100000;
matlabbatch{1}.spm.spatial.realignunwarp.uweoptions.jm = 0;
matlabbatch{1}.spm.spatial.realignunwarp.uweoptions.fot = [4 5];
matlabbatch{1}.spm.spatial.realignunwarp.uweoptions.sot = [];
matlabbatch{1}.spm.spatial.realignunwarp.uweoptions.uwfwhm = 4;
matlabbatch{1}.spm.spatial.realignunwarp.uweoptions.rem = 1;
matlabbatch{1}.spm.spatial.realignunwarp.uweoptions.noi = 5;
matlabbatch{1}.spm.spatial.realignunwarp.uweoptions.expround = 'Average';
matlabbatch{1}.spm.spatial.realignunwarp.uwroptions.uwwhich = [2 1];
matlabbatch{1}.spm.spatial.realignunwarp.uwroptions.rinterp = 4;
matlabbatch{1}.spm.spatial.realignunwarp.uwroptions.wrap = [0 0 0];
matlabbatch{1}.spm.spatial.realignunwarp.uwroptions.mask = 1;
matlabbatch{1}.spm.spatial.realignunwarp.uwroptions.prefix = 'u';

matlabbatch{2}.spm.spatial.normalise.estwrite.subj.vol = first_unwarped_volume;
matlabbatch{2}.spm.spatial.normalise.estwrite.subj.resample = unwarped_volumes;
matlabbatch{2}.spm.spatial.normalise.estwrite.eoptions.biasreg = 0.0001;
matlabbatch{2}.spm.spatial.normalise.estwrite.eoptions.biasfwhm = 60;
matlabbatch{2}.spm.spatial.normalise.estwrite.eoptions.tpm = {strcat(spm_path, 'tpm')};
matlabbatch{2}.spm.spatial.normalise.estwrite.eoptions.affreg = 'mni';
matlabbatch{2}.spm.spatial.normalise.estwrite.eoptions.reg = [0 0.001 0.5 0.05 0.2];
matlabbatch{2}.spm.spatial.normalise.estwrite.eoptions.fwhm = 0;
matlabbatch{2}.spm.spatial.normalise.estwrite.eoptions.samp = 3;
matlabbatch{2}.spm.spatial.normalise.estwrite.woptions.bb = [-78 -112 -70
                                                             78 76 85];
matlabbatch{2}.spm.spatial.normalise.estwrite.woptions.vox = [2 2 2];
matlabbatch{2}.spm.spatial.normalise.estwrite.woptions.interp = 4;
matlabbatch{2}.spm.spatial.normalise.estwrite.woptions.prefix = 'w';

matlabbatch{3}.spm.spatial.smooth.data(1) = cfg_dep('Normalise: Estimate & Write: Normalised Images (Subj 1)', substruct('.','val', '{}',{2}, '.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('()',{1}, '.','files'));
matlabbatch{3}.spm.spatial.smooth.fwhm = [8 8 8];
matlabbatch{3}.spm.spatial.smooth.dtype = 0;
matlabbatch{3}.spm.spatial.smooth.im = 0;
matlabbatch{3}.spm.spatial.smooth.prefix = 's';

matlabbatch{4}.spm.util.cat.vols(1) = cfg_dep('Smooth: Smoothed Images', substruct('.','val', '{}',{3}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','files'));
matlabbatch{4}.spm.util.cat.name = strcat('swu',char(subject),'_4D.nii');
matlabbatch{4}.spm.util.cat.dtype = 4;
matlabbatch{4}.spm.util.cat.RT = 2;

job = matlabbatch;
end
