%Script to extract BOLD signal variance from preprocessed volumes with the vartbx toolbox, using 1st-level SPM models. 
% creates one variance nifti per block (i.e. condition) and per subject, then uses marsbar to extract 
% average ROI variance per block and per subject.

%%%%%%%

clear
spm('defaults', 'fmri');
spm_jobman('initcfg');

preprocessed_data_dir = 'D:\Path\To\Global\SPM\1stLevel\Directory\' % contains individual subject directories named as subjects, 
% which themselves must contain rp*.txt realignment parameter file from SPM preprocessing and preprocessed 3D .nii files
% (e.g. \\subject001\\rp*.txt file)
variability_dir = 'D:\Path\To\Variability\Outputs\'; %variability .nii outputs will go there
output_data_dir = 'D:\\Path\\To\\Final\\Outputs\\'; %final .csv output will go there
subject_list = importdata('D:\Path\To\Subject\List.txt');
n_functional_volumes = [] %how many functional volumes in the scan (e.g. 300)

jobs = [];
for crun = 1:length(subject_list)
    subject = subject_list(crun)
    subject_dir = strcat(preprocessed_data_dir, char(subject), '\');
    output_subject_dir = strcat(variability_dir, char(subject),'\');
    
    %need to reconstruct SPM model file with subject file input, output directory, 
    % and subject motion file just like in 1st-level script
    [preprocessed_volumes_paths, ~] = concatenate_fvols (subject, subject_dir); % trying with 3D data instead of 4D
    subject_input = preprocessed_volumes_paths;
    subject_motion = dir(strcat(subject_dir,'rp_','*.txt')); subject_motion = strcat(subject_motion.folder,'\',subject_motion.name);
    model_job = SPM_1stlevel_job(subject_input, subject_motion, output_subject_dir);

    %need to save file because spm jobman can only take the filename 
    save(strcat(output_subject_dir, '1st_level_model_', char(subject)),'model_job');

    %variance extraction
    subject_job = var_extract_job(subject, output_subject_dir);
    jobs = [jobs subject_job];
end
spm_jobman('run', jobs);


%% extract mean ROI variability from ready-to-use variability niftis, using marsbar


marsbar('on');

roi_dir = 'D:\\Path\\To\\ROI-containing\\Directory'; %need .mat ROI files
roi_names = roi_names = importdata('D:\Path\To\List\Of\ROI\Names.txt'); %need .mat ROI names
block_names = {'PhasicFear','SustainedFear','NoFear'}; %names of conditions of interest as they have been written in SPM design

all_mean_vars = [];
for nsubject = 1:length(subject_list)
    subject_mean_vars = [];roi_mean_vars = [];
    for nblock = 1:length(block_names)
        block = block_names(nblock)
        block
        subject_effect_sizes = []
        subject = subject_list{nsubject}

        subject_input_dir = strcat(variability_dir, char(subject),'\');
        
        for roi_no = 1:length(roi_names)

            %Extract roi from roi list and make it a marsbar roi object
            roi_name = roi_names{roi_no};

            %assign roi object to current roi
            roi_file = load(fullfile(roi_dir, roi_name), 'roi');
            roi_file = roi_file.roi;
            roi = maroi(roi_file);

            % Extract roi data from the block variability nifti
            Y{roi_no} = get_marsy(roi_file, char(strcat(subject_input_dir,'var_', block,'.nii')) , 'mean');
            roi_mean_var = summary_data(Y{roi_no});
            roi_mean_vars = [roi_mean_vars, [strcat(block,'_', roi_name);roi_mean_var]];
        end
    end

    %subject output

    subject_mean_vars = [subject_mean_vars roi_mean_vars]; 
    if nsubject == 1
        all_mean_vars = [subject_mean_vars]; %keep first row with colnames 
        
    else 
        all_mean_vars = [all_mean_vars ; subject_mean_vars(2,:)];
    end
end
col_names = ['Subjects', all_mean_vars(1,:)];
col_names = [col_names;repmat({','},1,numel(col_names))]; % put in comma for csv
col_names = col_names(:)';
col_names = cell2mat(col_names);


%Final outputs
final_results = fopen(strcat(output_data_dir, 'Variability_ExampleCode.csv'),'w'); 
fprintf(final_results,'%s\n',col_names);
fclose(final_results);
%write data to end of file
writecell([subject_list, all_mean_vars(2:size(all_mean_vars,1),:)], ...
        strcat(output_data_dir, 'Variability_ExampleCode.csv'),'WriteMode','append');




%% Function to grab preprocessed functional volumes for one given subject
function [preprocessed_volumes_paths, subject_filenames] = concatenate_fvols(subject,subject_directory)
subject_filenames = {};
preprocessed_volumes_paths = {};

for i = 1:n_functional_volumes %grab the subject-specific full filenames of each preprocessed functional volume
   subject_files = dir(strcat(subject_directory,'wu*','.nii'));
   subject_filenames{end+1,1} = subject_files(i).name;
end
preprocessed_volumes_paths = cellstr(string(strcat(subject_directory, char(subject_filenames))));
end


%% Use of vartbx to extract variability nifti
function job = var_extract_job(subject, dir_subject_output);

spm_design_path = cellstr(strcat(dir_subject_output, '1st_level_model_', string(subject),'.mat'));
matlabbatch{1}.spm.tools.variability.modeltype = 'boxcar';
matlabbatch{1}.spm.tools.variability.modelmat = spm_design_path;
matlabbatch{1}.spm.tools.variability.metric = 'var';
matlabbatch{1}.spm.tools.variability.resultprefix = 'var';
matlabbatch{1}.spm.tools.variability.resultdir = {dir_subject_output};
job = matlabbatch
end


%% recreate SPM 1st-level job (example given is specific to the task used in the publication)

function job = SPM_1stlevel_job(subject_input, subject_motion, dir_subject_output)

matlabbatch{1}.spm.stats.fmri_spec.dir = {dir_subject_output};
matlabbatch{1}.spm.stats.fmri_spec.timing.units = 'secs';
matlabbatch{1}.spm.stats.fmri_spec.timing.RT = 2;
matlabbatch{1}.spm.stats.fmri_spec.timing.fmri_t = 16;
matlabbatch{1}.spm.stats.fmri_spec.timing.fmri_t0 = 8;
matlabbatch{1}.spm.stats.fmri_spec.sess.scans = subject_input;
matlabbatch{1}.spm.stats.fmri_spec.sess.cond(1).name = 'PhasicFear';
matlabbatch{1}.spm.stats.fmri_spec.sess.cond(1).onset = [4.4
                                                         126.2
                                                         207.4
                                                         329.2
                                                         451];
matlabbatch{1}.spm.stats.fmri_spec.sess.cond(1).duration = 20;
matlabbatch{1}.spm.stats.fmri_spec.sess.cond(1).tmod = 0;
matlabbatch{1}.spm.stats.fmri_spec.sess.cond(1).pmod = struct('name', {}, 'param', {}, 'poly', {});
matlabbatch{1}.spm.stats.fmri_spec.sess.cond(1).orth = 1;

matlabbatch{1}.spm.stats.fmri_spec.sess.cond(2).name = 'SustainedFear';
matlabbatch{1}.spm.stats.fmri_spec.sess.cond(2).onset = [85.6
                                                         166.8
                                                         248
                                                         369.8
                                                         532.2];
matlabbatch{1}.spm.stats.fmri_spec.sess.cond(2).duration = 20;
matlabbatch{1}.spm.stats.fmri_spec.sess.cond(2).tmod = 0;
matlabbatch{1}.spm.stats.fmri_spec.sess.cond(2).pmod = struct('name', {}, 'param', {}, 'poly', {});
matlabbatch{1}.spm.stats.fmri_spec.sess.cond(2).orth = 1;

matlabbatch{1}.spm.stats.fmri_spec.sess.cond(3).name = 'NoFear';
matlabbatch{1}.spm.stats.fmri_spec.sess.cond(3).onset = [45
                                                         288.6
                                                         410.4
                                                         491.6
                                                         572.8];
matlabbatch{1}.spm.stats.fmri_spec.sess.cond(3).duration = 20;
matlabbatch{1}.spm.stats.fmri_spec.sess.cond(3).tmod = 0;
matlabbatch{1}.spm.stats.fmri_spec.sess.cond(3).pmod = struct('name', {}, 'param', {}, 'poly', {});
matlabbatch{1}.spm.stats.fmri_spec.sess.cond(3).orth = 1;

matlabbatch{1}.spm.stats.fmri_spec.sess.cond(4).name = 'InactiveBlocks';
matlabbatch{1}.spm.stats.fmri_spec.sess.cond(4).onset = [27.6
                                                         68.2
                                                         108.8
                                                         149.4
                                                         190
                                                         231
                                                         271.2
                                                         311.8
                                                         352.4
                                                         393.9
                                                         433.6
                                                         474.2
                                                         514.9
                                                         555.5];
matlabbatch{1}.spm.stats.fmri_spec.sess.cond(4).duration = 15;
matlabbatch{1}.spm.stats.fmri_spec.sess.cond(4).tmod = 0;
matlabbatch{1}.spm.stats.fmri_spec.sess.cond(4).pmod = struct('name', {}, 'param', {}, 'poly', {});
matlabbatch{1}.spm.stats.fmri_spec.sess.cond(4).orth = 1;

matlabbatch{1}.spm.stats.fmri_spec.sess.cond(5).name = 'Instructions';
matlabbatch{1}.spm.stats.fmri_spec.sess.cond(5).onset = [2.4
                                                         43
                                                         83.6
                                                         124.2
                                                         164.8
                                                         205.4
                                                         246
                                                         286.6
                                                         327.2
                                                         367.8
                                                         408.4
                                                         449
                                                         489.6
                                                         530.2
                                                         570.6];
matlabbatch{1}.spm.stats.fmri_spec.sess.cond(5).duration = 2;
matlabbatch{1}.spm.stats.fmri_spec.sess.cond(5).tmod = 0;
matlabbatch{1}.spm.stats.fmri_spec.sess.cond(5).pmod = struct('name', {}, 'param', {}, 'poly', {});
matlabbatch{1}.spm.stats.fmri_spec.sess.cond(5).orth = 1;

matlabbatch{1}.spm.stats.fmri_spec.sess.cond(6).name = 'Ratings';
matlabbatch{1}.spm.stats.fmri_spec.sess.cond(6).onset = [24.6
                                                         65.2
                                                         105.8
                                                         146.4
                                                         187
                                                         228
                                                         268.2
                                                         308.8
                                                         349.4
                                                         390.9
                                                         430.6
                                                         471.2
                                                         511.9
                                                         552.5
                                                         593.1];
matlabbatch{1}.spm.stats.fmri_spec.sess.cond(6).duration = 3;
matlabbatch{1}.spm.stats.fmri_spec.sess.cond(6).tmod = 0;
matlabbatch{1}.spm.stats.fmri_spec.sess.cond(6).pmod = struct('name', {}, 'param', {}, 'poly', {});
matlabbatch{1}.spm.stats.fmri_spec.sess.cond(6).orth = 1;

matlabbatch{1}.spm.stats.fmri_spec.sess.multi = {''};
matlabbatch{1}.spm.stats.fmri_spec.sess.regress = struct('name', {}, 'val', {});
matlabbatch{1}.spm.stats.fmri_spec.sess.multi_reg = {subject_motion};
matlabbatch{1}.spm.stats.fmri_spec.sess.hpf = 128;

matlabbatch{1}.spm.stats.fmri_spec.fact = struct('name', {}, 'levels', {});
matlabbatch{1}.spm.stats.fmri_spec.bases.hrf.derivs = [0 0];
matlabbatch{1}.spm.stats.fmri_spec.volt = 1;
matlabbatch{1}.spm.stats.fmri_spec.global = 'None';
matlabbatch{1}.spm.stats.fmri_spec.mthresh = 0.8;
matlabbatch{1}.spm.stats.fmri_spec.mask = {''};
matlabbatch{1}.spm.stats.fmri_spec.cvi = 'AR(1)';

matlabbatch{2}.spm.stats.fmri_est.spmmat(1) = cfg_dep('fMRI model specification: SPM.mat File', substruct('.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','spmmat'));
matlabbatch{2}.spm.stats.fmri_est.write_residuals = 0;
matlabbatch{2}.spm.stats.fmri_est.method.Classical = 1;

matlabbatch{3}.spm.stats.con.spmmat(1) = cfg_dep('Model estimation: SPM.mat File', substruct('.','val', '{}',{2}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','spmmat'));
matlabbatch{3}.spm.stats.con.consess{1}.tcon.name = 'PhasicFear>NoFear';
matlabbatch{3}.spm.stats.con.consess{1}.tcon.weights = [1 0 -1];
matlabbatch{3}.spm.stats.con.consess{1}.tcon.sessrep = 'none';
matlabbatch{3}.spm.stats.con.consess{2}.tcon.name = 'NoFear>PhasicFear';
matlabbatch{3}.spm.stats.con.consess{2}.tcon.weights = [-1 0 1];
matlabbatch{3}.spm.stats.con.consess{2}.tcon.sessrep = 'none';
matlabbatch{3}.spm.stats.con.consess{3}.tcon.name = 'SustainedFear>NoFear';
matlabbatch{3}.spm.stats.con.consess{3}.tcon.weights = [0 1 -1];
matlabbatch{3}.spm.stats.con.consess{3}.tcon.sessrep = 'none';
matlabbatch{3}.spm.stats.con.consess{4}.tcon.name = 'NoFear>SustainedFear';
matlabbatch{3}.spm.stats.con.consess{4}.tcon.weights = [0 -1 1];
matlabbatch{3}.spm.stats.con.consess{4}.tcon.sessrep = 'none';
matlabbatch{3}.spm.stats.con.consess{5}.tcon.name = 'ActiveBlocks>InactiveBlocks';
matlabbatch{3}.spm.stats.con.consess{5}.tcon.weights = [1 1 1 -3];
matlabbatch{3}.spm.stats.con.consess{5}.tcon.sessrep = 'none';
matlabbatch{3}.spm.stats.con.delete = 0;
job = matlabbatch;
end
