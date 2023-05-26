
%%1st level script. 
% Takes preprocessed 4D data.


%%%%%%%

subject_list = importdata('D:\Path\To\Subject\List.txt');
preprocessed_data_dir = 'D:\Path\To\Global\SPM\1stLevel\Directory\';% contains individual subject directories named as subjects, 
% themselves containing preprocessed 4D .nii data and rp*.txt realignement parameters (e.g. \\subject001\\rp*.txt file)
output_data_dir = 'D:\\Path\\To\\Final\\Outputs\\'; %overall 1st-level outputs will go there
spm('defaults', 'FMRI');
spm_jobman('initcfg');    


jobs = [];
for crun = 1:length(subject_list)
    subject = subject_list(crun)
    subject_dir = strcat(preprocessed_data_dir, char(subject), '\');
    subject_output_dir = strcat(output_data_dir, char(subject));
    
    if exist(subject_output_dir,'file') == 7 %overwrite output dir if exists
        rmdir(subject_output_dir,'s');
        mkdir(subject_output_dir);
    end
    
    subject_input = {strcat(subject_dir,'swu', char(subject),'_4D.nii')}
    subject_motion = dir(strcat(subject_dir,'rp_','*.txt')); subject_motion = strcat(subject_motion.folder,'\',subject_motion.name)
    
    subject_job = SPM_1stlevel_job(subject_input, subject_motion, subject_output_dir)
    jobs = [jobs subject_job];
    
end
spm_jobman('run', jobs);


%% 1st-level SPM batch function (example is specific to the publication)
function job = SPM_1stlevel_job(subject_input, subject_motion, subject_output_dir)

matlabbatch{1}.spm.stats.fmri_spec.dir = {subject_output_dir};
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
