
%%CAT12 simple preprocessing script.
% Takes raw 3d data, conducts preprocessing, then extracts TIV, GM volumes, 
% cortical thickness and gyrification from preprocessing results.

%%%%%%%

subject_list = importdata('D:\Path\To\Subject\List.txt');

cat12_data_dir = 'D:\Path\To\Cat12\Processing\Outputs\';% cat12 processing outputs will go there, 
% must already contain folders for each subject, themselves containing subject raw nifti data
subject_filenames_list = importdata('D:\Path\To\Subject\Nifti\List.txt'); %list of structural .nii filenames of all subjects
output_data_dir = 'D:\Path\To\Final\Outputs\'; %final .csv output will go there

%% For each subject, make the job and run the job 

spm('defaults', 'FMRI');
spm_jobman('initcfg');

for crun = 1:length(subject_list)
    jobs = [];
    subject = subject_list(crun);
    subject_filename = subject_filenames_list(crun)
    subject_dir = strcat(cat12_data_dir, char(subject), '\')

    subject_job = cat12prepro_job(strcat(subject_dir,subject_filename));
    jobs = [jobs subject_job]
    spm_jobman('run', jobs);
end


%aggregate individual thickness, gyrification and volume measures into one csv
all_data = {}

for crun = 1:length(subject_list) 
    subject = subject_list(crun)
    subject_dir = strcat(cat12_data_dir, char(subject));

    subject_grabfiles = dir(strcat(subject_dir,  '\label\catROI', '*', '.mat'))
    struct_filename = subject_grabfiles(1).name
    surf_filename = subject_grabfiles(2).name

    subject_grabfiles = dir(strcat(subject_dir,  '\report\cat', '*', '.mat'))
    report_filename = subject_grabfiles.name

    struct_file = load(strcat(subject_dir, '\label\', struct_filename)); 
    surf_file = load (strcat(subject_dir, '\label\',surf_filename)); 
    report_file = load (strcat(subject_dir, '\report\',report_filename))

    if crun == 1
        sub_data = ['TIV', num2cell(report_file.S.subjectmeasures.vol_TIV);...;
            strcat(struct_file.S.neuromorphometrics.names, '_volume'), num2cell(struct_file.S.neuromorphometrics.data.Vgm);...;
            strcat(surf_file.S.aparc_a2009s.names,'_thickness'), num2cell(surf_file.S.aparc_a2009s.data.thickness); ...
            strcat(surf_file.S.aparc_a2009s.names,'_gyrification'),num2cell(surf_file.S.aparc_a2009s.data.gyrification)].';
    else 
        sub_data = [num2cell(report_file.S.subjectmeasures.vol_TIV); num2cell(struct_file.S.neuromorphometrics.data.Vgm); ...
            num2cell(surf_file.S.aparc_a2009s.data.thickness); num2cell(surf_file.S.aparc_a2009s.data.gyrification)].';
    end
    all_data = vertcat(all_data,sub_data);
end

Measures_sample = [['Subjects'; subject_list(1:nrun,:)],all_data];
writecell(Measures_sample, strcat('D:\Path\To\Final\Outputs\','CAT12_measures.csv'));

%% Cat12 preprocessing function

function job = cat12prepro_job(volume)
volume = cellstr(volume);
matlabbatch{1}.spm.tools.cat.cat_simple.data = volume;
matlabbatch{1}.spm.tools.cat.cat_simple.tpm = 'adults';
matlabbatch{1}.spm.tools.cat.cat_simple.affmod = 0;
matlabbatch{1}.spm.tools.cat.cat_simple.ROImenu.atlases.neuromorphometrics = 1;
matlabbatch{1}.spm.tools.cat.cat_simple.ROImenu.atlases.lpba40 = 1;
matlabbatch{1}.spm.tools.cat.cat_simple.ROImenu.atlases.cobra = 1;
matlabbatch{1}.spm.tools.cat.cat_simple.ROImenu.atlases.hammers = 0;
matlabbatch{1}.spm.tools.cat.cat_simple.ROImenu.atlases.thalamus = 1;
matlabbatch{1}.spm.tools.cat.cat_simple.ROImenu.atlases.thalamic_nuclei = 1;
matlabbatch{1}.spm.tools.cat.cat_simple.ROImenu.atlases.suit = 1;
matlabbatch{1}.spm.tools.cat.cat_simple.ROImenu.atlases.ibsr = 0;
matlabbatch{1}.spm.tools.cat.cat_simple.ROImenu.atlases.ownatlas = {''};
matlabbatch{1}.spm.tools.cat.cat_simple.fwhm_vol = 6;
matlabbatch{1}.spm.tools.cat.cat_simple.surface.yes.sROImenu.satlases.Desikan = 1;
matlabbatch{1}.spm.tools.cat.cat_simple.surface.yes.sROImenu.satlases.Destrieux = 1;
matlabbatch{1}.spm.tools.cat.cat_simple.surface.yes.sROImenu.satlases.HCP = 0;
matlabbatch{1}.spm.tools.cat.cat_simple.surface.yes.sROImenu.satlases.Schaefer2018_200P_17N = 0;
matlabbatch{1}.spm.tools.cat.cat_simple.surface.yes.sROImenu.satlases.ownatlas = {''};
matlabbatch{1}.spm.tools.cat.cat_simple.surface.yes.fwhm_surf1 = 12;
matlabbatch{1}.spm.tools.cat.cat_simple.surface.yes.fwhm_surf2 = 20;
matlabbatch{1}.spm.tools.cat.cat_simple.nproc = 6;
job = matlabbatch;
end
