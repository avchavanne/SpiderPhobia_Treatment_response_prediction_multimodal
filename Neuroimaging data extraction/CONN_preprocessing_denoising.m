%%Set-up of CONN project, preprocessing and denoising

%%%%%%%

clear
nses = 1; %number of sessions
ncond = 6;%number of conditions in task
subject_list = importdata('D:\Path\To\Subject\List.txt');
raw_data_dir = 'D:\Path\To\Global\Raw\Data\Dir';% contains individual subject directories containing subject 3D raw data
n_functional_volumes = ; %how many functional volumes in the scan (e.g. 300)

roi_dir = 'D:\\Path\\To\\ROI-containing\\Directory';
roi_names = importdata('D:\Path\To\List\Of\ROI\Names.txt'); %path to .txt file containing list of ROI names
roi_names = append(roi_names, '.nii'); %need .nii rois
all_rois = roi_names.';


connbatch.filename = 'D:\Path\To\New\CONN\Project.mat'; %set name of new CONN project

%% Parallelization

connbatch.parallel.N = 0; % 0 run locally

%% Setup

connbatch.Setup.nsubjects = numel(subject_list);
connbatch.Setup.RT = 2;
connbatch.Setup.acquisitiontype = 1;

for isub = 1:numel(subject_list);
    subject = subject_list(isub);
    subject_dir = strcat(raw_data_dir, char(subject), '\');
    connbatch.Setup.nsessions{isub} = [1];
    [fvols,~] = concatenate_fvols(subject, subject_dir);
    connbatch.Setup.functionals{isub}{nses} = {fvols};
    [isvol,~,~] = grab_svol(subject, subject_dir);
    connbatch.Setup.structurals{isub}{nses} = [isvol];
end

for iroi = 1:numel(all_rois);
    iroi_name = char(all_rois(iroi));
% for custom rois
    connbatch.Setup.rois.names{iroi} = [iroi_name]; 
    connbatch.Setup.rois.files{iroi} = [strcat(roi_dir,iroi_name)];
    connbatch.Setup.rois.dimensions{iroi} =  [1]; 
    connbatch.Setup.rois.weighted(iroi) = [0];
end

for isub = 1:numel(subject_list);
    connbatch.Setup.conditions.names = {'Phasic Fear'	'Sustained Fear'	'No Fear'	'Inactive Block'	'Instructions'	'Rating'};
    connbatch.Setup.conditions.onsets{1}{isub}{nses} = [4.4 126.2 207.4 329.2 451]; %in seconds
    connbatch.Setup.conditions.onsets{2}{isub}{nses} = [85.6 166.8 248 369.8 532.2];
    connbatch.Setup.conditions.onsets{3}{isub}{nses}= [45 288.6 410.4 491.6 572.8];
    connbatch.Setup.conditions.onsets{4}{isub}{nses} = [27.6 68.2 108.8 149.4 190 231 271.2 311.8 352.4 393.9 433.6 474.2 514.9 555.5];
    connbatch.Setup.conditions.onsets{5}{isub}{nses} = [2.4 43 83.6 124.2 164.8 205.4 246 286.6 327.2 367.8 408.4 449 489.6 530.2 570.6];
    connbatch.Setup.conditions.onsets{6}{isub}{nses} = [24.6 65.2 105.8 146.4 187 228 268.2 308.8 349.4 390.9 430.6 471.2 511.9 552.5 593.1];
    connbatch.Setup.conditions.durations{1}{isub}{nses} = [20];%in seconds
    connbatch.Setup.conditions.durations{2}{isub}{nses} = [20];
    connbatch.Setup.conditions.durations{3}{isub}{nses} = [20];
    connbatch.Setup.conditions.durations{4}{isub}{nses} = [15];
    connbatch.Setup.conditions.durations{5}{isub}{nses} = [2];
    connbatch.Setup.conditions.durations{6}{isub}{nses} = [3];
    
end
for icond = 1:numel(ncond)
connbatch.Setup.conditions.param(icond) = [0];
end
connbatch.Setup.isnew = 1;
connbatch.Setup.done = 1;
connbatch.Setup.overwrite = 1;

%% Preprocessing

connbatch.Setup.preprocessing.steps = {'functional_label_as_original'	'functional_realign&unwarp'	'functional_center'	'functional_art'...
                                        'functional_segment&normalize_direct'	'functional_label_as_mnispace'	'structural_center'	...
                                        'structural_segment&normalize'	'functional_smooth'	'functional_label_as_smoothed'};
connbatch.Setup.preprocessing.art_thresholds = [3 0.5 1 1 1 0 3.3 0];%conservative parameters
connbatch.Setup.preprocessing.fwhm = [8];

%% Denoising
connbatch.Denoising.filter = [0.008,0.09]; % in Hz
connbatch.Denoising.detrending = 1; %linear detrending
connbatch.Denoising.despiking = 0;
connbatch.Denoising.regbp = 1;
connbatch.Denoising.done = 1;
connbatch.Denoising.overwrite = 1;

conn_batch(connbatch)

%% Function to grab functional volumes, provided directory
function [volumes_paths, subject_filenames] = concatenate_fvols(subject,subject_directory)
subject_filenames = {};
volumes_paths = {};

for i = 1:n_functional_volumes %grab the subject-specific full filenames of each raw functional volume
   subject_files = dir(strcat(subject_directory,'*.nii')); %add specific name element of functional volumes
   subject_filenames{end+1,1} = subject_files(i).name;
end
volumes_paths = cellstr(string(strcat(subject_directory, char(subject_filenames))));
end
%% Function to grab structural scan volumes for one given MS or WU subject, provided directory
function [volumes_path,subject_filename] = grab_svol(subject, subject_directory)
subject_filename = {};
volumes_path = {};
subject_file = dir(strcat(subject_directory,'*.nii')); %add specific name element of structural volumes
subject_filename = subject_file.name; % get full filename 
volumes_path = strcat(subject_directory, char(subject_filename));
end
