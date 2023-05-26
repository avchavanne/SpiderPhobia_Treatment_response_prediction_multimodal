%CONN 1st-level analyses for ROI-to-ROI gPPI

%%%%%%%

clear
conn_project_path = 'D:\Path\To\CONN\Project'; %path to CONN project folder
load(conn_project_path, 'CONN_x')
subject_list = importdata('D:\Path\To\Subject\List.txt');

contrasts = [[1 0 -1 0 0 0];[-1 0 1 0 0 0];[0 1 -1 0 0 0];[0 -1 1 0 0 0]];
main_effects = [[1 0 0 0 0 0];[0 1 0 0 0 0];[0 0 1 0 0 0];[0 0 0 1 0 0];[0 0 0 0 1 0];[0 0 0 0 0 1]];
cont_names = {'PhasicFearvsNoFear' 'NoFearvsPhasicFear' 'SustainedFearvsNoFear' 'NoFearvsSustainedFear'};
cond_names = {'Phasic Fear'	'Sustained Fear'	'No Fear'	'Inactive Block'	'Instructions'	'Rating'};

roi_dir = 'D:\\Path\\To\\ROI-containing\\Directory'; %path to directory containing all ROI data
roi_names = importdata('D:\Path\To\List\Of\ROI\Names.txt'); %path to .txt file containing list of ROI names
roi_names = append(roi_names, '.nii'); %need nifti format for rois
all_rois = roi_names.';


%% Analysis
%ROI-to-ROI only
batch(1).filename = strcat(conn_project_path,'.mat');
batch(1).folders.rois = roi_dir;
batch(1).Analysis.done = 1;
batch(1).Analysis.overwrite = 1;
batch(1).Analysis.name = ['gPPI']; 
batch(1).Analysis.measure = 3;
batch(1).Analysis.weight = 2; %weighted with hrf within-condition
batch(1).Analysis.modulation = 1;% gppi
batch(1).Analysis.conditions = []; %take existing conditions
batch(1).Analysis.type = 1; %  1 ROI-to-ROI 
batch(1).Analysis.sources = all_rois;


conn_batch(batch)
clear batch



%% Results

%gPPI analysis
for i = 1:height(contrasts)
batch(i).filename = strcat(conn_project_path,'.mat');
batch(i).parallel.N = 0;
batch(i).Results.name = ['gPPI'];
batch(i).Results.saveas = cont_names{i};
batch(i).Results.between_conditions.effect_names = cond_names;
batch(i).Results.between_conditions.contrast = contrasts(i,:);
batch(i).Results.between_subjects.effect_names = {'AllSubjects'};
batch(i).Results.between_subjects.contrast=[1];
batch(i).Results.between_sources.effect_names = [];
batch(i).Results.between_sources.contrast = [];
batch(i).Results.done = 1;
batch(i).Results.overwrite = 1;
batch(i).Results.display = 1;
end

conn_batch(batch)
clear batch


%% Take 1st level connectivity matrices and save them with

roi_names = importdata(strcat(roi_dir,'ROI_names.txt')); %import initial ROI names to produce column names
colnames = [];

for iroi  = 1:numel(roi_names)
   for jroi = 1:numel(roi_names)
       colnames = [colnames, strcat(roi_names(iroi),'_',roi_names(jroi))];
   end
end
 
%gPPI

Z_matrix = load(strcat(conn_project_path,'\results\firstlevel\gPPI\resultsROI_Condition001.mat'), 'Z');
Z_matrix = Z_matrix.Z;
gPPI = reshape(Z_matrix,length(roi_names)*length(roi_names),length(subject_list));  gPPI = gPPI.'; gPPI = [colnames; string(gPPI)]; 
gPPI = [['Subjects'; subject_list],gPPI];
writematrix(gPPI, strcat(conn_project_path,'\gPPI_ExampleCode.csv'));

