%%Script to run the marsbar ROI values extraction. 
% Takes SPM.mat data from completed SPM 1st level analysis. 

%%%%%%%

spm('defaults', 'fmri');
marsbar('on');


output_data_dir = 'D:\\Path\\To\\Final\\Outputs\\' %final .csv output will go there

design_dir = 'D:\\Path\\To\\Global\\SPM\\1stLevel\\Directory\\';% contains folders for each subject named as subject, themselves containing 
% subject SPM design (i.e. \\subject001\\SPM.mat file)
subject_list = importdata('D:\Path\To\Subject\List.txt');
roi_dir = 'D:\\Path\\To\\ROI-containing\\Directory'; %need .mat ROI files
roi_names = importdata('D:\Path\To\List\Of\ROI\Names.txt'); %need .mat ROI names

%contrasts in spm design and names in order
con_names = {'PhasicFear_vs_NoFear','NoFear_vs_PhasicFear','SustainedFear_vs_NoFear','NoFear_vs_SustainedFear', 'ActiveBlocks_vs_InactiveBlocks'};

all_effect_sizes = [];

for crun = 1:length(subject_list)
    subject_effect_sizes = []
    subject = subject_list(crun)
    spm_name = strcat(char(design_dir), char(subject),'\\SPM.mat')
    
    % Make marsbar design object - we assume that the design has been estimated in SPM
    clear design
    design = mardo(spm_name);

    % Get contrasts from original SPM design
    xCon = get_contrasts(design);

    
    roi_effect_sizes = [];col_names = []; col_name_one_con = [];
    
    for roi_no = 1:length(roi_names)
        
        %Extract roi from roi list and make it a marsbar roi object
        roi_name = roi_names{roi_no};
        
        %assign roi object to current roi
        roi_file = load(fullfile(roi_dir, roi_name), 'roi');
        roi_file = roi_file.roi;
        %roi = maroi(roi_file);
        
        % Extract roi data into the design
        Y{roi_no} = get_marsy(roi_file, design, 'median');
        
        % Estimate design on ROI data
        E{roi_no} = estimate(design, Y{roi_no});
        
        % Put contrasts from original design back into design object
        E{roi_no} = set_contrasts(E{roi_no}, xCon);
        
        % get design betas
        b{roi_no} = betas(E{roi_no});
        
        % get stats for all contrasts into statistics structure
        marsS{roi_no} = compute_contrasts(E{roi_no}, 1:length(xCon));
        roi_effect_sizes = [roi_effect_sizes marsS{roi_no}.con];
        
    end

    %subject output

    subject_effect_sizes = reshape(roi_effect_sizes.',1, []); % reshape in row order (i.e. one contrast and all rois, then next contrast and all rois etc.)
    all_effect_sizes = [all_effect_sizes; subject_effect_sizes];

end
%changing roi_names without .mat again   
%roi_names = importdata("D:\\Prediction_SPIDER_VR\\Scripts\\conn_atlas_rois\\atlas_shortnames.txt");
roi_names = erase(roi_names,'.mat')

col_names = [];
for con = 1:length(con_names) % for naming purposes
    col_name_one_con = strcat(con_names(con), '_', roi_names);
    col_names = [col_names, col_name_one_con];
end
col_names = reshape(col_names, [1,5*length(roi_names)]) %reshape (i.e. one contrast and all rois, then next contrast and all rois etc.)
col_names = [col_names;repmat({','},1,numel(col_names))];
col_names = col_names(:)';
col_names = cell2mat(col_names);
col_names = ['Subjects', ',',  col_names];

%Outputs

final_results = fopen(strcat(output_data_dir, 'Effect_sizes_marsbar_ExampleCode.csv'),'w'); 
fprintf(final_results,'%s\n',col_names);
fclose(final_results);

%write data to end of file

writecell([subject_list, arrayfun(@num2str,all_effect_sizes, 'UniformOutput', false)], ...
        strcat(output_data_dir, 'Effect_sizes_marsbar_ExampleCode.csv'),'WriteMode','append');
