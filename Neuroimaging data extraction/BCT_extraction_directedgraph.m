% Use BCT to extract graph measures for an directed weighted graph (gPPI connectivity), i.e. the 
% result of CONN 1st-level, for one given condition. 

%%%%%%%

clear
conn_project_path = 'D:\Path\To\CONN\Project'; %path to CONN project folder
data = load(strcat(conn_project_path, '\results\firstlevel\gPPI\resultsROI_Condition001.mat'), 'Z');
data = data.Z;
roi_names = importdata('D:\Prediction_SPIDER_VR\Scripts\SPM\Marsbar_extraction\ROIs\ROI_names.txt');
roi_names = roi_names.';
subject_list = importdata('D:\Path\To\Subject\List.txt');
output_data_dir = 'D:\Path\To\Final\Outputs\'; %final .csv output will go there


all_subject_measures_struct = {};
all_subject_measures = [];
colnames = [];
for nsubject = 1:length(subject_list); 
    
%%network construction
    Z_subject = data(:,:,nsubject);
    fix_Z = weight_conversion(Z_subject, 'autofix');
    thr_fix_Z = threshold_absolute(fix_Z, 0.3);
    bin_thr_fix_Z = weight_conversion(thr_fix_Z, 'binarize');
    L_thr_fix_Z = weight_conversion(thr_fix_Z,'lengths');
    
%%network measures

    %Degree

    [~,~,deg] = degrees_dir(thr_fix_Z);
    all_subject_measures_struct{nsubject}.degree = [append('degree_',roi_names); string(deg)];

    %Strength

    all_subject_measures_struct{nsubject}.strength = [append('strength_',roi_names); string(strengths_dir(thr_fix_Z))];

    %Density

    [Density,~,~] = density_dir(thr_fix_Z);
    all_subject_measures_struct{nsubject}.density = ['density'; string(Density)];

    %Clustering coefficient

    all_subject_measures_struct{nsubject}.clustering_coef = [append('clustering_coef_',roi_names); string(clustering_coef_wd(thr_fix_Z).')];

    %Transitivity

    all_subject_measures_struct{nsubject}.transitivity = ['transitivity'; string(transitivity_wd(thr_fix_Z))];

    %Local efficiency
    
    all_subject_measures_struct{nsubject}.local_eff = [append('local_eff_',roi_names); string(efficiency_wei(thr_fix_Z,2).')];

    %Assortativity

    all_subject_measures_struct{nsubject}.assortativity = ['assortativity'; string(assortativity_wei(thr_fix_Z,0))]; %change 0 if directed graph

    %Core/periphery structure % 1 for nodes in the core, 0 in the periphery

    all_subject_measures_struct{nsubject}.core_periph = [append('core_periph_',roi_names); string(core_periphery_dir(thr_fix_Z)+0)]; 
    
  
    %Characteristic path length, global efficiency, eccentricity, radius, diameter: The characteristic path length is the average shortest path length
    %in the network. The global efficiency is the average inverse shortest path length in the network. The node eccentricity is the maximal shortest 
    %path length between a node and any other node. The radius is the minimum eccentricity and the diameter is the maximum eccentricity.

    all_subject_measures_struct{nsubject}.charpath = ['char_path_length'; string(charpath(thr_fix_Z))];


    %Global and local efficiency: The global efficiency is the average inverse shortest path length in the network, and is inversely related
    % to the characteristic path length. The local efficiency is the global efficiency computed on the neighborhood of the node, and is
    % related to the clustering coefficient.

    all_subject_measures_struct{nsubject}.global_eff = ['global_eff'; string(efficiency_wei(thr_fix_Z))]; 

    %Betweenness centrality

    all_subject_measures_struct{nsubject}.betweenness_centrality = betweenness_wei(L_thr_fix_Z);
    all_subject_measures_struct{nsubject}.betweenness_centrality = [append('betweenness_centrality_', roi_names); ...
                                                    string(all_subject_measures_struct{nsubject}.betweenness_centrality.')];
   
    %K-coreness centrality

    [coreness, ~] = kcoreness_centrality_bd(bin_thr_fix_Z);
    all_subject_measures_struct{nsubject}.kcoreness = [append('kcoreness_', roi_names); string(coreness)];

    %Flow coefficient

    [fc,FC,~] = flow_coef_bd(bin_thr_fix_Z);
    all_subject_measures_struct{nsubject}.flow_coef = [[append('flow_coef_', roi_names), {'global_flow'}];[string(fc),string(FC)]];
                                        
    %Shortcuts
    
    [~,eta,~,fs] = erange(bin_thr_fix_Z);
    all_subject_measures_struct{nsubject}.shortcuts = [[string('average_range'), string('fraction_shorcuts')]; [string(eta), string(fs)]];

    %Structural motifs:
    
    cd(conn_project_path);
    make_motif34lib; % have to save things directly in current dir 
    [I,Q,F] = motif3struct_wei(thr_fix_Z);
    intensity = []; coherence = []; fingerprint = []; motif_intensity = []; motif_coherence = []; motif_fingerprint = [];
    for motif = 1:size(I,1); 
        motif_intensity = [append(strcat('s_motif_intensity_', string(motif),'_', roi_names)); I(motif,:)];
        motif_coherence = [append(strcat('s_motif_intensity_', string(motif),'_', roi_names)); Q(motif,:)];
        motif_fingerprint = [append(strcat('s_motif_intensity_', string(motif),'_', roi_names)); F(motif,:)];
        intensity = [intensity, motif_intensity];
        coherence = [coherence, motif_coherence];
        fingerprint = [fingerprint, motif_fingerprint];
    end
    all_subject_measures_struct{nsubject}.s_motif_intensity = intensity;
    all_subject_measures_struct{nsubject}.s_motif_coherence = coherence;
    all_subject_measures_struct{nsubject}.s_motif_fingerprint = fingerprint;

    %Functional motifs:

    [I,Q,F] = motif3funct_wei(thr_fix_Z);
    intensity = []; coherence = []; fingerprint = []; motif_intensity = []; motif_coherence = []; motif_fingerprint = [];
    for motif = 1:size(I,1); 
        motif_intensity = [append(strcat('motif_intensity_', string(motif),'_', roi_names)); I(motif,:)];
        motif_coherence = [append(strcat('motif_intensity_', string(motif),'_', roi_names)); Q(motif,:)];
        motif_fingerprint = [append(strcat('motif_intensity_', string(motif),'_', roi_names)); F(motif,:)];
        intensity = [intensity, motif_intensity];
        coherence = [coherence, motif_coherence];
        fingerprint = [fingerprint, motif_fingerprint];
    end
    all_subject_measures_struct{nsubject}.f_motif_intensity = intensity;
    all_subject_measures_struct{nsubject}.f_motif_coherence = coherence;
    all_subject_measures_struct{nsubject}.f_motif_fingerprint = fingerprint;

%%Aggregate all measures
    all_fields = fieldnames(all_subject_measures_struct{1,1});
    subject_measures = [];
    
    for nfield = 1:numel(all_fields);
        if nsubject == 1; 
            colnames = [colnames, string(char(all_subject_measures_struct{1,1}.(all_fields{nfield}){1,:})).'];
        end;   
        subject_measures = [subject_measures, num2cell(char(all_subject_measures_struct{1,nsubject}.(all_fields{nfield}){2:end,:}),2).'];
    end;
    all_subject_measures = [all_subject_measures; subject_measures];
end
all_subject_measures = [colnames; all_subject_measures];
all_subject_measures = [['Subjects'; subject_list(1:nsubject,:)],all_subject_measures];
writematrix(all_subject_measures, strcat(output_data_dir,'Graph_measures_ExampleCode',string(nsubject),'.csv'));
