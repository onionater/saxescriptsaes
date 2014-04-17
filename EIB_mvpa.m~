function EIB_mvpa(singlesubjresultsdir)

%e.g. EIB_mvpa('EIB_main_results_normed')
%e.g. EIB_mvpa('EIB_main_with_art_reg_results_normed')

%3/12/13 AES added functionality to do group and individual rois in same
%command, added option to run classification on spmT or con images, changed
%name of folder analyses save to to be more descriptive, call
%analyMVPA_amy and MVPA_withinAcross_amy which each have their own small
%changes. inputs to analyMPVA_amy differ in the addition of conORt as an
%input parameter. input to MVPA_withinACross_amy now includes whether
%provided ROIs are group or ind

saxestart SPM8

addpath /software/spm8
addpath /software/gablab/conn
addpath /software/spm_ss


%single ROIs
 indROIs = {
     'RTPJ_tomloc','RSTS_tomloc', 'LTPJ_tomloc', 'LSTS_tomloc', 'MMPFC_tomloc', 'DMPFC_tomloc', 'PC_tomloc',...  
     'rSTS_kanparcelFaceObj_EmoBioLoc', 'lSTS_kanparcelFaceObj_EmoBioLoc', 'rFFA_kanparcelFaceObj_EmoBioLoc', 'lFFA_kanparcelFaceObj_EmoBioLoc', 'rOFA_kanparcelFaceObj_EmoBioLoc', 'lOFA_kanparcelFaceObj_EmoBioLoc', 'rpSTS_BDbiomotBioObj_EmoBioLoc', 'lLOC_foundObjFace', 'rLOC_foundObjFace'
     };
 ind=1; %0 if no individual ROIs
 % group ROIs
 groupROIs={'lSTS_peelenpeak', 'MPFC_peelenpeak', 'rinsula_wfu', 'linsula_wfu', 'rvSTR_reward', 'lvSTR_reward','ramygdala_wfu','lamygdala_wfu', 'right_ant_temporal','left_ant_temporal','vmPFC_reward'};
    
 group=1; %0 if no group ROIs


ind_loc=''; %just leave this empty
group_loc ='/mindhive/saxelab2/EIB/Group_ROI';
parametric=0;
centering=0;
conORt='con_'; %or 'spmT_'
results=['results/', singlesubjresultsdir];


experiments=struct(...
    'name','EIB',...
    'pwd1','/mindhive/saxelab2/EIB/',...   %folder with participants 
    'pwd2',results,...   %inside each participant, path to .spm  
'data',{{'SAX_EIB_01','SAX_EIB_02','SAX_EIB_03','SAX_EIB_04','SAX_EIB_05','SAX_EIB_07','SAX_EIB_08','SAX_EIB_09'}}); %,'SAX_EIB_10','SAX_EIB_11','SAX_EIB_12','SAX_EIB_13','SAX_EIB_16', 'SAX_EIB_17', 'SAX_EIB_18', 'SAX_EIB_19', 'SAX_EIB_20'}}); %% full subject set
 %'data',{{'SAX_EIB_01','SAX_EIB_02','SAX_EIB_03','SAX_EIB_04','SAX_EIB_05','SAX_EIB_06','SAX_EIB_07','SAX_EIB_08','SAX_EIB_09','SAX_EIB_10','SAX_EIB_11','SAX_EIB_12','SAX_EIB_13','SAX_EIB_15',
 %'SAX_EIB_16', 'SAX_EIB_17', 'SAX_EIB_18', 'SAX_EIB_19', 'SAX_EIB_20'}}; %%
 %% first half set
 %'data',{{'SAX_EIB_01','SAX_EIB_02','SAX_EIB_03','SAX_EIB_04','SAX_EIB_05','SAX_EIB_06','SAX_EIB_07','SAX_EIB_08','SAX_EIB_09'}});
 %%
 %% second half set
 %'data',{{'SAX_EIB_10','SAX_EIB_11','SAX_EIB_12','SAX_EIB_13','SAX_EIB_15',
 %'SAX_EIB_16', 'SAX_EIB_17', 'SAX_EIB_18', 'SAX_EIB_19'}}); %%

%make string to add subject IDs to output names
numSubj=length(experiments.data);
firstSubj=experiments.data{1};
firstSubj=firstSubj(end-1:end);
lastSubj=experiments.data{numSubj};
lastSubj=lastSubj(end-1:end);
subjectString=['subjects' firstSubj 'to' lastSubj];
 
savedirectory = ['/mindhive/saxelab2/EIB/EIB_mvpa/MVPA_' singlesubjresultsdir '_' conORt subjectString];
mkdir(savedirectory);


partition_names={'ODD_','EVEN_'};

condition_names_all={
    {'pos_all_vsr','neg_all_vsr'},...
    {'pos_p_vsr','neg_p_vsr'},...
    {'pos_c_vsr','neg_c_vsr'},...
    {'fake_pos_vsr','fake_neg_vsr'},...
    {'possocial_vsr', 'negsocial_vsr'},...
    {'posnonsoc_vsr', 'negnonsoc_vsr'},...
    {'posmale_vsr', 'negmale_vsr'},...
    {'posfema_vsr', 'negfema_vsr'},...
    {'fake_poscontext_vsr', 'fake_negcontext_vsr'},...
    {'fake_posperson_vsr', 'fake_negperson_vsr'},...
    {'social_vsr', 'nonsoc_vsr'},...
    {'cartoon_vsr', 'persons_vsr'},... 
    {'gendermale_vsr', 'genderfema_vsr'},...
    {'pos_all_vso','neg_all_vso'},...
    {'pos_p_vso','neg_p_vso'},...
    {'pos_c_vso','neg_c_vso'},...
    {'fake_pos_vso','fake_neg_vso'},...
    {'possocial_vso', 'negsocial_vso'},...
    {'posnonsoc_vso', 'negnonsoc_vso'},...
    {'posmale_vso', 'negmale_vso'},...
    {'posfema_vso', 'negfema_vso'},...
    {'fake_poscontext_vso', 'fake_negcontext_vso'},...
    {'fake_posperson_vso', 'fake_negperson_vso'},...
    {'social_vso', 'nonsoc_vso'},...
    {'cartoon_vso', 'persons_vso'},... 
    {'gendermale_vso', 'genderfema_vso'}
    };


if ind
for i=1:length(indROIs)
    roin = indROIs{i}
    roinum = i;
    addpath('/mindhive/saxelab/scripts/aesscripts/')
    analyMVPA_amy(roin, experiments,partition_names,condition_names_all,savedirectory, roinum, 0, ind_loc,parametric, conORt, centering);
end
addpath('/mindhive/saxelab/scripts/aesscripts/')
MVPA_withinAcross_amy(indROIs, 'ind')
end

if group
for i=1:length(groupROIs)
    roin = groupROIs{i}
    roinum = i;
    addpath('/mindhive/saxelab/scripts/aesscripts/')
    analyMVPA_amy(roin, experiments,partition_names,condition_names_all,savedirectory, roinum, 1, group_loc,parametric, conORt, centering);
end
addpath('/mindhive/saxelab/scripts/aesscripts/')
MVPA_withinAcross_amy(groupROIs, 'group')
end

end
