function prep_for_mvpaptb(study, task, subjectlist, resultfolder)
%created by AES 4/16/13, gets data from standard saxelab SPM organization
%ready for mvpa analysis using princeton toolbox

rootdir=['/mindhive/saxelab2/'];
studydir=[rootdir study '/'];
prefix='swarf'; % type of preprocessed image to look for
hemodynamic_delay=6;
TR=2;
responsefunction='hrf';
minVoxels=20; % a given subject's ROI needs at least this many voxels to be used in analyses

if ~iscell(subjectlist)
    subjectlist={subjectlist};
end

numSubj=length(subjectlist);
runs=listbolds(task, subjectlist);
fextension='.img';


for s=1:numSubj

subject=subjectlist{s}

    %figure out what bold dirs this subject has
    boldlist=runs{s}; % get list of bolds for this task for this subject
    if ~isempty(boldlist) % assuming they have some bold dir for this task, do...
    numruns=length(boldlist)
    
subjdir=[studydir subject '/'];
cd(subjdir)
mvpadir='mvpa_ptb/';
mkdir(mvpadir)
    
% get masks
maskdir='3danat/';
mvpa_mask=dir([maskdir 'ws*' fextension]);
mvpa_mask_description{1}='whole_brain_graymatter';
mvpa_mask_name{1}=[maskdir mvpa_mask.name];


%hardcode the rois you are interested in
ROISind={'ROI_RTPJ_tomloc','ROI_LTPJ_tomloc','ROI_RSTS_tomloc','ROI_LSTS_tomloc','ROI_PC_tomloc', 'ROI_MMPFC_tomloc', 'ROI_DMPFC_tomloc', 'ROI_rFFA_kanparcelFaceObj_EmoBioLoc','ROI_lFFA_kanparcelFaceObj_EmoBioLoc', 'ROI_rSTS_kanparcelFaceObj_EmoBioLoc', 'ROI_lSTS_kanparcelFaceObj_EmoBioLoc', 'ROI_rOFA_kanparcelFaceObj_EmoBioLoc', 'ROI_lOFA_kanparcelFaceObj_EmoBioLoc', 'ROI_rLOC_foundObjFace_EmoBioLoc', 'ROI_lLOC_foundObjFace_EmoBioLoc'};
roi_descriptions_ind={'RTPJ_tomloc_ind', 'LTPJ_tomloc_ind', 'RSTS_tomloc_ind', 'LSTS_tomloc_ind', 'PC_tomloc_ind', 'MMPFC_tomloc_ind', 'DMPFC_tomloc_ind','rFFA_kanparcelFaceObj_EmoBioLoc','lFFA_kanparcelFaceObj_EmoBioLoc', 'rSTS_kanparcelFaceObj_EmoBioLoc', 'lSTS_kanparcelFaceObj_EmoBioLoc', 'rOFA_kanparcelFaceObj_EmoBioLoc', 'lOFA_kanparcelFaceObj_EmoBioLoc', 'rLOC_foundObjFace_EmoBioLoc', 'lLOC_foundObjFace_EmoBioLoc'};


ROISgroup={
    '/mindhive/saxelab/roi_library/functional/EIBrois/ROI_rinsula_wfu_xyz.img',...
    '/mindhive/saxelab/roi_library/functional/EIBrois/ROI_rvSTR_reward_xyz.img',...
    '/mindhive/saxelab/roi_library/functional/EIBrois/ROI_vmPFC_reward_xyz.img',...
    '/mindhive/saxelab/roi_library/functional/EIBrois/ROI_right_ant_temporal_xyz.img',...
    '/mindhive/saxelab/roi_library/functional/EIBrois/ROI_ramygdala_wfu_xyz.img',...
    '/mindhive/saxelab/roi_library/functional/EIBrois/ROI_MPFC_peelenpeak_xyz.img',...
    '/mindhive/saxelab/roi_library/functional/EIBrois/ROI_lSTS_peelenpeak_xyz.img',...
    '/mindhive/saxelab/roi_library/functional/EIBrois/ROI_lvSTR_reward_xyz.img',...
    '/mindhive/saxelab/roi_library/functional/EIBrois/ROI_linsula_wfu_xyz.img',...
    '/mindhive/saxelab/roi_library/functional/EIBrois/ROI_left_ant_temporal_xyz.img',...
    '/mindhive/saxelab/roi_library/functional/EIBrois/ROI_lamygdala_wfu_xyz.img'
    };

roi_descriptions_group={
    'rinsula_wfu_xyz_group',...
    'rvSTR_reward_xyz_group',...
    'vmPFC_reward_xyz_group',...
    'right_ant_temporal_xyz_group',...
    'ramygdala_wfu_xyz_group',...
    'MPFC_peelenpeak_xyz_group',...
    'lSTS_peelenpeak_xyz_group',...
    'lvSTR_reward_xyz_group',...
    'linsula_wfu_xyz_group',...
    'left_ant_temporal_xyz_group',...
    'lamygdala_wfu_xyz_group'
    };


%roidir=[subjdir 'roi/'];
roidir=[rootdir 'EIB/' subject '/roi/'];%temp so that rois can be found in EIB even though analyzing EIB_3mm
count=1;
% get invidual rois
for roin=1:length(ROISind)
        roiname=ROISind{roin};
		searchROI=dir([roidir roiname '*' fextension]);
        if ~isempty(searchROI)
            roi_vol=spm_vol([roidir searchROI.name]);
            roi_mat=spm_read_vols(roi_vol);
            numVoxels=sum(roi_mat(:));
            if numVoxels>minVoxels
            count= count+1;
            mvpa_mask_description{count}=roi_descriptions_ind{roin};
            [roidir searchROI.name];
            mvpa_mask_name{count}=[roidir searchROI.name];
            end
        end
end

% get group rois
for roin=1:length(ROISgroup)
    ROISgroup{roin};
    count=count+1;
    mvpa_mask_name{count}=[ROISgroup{roin}];
    mvpa_mask_description{count}=roi_descriptions_group{roin};
end


% get some design details from the results folders
%load(['results/' resultfolder '/SPM.mat'])
load(['/mindhive/saxelab2/EIB/' subject '/results/' resultfolder '/SPM.mat']) %temp to load from EIB
numTimepoints=size(SPM.xY.VY); numTimepoints=numTimepoints(1)
IPS=numTimepoints/numruns;

%just look at run 1, cond 1 to get relevant info about number of conditions
%and durations of each (ghetto constraint: assumes these are constant across runs and conditions)
    r=1;  
    taskStruct=SPM.Sess(r).U;
    numConds=size(taskStruct); numConds=numConds(2);
%initialize regressors and selectors matrices

    binarizedregressors=zeros(numTimepoints, numConds);
    blocklabelmatrix=zeros(numTimepoints, 1);
    rselectors=zeros(numTimepoints, 1); 
    art_selectors=ones(numTimepoints, 1);
    
    for c=1:numConds
        conditionNames{c}=SPM.Sess(r).U(c).name{:};
        trialDurations{c}=ceil(SPM.Sess(r).U(c).dur(1)); %% assumes each event for a condition is same length across presentations, and rounds up to nearest TR
    end

for r=1:numruns
    run=boldlist(r);
    if mod(r,2)==0
        evenodd=2; %even
    else
        evenodd=1; %odd
    end
    startRow=(r-1)*numTimepoints/numruns+1;
    endRow=(r-1)*numTimepoints/numruns+numTimepoints/numruns;
    binarizedrunmatrix=zeros(IPS,numConds);
    blockrunmatrix=zeros(IPS,1); % single vector that marks each event with number corresponding to condition
    rselectors(startRow:endRow,:)=rselectors(startRow:endRow,:)+r;
    eoselectors(startRow:endRow,:)=evenodd;

for c=1:numConds
    condduration=trialDurations{c};
    condonsets=SPM.Sess(r).U(c).ons;
    numonsets=length(condonsets);
    binarizedrunmatrix(condonsets,c)=1;
    blockrunmatrix(condonsets,1)=c;
    for o=1:numonsets
        onset=condonsets(o);
        binarizedrunmatrix(onset:onset+(condduration-1),c)=1;
    end
end

binarizedregressors(startRow:endRow,:)=binarizedrunmatrix;
blocklabelmatrix(startRow:endRow,1)=blockrunmatrix;

    % load art regressors and use to create art_selector vector (=0 for any
    % artifact timepoint)
    matfile=adir(['/mindhive/saxelab2/EIB/' subject '/bold/0' num2str(run) '/art_regression_outliers_sw*.mat']);
    %temp using EIB matfile=adir(['bold/0' num2str(run) '/art_regression_outliers_sw*.mat']);
    
    load(matfile{1})
    runarts=R(:,1:end-6);
    runarts=sum(runarts,2);
    art_selectors(startRow:endRow,1)=art_selectors(startRow:endRow,1)-runarts;
    art_selectors(art_selectors<1)=0;
end

% go through  blocklabel vector. if the event has a condition, relabel it
% with a separate number designating the event (this will be used for
% creating averaged event examples)
count=0;
blocklabel=blocklabelmatrix;
for x=1:length(blocklabelmatrix)
       if blocklabelmatrix(x)~=0
       count=count+1;
       blocklabel(x)=count;
       end
end

%the above just relabeled the onset, now go through and label all TRs in
%that event, based on event duration
for x=1:length(blocklabelmatrix)
    value=blocklabelmatrix(x);
    if blocklabelmatrix(x)~=0
    duration=trialDurations{value};
    count=blocklabel(x);
    blocklabel(x:x+(duration-1))=count;
    end
end

%make selector to exclude all rest points (=0 for any rest timepoint)
restregressor=blocklabel;
restregressor(restregressor>0)=1;

%make single selector so exclude all rest AND artifact points
combinedselector=zeros(numTimepoints, 1);
combinedselector(restregressor==1 & art_selectors==1)=1;

%make block selector that also excludes artifact points (already excludes
%rest points)

combblockselector=blocklabel;
combblockselector(art_selectors==0)=0;


% conditions: 'mu','fu','mh','fh','nu','su','nh','sh'
stimselector= makeselector(binarizedregressors, [1 1 1 1 2 2 2 2]);  %% first fold=faces, second = contexts
contextselector= makeselector(binarizedregressors, [0 0 0 0 1 2 1 2]); %% first fold = nonsocial, second=social
genderselector= makeselector(binarizedregressors, [1 2 1 2 0 0 0 0]); %% first fold = male, second = female

%% convolve binarized with hrf
hrfinfo.dt=TR;
hrfinfo.name=responsefunction;
bf = spm_get_bf(hrfinfo);

for c=1:numConds
U.u=binarizedregressors(:,c);
U.name={'reg'}; % just because it needs a name
convolvedregressors(:,c) = spm_Volterra(U, bf.bf);
end

alleight=reduce_regressors(binarizedregressors, convolvedregressors, [1 2 3 4 5 6 7 8], {'mu','fu','mh','fh','nu','su','nh','sh'}, 'none', blocklabel, combblockselector);
faceVScontext=reduce_regressors(binarizedregressors, convolvedregressors, [1 1 1 1 2 2 2 2], {'face', 'context'}, 'none', blocklabel, combblockselector);
negVSpos=reduce_regressors(binarizedregressors, convolvedregressors, [1 1 2 2 1 1 2 2], {'neg', 'pos'}, 'stimselector', blocklabel, combblockselector);
negfVSposf=reduce_regressors(binarizedregressors, convolvedregressors, [1 1 2 2 0 0 0 0], {'negf', 'posf'}, 'genderselector', blocklabel, combblockselector);
negcVSposc=reduce_regressors(binarizedregressors, convolvedregressors, [0 0 0 0 1 1 2 2], {'negc', 'posc'}, 'contextselector', blocklabel, combblockselector);
maleVSfemale=reduce_regressors(binarizedregressors, convolvedregressors, [1 2 1 2 0 0 0 0], {'male', 'female'}, 'none', blocklabel, combblockselector);
socialVSnonsoc=reduce_regressors(binarizedregressors, convolvedregressors, [0 0 0 0 1 2 1 2], {'nonsoc', 'social'}, 'none', blocklabel, combblockselector);
%% these are new and online included in 3mm for now

socialnVSsocialp=reduce_regressors(binarizedregressors, convolvedregressors, [0 0 0 0 0 1 0 2], {'socialn', 'socialp'}, 'none', blocklabel, combblockselector);
nonsocnVSnonsocp=reduce_regressors(binarizedregressors, convolvedregressors, [0 0 0 0 1 0 2 0], {'nonsocn', 'nonsocp'}, 'none', blocklabel, combblockselector);

%%

%%list of all the bold images to use
boldimages=mat2cell(SPM.xY.P, ones(numTimepoints,1));

%some silliness to ensure intuitive naming
boldnames=boldimages;
maskname=mvpa_mask_name;
eightconds_convolved=convolvedregressors'; %% note these are transposed to be appropriate orientation for the toolbox
eightconds_binarized=binarizedregressors'; %% ditto
blocklabels=blocklabel;
eightcond_names=conditionNames;
runselectors=rselectors;
artselectors=art_selectors;
restselectors=restregressor;
combselectors=combinedselector;
blocklabelscomb=combblockselector;
evenoddselectors=eoselectors;
clearvars boldimages mvpa_mask_name convolvedregressors binarizedregressors blocklabel conditionNames selectors art_selectors restregressor combinedselector combblockselector eoselector

%print some stuff just to visualize/sanity check
p= imagesc([blocklabels/max(blocklabels) blocklabelscomb/max(blocklabelscomb) artselectors/max(artselectors) restselectors/max(restselectors) combselectors/max(combselectors) runselectors/max(runselectors) evenoddselectors/max(evenoddselectors) stimselector/max(stimselector) contextselector/max(contextselector) genderselector/max(genderselector)]);
plotselectors=gcf;
saveas(plotselectors, [mvpadir 'selectorvis']);
clear gcf
close all
p=imagesc([(eightconds_binarized*.5)' eightconds_convolved']);
plotregressors=gcf;
saveas(plotregressors, [mvpadir 'regressorvis']);
clear gcf
close all

%save the relevant structures
 save([mvpadir 'subjinfo'], 'boldnames', 'maskname', 'mvpa_mask_description', 'eightconds_binarized', 'eightconds_convolved',  'eightcond_names','blocklabels', 'blocklabelscomb', 'runselectors', 'evenoddselectors', 'artselectors', 'restselectors', 'combselectors', 'stimselector', 'contextselector', 'genderselector', 'hemodynamic_delay', 'TR');
 save([mvpadir 'discriminations'], 'alleight', 'faceVScontext', 'negVSpos', 'negfVSposf', 'negcVSposc', 'maleVSfemale', 'socialVSnonsoc','socialnVSsocialp', 'nonsocnVSnonsocp');
    end
clearvars 'boldnames' 'maskname' 'mvpa_mask_description' 'eightconds_binarized' 'eightconds_convolved' 'eightcond_names' 'blocklabels'  'blocklabelscomb'  'runselectors'  'evenoddselectors' 'artselectors' 'restselectors' 'combselectors' 'stimselector' 'contextselector' 'genderselector'
clearvars 'alleight' 'faceVScontext' 'negVSpos' 'negfVSposf' 'negcVSposc' 'maleVSfemale' 'socialVSnonsoc' 'socialnVSsocialp' 'nonsocnVSnonsocp'
    end
end

function [reduced]=reduce_regressors(bregressors, cregressors, newcondindices, newcondnames, acrosssel,blocklabel, combblockselector)

reducedNum=length(newcondnames);
reduced.names=newcondnames;

for cond=1:reducedNum
   indices=find(newcondindices==cond);
   thiscondregressors=bregressors(:,indices);
   thiscondconvolved=cregressors(:,indices);
   summedregressors=sum(thiscondregressors,2);
   summedconvolved=sum(thiscondconvolved,2);
   reg(:,cond)=summedregressors;
   regc(:,cond)=summedconvolved;
end

reducedlabel=blocklabel.*sum(reg,2);
oldmax=max(blocklabel);
count=0;
for c=1:oldmax
    temp=find(reducedlabel==c);
    if ~isempty(temp)
       count=count+1;
       reducedlabel(reducedlabel==c)=count;
    end
end

reducedcomblabel=combblockselector.*sum(reg,2);
oldmax=max(combblockselector);
count=0;
for c=1:oldmax
    temp=find(reducedcomblabel==c);
    if ~isempty(temp)
       count=count+1;
       reducedcomblabel(reducedcomblabel==c)=count;
    end
end


reduced.binarizedreg=reg';
reduced.convolvedreg=regc';
reduced.crossselector=acrosssel;
reduced.averaginglabels=reducedlabel;
reduced.averaginglabelscomb=reducedcomblabel;
end

function selector= makeselector(bregressors, labels)
    numlabels=max(labels);
    selector=0*sum(bregressors,2); %lazy hack
    for lb=1:numlabels
       indices=find(labels==lb);
       thisselector=bregressors(:,indices);
       summedselector=sum(thisselector,2);
       selector(summedselector==1)=lb;
       
    end

end