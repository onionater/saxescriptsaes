function prep_for_mvpaptb_emobioloc(study, task, subjectlist, resultfolder, runsrange)
%created by AES 4/16/13, takes data from standard saxelab SPM organization
%and gets it ready for mvpa analysis using princeton toolbox

rootdir=['/mindhive/saxelab2/'];
studydir=[rootdir study '/'];
prefix='swrf'; % type of preprocessed image to look for
TR=2;
responsefunction='hrf';
minVoxels=20; % a given subject's ROI needs at least this many voxels to be used in analyses. if you are going to use rank-based feature selection ultimately, would be most straightforward to just have this number be the same as whatever you decide to use as your other_args.fixednum in run_classification_EIB.m

if ~iscell(subjectlist)
    subjectlist={subjectlist};
end

numSubj=length(subjectlist);
runs=listbolds(task, subjectlist); % calls a simple little script in scripts/aesscripts that finds which bolds correspond to particular task for each subject (by calling a file called EIB_subject_taskruns.mat that lives in the main EIB directory)
fextension='.img'; %what kind of images are we looking for?


for s=1:numSubj

subject=subjectlist{s}

    %figure out what bold dirs this subject has
    boldlist=runs{s}; % get list of bolds for this task for this subject
    boldlist=boldlist(runsrange)    
    runsincluded=[int2str(runsrange(1)) 'to' int2str(runsrange(end))];
    if ~isempty(boldlist) % assuming they have some bold dir for this task, do...

numruns=length(boldlist)    
subjdir=[studydir subject '/'];
cd(subjdir)
mvpadir='mvpa_ptb/';
mkdir(mvpadir)
    
% get whole brain masks (this assumes we are using normalized data)
maskdir='3danat/';
mvpa_mask=dir([maskdir 'skull_strip_mask' fextension]);
mvpa_mask_description{1}='whole_brain_skullstripped';
mvpa_mask_name{1}=[maskdir mvpa_mask.name];


%hardcode the rois you are interested in

ROISind={'RTPJ_tomloc','LTPJ_tomloc','RSTS_tomloc','LSTS_tomloc','PC_tomloc', 'MMPFC_tomloc', 'DMPFC_tomloc', 'ROI_rpSTS_BDbiomot', 'ROI_rFFA_kanparcel_EmoBioLoc','ROI_lFFA_kanparcel_EmoBioLoc', 'ROI_rSTS_kanparcel_EmoBioLoc', 'ROI_lSTS_kanparcel_EmoBioLoc', 'ROI_rOFA_kanparcel_EmoBioLoc', 'ROI_lOFA_kanparcel_EmoBioLoc'};
roi_descriptions_ind={'RTPJ_tomloc_ind', 'LTPJ_tomloc_ind', 'RSTS_tomloc_ind', 'LSTS_tomloc_ind', 'PC_tomloc_ind', 'MMPFC_tomloc_ind', 'DMPFC_tomloc_ind', 'rpSTS_BDbiomot_ind', 'rFFA_kanparcel_EmoBioLoc','lFFA_kanparcel_EmoBioLoc', 'rSTS_kanparcel_EmoBioLoc', 'lSTS_kanparcel_EmoBioLoc', 'rOFA_kanparcel_EmoBioLoc', 'lOFA_kanparcel_EmoBioLoc'};
%group rois that stored elsewhere
ROISgroup={
    %'/mindhive/saxelab/roi_library/functional/EIBrois/ROI_rinsula_wfu_xyz.img',...
    %'/mindhive/saxelab/roi_library/functional/EIBrois/ROI_rvSTR_reward_xyz.img',...
    %'/mindhive/saxelab/roi_library/functional/EIBrois/ROI_vmPFC_reward_xyz.img',...
    %'/mindhive/saxelab/roi_library/functional/EIBrois/ROI_right_ant_temporal_xyz.img',...
    %'/mindhive/saxelab/roi_library/functional/EIBrois/ROI_ramygdala_wfu_xyz.img',...
    '/mindhive/saxelab/roi_library/functional/EIBrois/ROI_MPFC_peelenpeak_xyz.img',...
    '/mindhive/saxelab/roi_library/functional/EIBrois/ROI_lSTS_peelenpeak_xyz.img'
    %'/mindhive/saxelab/roi_library/functional/EIBrois/ROI_rSTS_peelenflip_xyz.img',...
    %'/mindhive/saxelab/roi_library/functional/EIBrois/ROI_lvSTR_reward_xyz.img',...
    %'/mindhive/saxelab/roi_library/functional/EIBrois/ROI_linsula_wfu_xyz.img',...
    %'/mindhive/saxelab/roi_library/functional/EIBrois/ROI_left_ant_temporal_xyz.img',...
    %'/mindhive/saxelab/roi_library/functional/EIBrois/ROI_lamygdala_wfu_xyz.img'
    };

roi_descriptions_group={
    %'rinsula_wfu_xyz_group',...
    %'rvSTR_reward_xyz_group',...
    %'vmPFC_reward_xyz_group',...
    %'right_ant_temporal_xyz_group',...
    %'ramygdala_wfu_xyz_group',...
    'MPFC_peelenpeak_xyz_group',...
    'lSTS_peelenpeak_xyz_group'
    %'rSTS_peelenflip_xyz_group',...
    %'lvSTR_reward_xyz_group',...
    %'linsula_wfu_xyz_group',...
    %'left_ant_temporal_xyz_group',...
    %'lamygdala_wfu_xyz_group'
    };


%roidir=[subjdir 'roi/'];
roidir=[subjdir 'autoROI/'];
numindrois=length(ROISind);
numgrouprois=length(ROISgroup);
count=1;
voxelcount(s,:)=zeros(1,numindrois+numgrouprois); %make a matrix of the number of voxels each subject has in each roi
voxelcountlabel=cell(1,numindrois+numgrouprois); %make an array of names corresponding to each of those rois

% get invidual rois
for roin=1:numindrois
        roiname=ROISind{roin};
        voxelcountlabel{roin}=roiname;
		searchROI=dir([roidir roiname '*' fextension]);
        if ~isempty(searchROI) %if the subject has this roi...
            roi_vol=spm_vol([roidir searchROI.name]);
            roi_mat=spm_read_vols(roi_vol);
            numVoxels=sum(roi_mat(:)); % find out how many voxels are in it
            voxelcount(s, roin)=numVoxels;
            if numVoxels>minVoxels %if it is more than your specified min
            count= count+1;
            mvpa_mask_description{count}=roi_descriptions_ind{roin}; %add it to the list of rois you'll use for this subject
            mvpa_mask_name{count}=[roidir searchROI.name];
            end
        end
end

% now get group rois
for roin=1:numgrouprois
    voxelcountlabel{roin+numindrois}=ROISgroup{roin};
    roi_vol=spm_vol(ROISgroup{roin});
    roi_mat=spm_read_vols(roi_vol);
    numVoxels=sum(roi_mat(:));
    count=count+1;
    mvpa_mask_name{count}=[ROISgroup{roin}];
    mvpa_mask_description{count}=roi_descriptions_group{roin}; 
    voxelcount(s, roin+numindrois)=numVoxels;
end
subjvoxelcount=voxelcount(s,:);


% get some design details from the results folders (this result folder is
% only being used to grab an SPM.mat to figure out the number of timepoints and the onsets, so
% it shouldn't matter what the modeling parameters were. there do need to exist 'art_regression_outliers*.mat')
load(['results/' resultfolder '/SPM.mat'])
numTimepoints=size(SPM.xY.VY,1);
IPS=numTimepoints/numruns;

%here we are just going to pull up run 1, cond 1 to get relevant info about number of conditions
%and durations of each 
%(ghetto constraint: the following assumes that the # of conditions in run 1 is the # of conditions in the whole exp, and that the duration of each condition is a constant across events)
    r=1;  
    taskStruct=SPM.Sess(r).U;
    numConds=size(taskStruct,2);
%initialize regressors and selectors matrices
    binarizedregressors=zeros(numTimepoints, numConds);
    blocklabelmatrix=zeros(numTimepoints, 1);
    rselectors=zeros(numTimepoints, 1); 
    art_selectors=ones(numTimepoints, 1);
    
    for c=1:numConds
        conditionNames{c}=SPM.Sess(r).U(c).name{:}; % these are all the names of conditions in this exp
        trialDurations{c}=ceil(SPM.Sess(r).U(c).dur(1)); %% assumes each event for a condition is same length across presentations, and rounds up duration up to nearest TR
    end

% now go through each run for this task    
for r=1:numruns
    run=boldlist(r);
    if mod(r,2)==0
        evenodd=2; %even
    else
        evenodd=1; %odd
    end
    %we are making one long design matrix but creating it run by run so we
    %need to know where in the matrix we should be adding values
    startRow=(r-1)*numTimepoints/numruns+1;
    endRow=(r-1)*numTimepoints/numruns+numTimepoints/numruns;
    binarizedrunmatrix=zeros(IPS,numConds); %matrix for a single run
    blockrunmatrix=zeros(IPS,1); % single vector that marks each event in the run with number corresponding to condition
    rselectors(startRow:endRow,:)=rselectors(startRow:endRow,:)+r; % selector vector specifying the run #
    eoselectors(startRow:endRow,:)=evenodd; %creating a selector vector specifying whether we are in an even or odd run

% go through each condition and find its duration (assumed to be constant) and its onsets    
for c=1:numConds
    condduration=trialDurations{c};
    condonsets=round(SPM.Sess(r).U(c).ons); %rounding assumes that all onsets are on TR. change this f ever jittering off TR
    numonsets=length(condonsets);
    binarizedrunmatrix(condonsets,c)=1; %for each onset, add a 1 to the appropriate condition column
    blockrunmatrix(condonsets,1)=c; %for each onset specify the condition
    %right now we only have a 1 at the onset of each event. go through and
    %add 1s to subsequent timepoints based on duration value
    for o=1:numonsets
        onset=condonsets(o);
        binarizedrunmatrix(onset:onset+(condduration-1),c)=1;
    end
end

binarizedregressors(startRow:endRow,:)=binarizedrunmatrix; %plop that single run matrix into the larger design matrix
blocklabelmatrix(startRow:endRow,1)=blockrunmatrix; %plot the single run vector into the whole timecourse vector

    % load art regressors and use to create art_selector vector (=0 for any
    % artifact timepoint)
    if ~isempty(dir(['bold/0' num2str(run) '/art_regression_outliers_sw*.mat']))
    matfile=adir(['bold/0' num2str(run) '/art_regression_outliers_sw*.mat']);
    load(matfile{1})
    runarts=R(:,:); %assumes that all columns are single timepoint exclusion vectors
    else if ~isempty(dir(['bold/0' num2str(run) '/art_regression_outliers_and_movement*sw*.mat']))
    matfile=adir(['bold/0' num2str(run) '/art_regression_outliers_and_movement*sw*.mat']);
    load(matfile{1})
    runarts=R(:,1:end-6); %assumes that the last six columns in this file are motion parameters and that the rest are single timepoint exclusion vectors
    else
       error('it looks like you do not have an art_regression_outliers file in your bold directory');
    end
    end
    runarts=sum(runarts,2); %sum across the regressors to get one vector with a 1 for every outlier timepoint
    art_selectors(startRow:endRow,1)=art_selectors(startRow:endRow,1)-runarts; %this vector is full of ones, so we are going to subtract out the outlier timepoints to make it 0 for every excluded timepoint
    art_selectors(art_selectors<1)=0; %erm, just in case?
end

% go through  blocklabel vector. if the event has a condition, relabel it
% with a separate number designating the event (this will be used later for
% creating event examples that average across timepoints)
count=0;
blocklabel=blocklabelmatrix;
for x=1:length(blocklabelmatrix)
       value=blocklabelmatrix(x); %this is the condition of this timepoint
       if value~=0 %if the vector has a value...
       count=count+1; %we've come across another example
       duration=trialDurations{value}; %figure out its duration
       blocklabel(x)=count; % h
       blocklabel(x:x+(duration-1))=count; %label that timepoint and subsequent timepoints as a new example
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



%% convolve binarized regressor with hrf (currently not using this for anything)
hrfinfo.dt=TR;
hrfinfo.name=responsefunction;
bf = spm_get_bf(hrfinfo);
for c=1:numConds
U.u=binarizedregressors(:,c);
U.name={'reg'}; % just because it needs a name
convolvedregressors(:,c) = spm_Volterra(U, bf.bf);
end

%make regressors for specific descriminations of interest
%by using combblockselector we are ignoring any example that has a single
%timepoint missing as an art outlier

%%
% pilot subj conditions: Bio, ObjMot, Happy, Sad
%all subj after subj 4 ran on full task: Bio, ObjMot, Happy, Sad, Faces, Objects
allcond=reduce_regressors(binarizedregressors, convolvedregressors, [1 2 3 4 5 6], {'Bio', 'ObjMot', 'Happy', 'Sad', 'Faces', 'Objects'}, 'none', blocklabel, combblockselector);
bioposVSbioneg=reduce_regressors(binarizedregressors, convolvedregressors, [0 0 1 2 0 0], {'Happy', 'Sad'}, 'none', blocklabel, combblockselector);
bioVSemo=reduce_regressors(binarizedregressors, convolvedregressors, [1 0 2 2 0 0], {'Bio', 'Emo'}, 'none', blocklabel, combblockselector);
facesVSobjects=reduce_regressors(binarizedregressors, convolvedregressors, [0 0 0 0 1 2], {'Faces', 'Object'}, 'none', blocklabel, combblockselector);
bioVSobjmot=reduce_regressors(binarizedregressors, convolvedregressors, [1 2 0 0 0 0], {'Bio', 'ObjMot'}, 'none', blocklabel, combblockselector);



%%list of all the bold images to use
boldimages=mat2cell(SPM.xY.P, ones(numTimepoints,1));

%this is just some silliness to ensure intuitive naming that matches with
%other scripts. should just fix these above.
boldnames=boldimages;
maskname=mvpa_mask_name;
allconds_convolved=convolvedregressors'; %% note these are transposed to be appropriate orientation for the toolbox
allconds_binarized=binarizedregressors'; %% ditto
blocklabels=blocklabel;
allcond_names=conditionNames;
runselectors=rselectors;
artselectors=art_selectors;
restselectors=restregressor;
combselectors=combinedselector;
blocklabelscomb=combblockselector;
evenoddselectors=eoselectors;
clearvars boldimages mvpa_mask_name convolvedregressors binarizedregressors blocklabel conditionNames selectors art_selectors restregressor combinedselector combblockselector eoselector

%print some stuff just to visualize/sanity check
%p= imagesc([blocklabels/max(blocklabels) blocklabelscomb/max(blocklabelscomb) artselectors/max(artselectors) restselectors/max(restselectors) combselectors/max(combselectors) runselectors/max(runselectors) evenoddselectors/max(evenoddselectors) stimselector/max(stimselector) contextselector/max(contextselector) genderselector/max(genderselector)]);
plotselectors=gcf;
saveas(plotselectors, [mvpadir 'selectorvis']);
clear gcf
close all
p=imagesc([(allconds_binarized*.5)' allconds_convolved']);
plotregressors=gcf;
saveas(plotregressors, [mvpadir 'regressorvis']);
clear gcf
close all

%save the relevant structures for the subject
 save([mvpadir 'subjinfo_EmoBioLoc_' runsincluded], 'boldnames', 'maskname', 'mvpa_mask_description', 'allconds_binarized', 'allconds_convolved',  'allcond_names','blocklabels', 'blocklabelscomb', 'runselectors', 'evenoddselectors', 'artselectors', 'restselectors', 'combselectors', 'TR', 'subjvoxelcount', 'voxelcountlabel');
 save([mvpadir 'discriminations_EmoBioLoc_' runsincluded], 'allcond', 'bioposVSbioneg', 'facesVSobjects', 'bioVSemo', 'bioVSobjmot');
    end
clearvars 'boldnames' 'maskname' 'mvpa_mask_description' 'allconds_binarized' 'allconds_convolved' 'allcond_names' 'blocklabels'  'blocklabelscomb'  'runselectors'  'evenoddselectors' 'artselectors' 'restselectors' 'combselectors'
clearvars 'allcond' 'bioposVSbioneg' 'facesVSobjects' 'bioVSemo' 'bioVSobjmot'
end
    voxelcount(voxelcount==0)=nan;
    voxelcount(numSubj+1,:)=nanmean(voxelcount); %add a row with mean # of voxels in the ROI across subjects
    voxelcount(numSubj+2,:)=min(voxelcount); %add a row with min # of voxels in the ROI in any subject
    save('/mindhive/saxelab2/EIB/mvpaptb/voxelcountsummary.mat', 'voxelcount', 'voxelcountlabel'); %save a general summary of the rois across subjects
end

function [reduced]=reduce_regressors(bregressors, cregressors, newcondindices, newcondnames, acrosssel,blocklabel, combblockselector)
%this takes the main regressor and makes a regressor expressing the
%descrimination of interest
reducedNum=length(newcondnames);
reduced.names=newcondnames;

for cond=1:reducedNum
   indices=find(newcondindices==cond); %figure out all the columns that correspond to conditions that will be part of your "reduced" condition
   thiscondregressors=bregressors(:,indices); 
   thiscondconvolved=cregressors(:,indices);
   summedregressors=sum(thiscondregressors,2); %sum across those columns to get one vector for the new condition
   summedconvolved=sum(thiscondconvolved,2);
   reg(:,cond)=summedregressors;
   regc(:,cond)=summedconvolved;
end

isanexample=sum(reg,2); % 1 for every timepoint that is part of an example in this discrimination
reducedlabel=blocklabel.*isanexample;
oldmax=max(blocklabel); %how many examples were in the full dataset
count=0;
%go through every old example and see if it is in the current set of
%examples
for c=1:oldmax
    temp=find(reducedlabel==c);
    if ~isempty(temp) %when you find one, relabel it with a new example counter
       count=count+1;
       reducedlabel(reducedlabel==c)=count;
    end
end

%do the same thing but for the combblock selector
reducedcomblabel=combblockselector.*isanexample;
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
    selector=0*sum(bregressors,2); %make vector of zeros of same length as regressors
    for lb=1:numlabels
       indices=find(labels==lb); % get indices of all conditions corresponding to this label
       thisselector=bregressors(:,indices);
       summedselector=sum(thisselector,2); %sum together regressors corresponding to those conditions
       selector(summedselector==1)=lb; %label selector vector appropriately
    end
end