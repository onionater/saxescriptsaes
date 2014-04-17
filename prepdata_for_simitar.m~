function prepdata_for_simitar(study, task, subjectlist)
%% created by AES 4/8/13
% takes images from bold dirs, and task info from behaviourals and creates
% appropriate meta structure, examples, etc. for simitar

numSubj=length(subjectlist);


for subject=1:numSubj
    prepsubject(subject, study, task, subjectlist);
end
end


function prepsubject(subject, study, task, subjectlist)

subjectID=subjectlist{subject};
disp(['working on subject ' subjectID])

%% parameters:
rootdir = ['/mindhive/saxelab2/' study '/'];
subjectdir = [rootdir subjectID '/'];
mvpadir=[rootdir 'EIB_mvpa/'];
resultsdir ='_with_art_reg_results_normed_global'; % this specifies the modeling version you want to use (really will only matter for getting conditions from SPM.mat and for getting artifact regressors if I incorporate that)
filterType='swrf'; %% specifies images to use
hemodynamic_delay=6;
runs=listbolds(task, subjectlist);


    
    %figure out what bold dirs this subject has
    boldlist=runs{subject}; % get list of bolds for this task for this subject
    if ~isempty(boldlist) % assuming they have some bold dir for this task
    numruns=length(boldlist);
    % load SPM.mat file from results directory specified above
    resultsString=[task, resultsdir];
    load([subjectdir 'results/' resultsString '/SPM.mat']) 
    

%% get .imgs from bold dirs and make into single 4d data matrix, and single .img from 3danat to mask3D

% make meta mask: mask3D - a 3D (dimx x dimy x dimz) binary mask with 1 where brain voxels are and 0 everywhere else
brainmask=[subjectdir '/3danat/skull_strip_mask.img']; %% assumes you just want brain mask. should try other feature selection methods and integrate to this script.
maskfile=spm_vol(brainmask);
mask3D=spm_read_vols(maskfile);
disp('working on mask...')
meta = createMetaFromMask(mask3D,'radius',1);
disp('created mask')

% go through each run and add bold volumes to big data matrix:
% data4D - a 4D (dimx x dimy x dimz x dimt) dataset matrix, with <dimt> 3D volumes. this will be for the entire task timecourse, rather than each run separately 
% need to figure out how to incorporate art regressors/eliminate timepoints
% (should we eliminate an entire example or just eliminate bad timepoint and have examples with
% different numbers of timepoints)
for r=1:numruns
    bold=boldlist(r);
    disp(['bold directory 0' num2str(bold)]);
    bolddir=[subjectdir,'bold/0' num2str(bold)];
    cd(bolddir)
    filterstr=['^' filterType '.*\.img$']; %% not sure why you need this special filter instead of swrf*.img, but you do
    p=spm_select('list', bolddir, filterstr);
    ips=size(p); ips=ips(1);
    timepoints = length(SPM.Sess(r).row);
    if ips ~= timepoints
       disp('number of images does not equal timepoints in SPM.mat') 
    end

% make data matrix
 files = spm_vol(p);
 disp('generating data matrix for run')
 for i=1:ips
        rundata(:,:,:,i)=spm_read_vols(files(i));
 end
 
 % create stimulus examples from the data

[dimx,dimy,dimz,dimt] = size(rundata);
nvoxels = length(meta.indicesIn3D);

disp('generating condition labels for run')
taskStruct=SPM.Sess(r).U;
numConds=size(taskStruct); numConds=numConds(2);
for cond=1:numConds
    condStruct=SPM.Sess(r).U(cond);
    conditionNames{cond}=SPM.Sess(r).U(cond).name;
    numPresentations=length(SPM.Sess(r).U(cond).ons);
    for e=1:numPresentations
        event((cond-1)*numPresentations+e,1)=cond;
        event((cond-1)*numPresentations+e,2)=SPM.Sess(r).U(cond).ons(e);
       % in line below, if there were separate duration for each event, replace dur(1) with dur(e)
        event((cond-1)*numPresentations+e,3)=ceil(SPM.Sess(r).U(cond).dur(1)); %% decided to round up to nearest TR, in this case model events as 3 TRs
    end
end

numEvents= length(event(:,1)); % based on design matrix figure out how many events we have
runlabel=zeros(numEvents,1);
runlabel=runlabel+r;

orderedEvents=sortrows(event, 2); % sort matrix of events based on onset
stimlabels=orderedEvents(:,1);

examplesOnsets=orderedEvents(:,2)+hemodynamic_delay;

runexamples = zeros(numEvents,nvoxels); % we are treating the brain now as a vector of voxels (so examples = a numevents x nvoxels matrix)
% but change this based on num events

%make examples for the run
disp('creating examples matrix for run')
for t = 1:ips

    volume = rundata(:,:,:,t);
    if any(examplesOnsets==t) % if this timepoint is an onset
        eventcount=find(examplesOnsets==t); %which event is it?
        eventDur=ceil(orderedEvents(eventcount,3));
        for v=1:eventDur
            eventvolumes(:,:,:,v)=rundata(:,:,:,t+v-1); % get volumes for t through duration of event
        end
        eventAvg=mean(eventvolumes,4); %average them together across the 4th dimension (time)
    runexamples(eventcount,:) = eventAvg(meta.indicesIn3D); 
    end
end 

%concatenate all of these things across runs
 if r==1
    data=rundata;
    labelGroups=runlabel;
    labels=stimlabels;
    examples=runexamples;
    else 
    data=cat(4, data, rundata); 
    labelGroups=[labelGroups; runlabel];
    labels=[labels; stimlabels];
    examples=[examples; runexamples]; 
 end
cd ..
cd ..
end
[dimx,dimy,dimz,dimt] = size(data); %not sure I need this for anything
cd(mvpadir)
mkdir('simitar')
cd('simitar')
disp(['saving file for subject ' subjectID]);
savefile=['simdata_' subjectID '_' resultsString '.mat'];
save(savefile, 'examples', 'labels', 'labelGroups', 'meta')
    end
    cd ..
    cd ..
end
