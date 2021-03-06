function checkclassguesses(subjectlist,mvparesults, disc)
%%don't just run, read and change specifics below
rootdir='/mindhive/saxelab2/EIB/'
findstart=length([rootdir 'SAX_EIB_xx/mvpa_ptb/' mvparesults '/x'])
numSubj=length(subjectlist);
classfile='MMPFC_tomloc_ind_negVSposONE.binarizedreg_crossrunsONEselector_classification.mat';
if isempty(classfile)
classfile=spm_select();
classfile=classfile(findstart:end);
end
if strcmp(disc,'negVSpos') || strcmp(disc,'negVSposONE') || strcmp(disc,'negVSposTWO')
    stimlabels='stimlabels_nums';
    responses='stimresponses';
else if strcmp(disc,'negfVSposf')
    stimlabels='persons_stimlabels_nums';
    responses='persons_stimresponses';
else if strcmp(disc,'negcVSposc')'
    stimlabels='context_stimlabels_nums';
    responses='context_stimresponses';
end
end
end

subjconcat=[]
subjectlabellist=[]
scount=0
for s=1:numSubj
    subject=subjectlist{s}
    subjdir=[rootdir subject '/mvpa_ptb/']

trialinfofile=[subjdir 'trialinfo_EIB_main_1to8.mat']
load(trialinfofile);

tryclass=dir([subjdir mvparesults '/' classfile]);
if ~isempty(tryclass)
    scount=scount+1
load([[subjdir mvparesults '/' classfile]])
newclassfile=classfile(1:end-19);
newclassfile=[newclassfile '_singletrialguesses.mat']
numiterations=length(results.iterations)
accuracy=eval(stimlabels).*0;
accuracy(accuracy==0)=NaN;
for i=1:numiterations
    numtestexamples=length(results.iterations(1,i).test_idx)
    examplecorrect=results.iterations(1,i).perfmet.corrects
    for e=1:numtestexamples
        accuracy(results.iterations(1,i).test_idx(e))=examplecorrect(e);
    end
end
numtrials=length(accuracy)
countit=0;
for stemp=1:numtrials
    if mod(stemp,2)==0
        countit=countit+1;
        subjlabel(countit,:)=subject;
    end
end

singletrialguesses(:,1)=accuracy;
singletrialguesses(:,2)=eval(stimlabels);
theseresponses=eval(responses);
theseresponses(theseresponses==0)=NaN;
singletrialguesses(:,3)=theseresponses

singletrialguesses=sortrows(singletrialguesses,2)
trialguesses(1:384/2,1)=NaN;
trialguesses(:,2)=[1:384/2];
trialguesses(:,3)=NaN;
for item=1:size(trialguesses,1)
    row=find(singletrialguesses(:,2)==item);
    row=singletrialguesses(row,:,:)
    row=nanmean(row)  
    trialguesses(item, 1)=row(1)
    trialguesses(item, 3)=row(3)
end
subjtrials(:,:,scount)=trialguesses;

subjectlabellist=[subjectlabellist; subjlabel];
subjconcat=[subjconcat; subjtrials(:,:,scount)]
end
end
meantrialguessacc=nanmean(subjtrials(:,1,:),3)
meantrialguessresponse=nanmean(subjtrials(:,3,:),3)
semtrialguessresponse=nanstd(subjtrials(:,3,:),1,3)/sqrt(scount)
save([rootdir 'mvpaptb/' newclassfile], 'trialguesses', 'subjectlabellist', 'subjconcat', 'meantrialguessacc', 'meantrialguessresponse', 'semtrialguessresponse')
end
