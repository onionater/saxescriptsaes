function group_mvpaptbSL(subjectlist, classfolder, task, runsincluded, discnum, varargin)
%this script requires hard coding based on the contrasts you are interested in (and the selectors). fill in the contrast details structure with the relevant info%created by AES on 6/27 for group analysis of searchlight accuracy mapes
%within subjects averages across folds and creates crossfold avg map in
%each subject's directory
%then averages across subjects and makes group stat map (along with avg acc
%map and std map
%optional argument: {'chance', 0.25}
numSubj=length(subjectlist);
firstsubj=subjectlist{1}
lastsubj=subjectlist{numSubj}
subjectrange=['subj_' firstsubj(end-1:end) 'to' lastsubj(end-1:end)];
runRFX=1;

studydir='/mindhive/saxelab2/EIB/';
groupdir=([studydir '/RandomEffects/group_' classfolder, subjectrange '/'])
maskfile='/mindhive/saxelab2/EIB/SearchspacesGroupRois/binarized40percent_grey_matter_MNI_fromSPMapriori.img'

% these are references by disc num
contrastdetails(1).name='negfVSposf';
contrastdetails(2).name='negcVSposc';
contrastdetails(3).name='negVSposONE';
contrastdetails(4).name='negVSposTWO';
contrastdetails(1).selectors={'runs', 'crossmatchedruns'};
contrastdetails(2).selectors={'runs', 'crossmatchedruns'};
contrastdetails(3).selectors={'crossrunsONEselector'};
contrastdetails(4).selectors={'crossrunsTWOselector'};
contrastdetails(1).partitions={'runs'};
contrastdetails(2).partitions={'runs'};
contrastdetails(3).partitions={'even', 'odd'};
contrastdetails(4).partitions={'even', 'odd'};


mkdir(groupdir)
nonparametric=0; % see beginning of nonparametric impliementation below (but signrank is weird in matlab...)

chance=0.5; %default
if size(varargin)>0
    if strcmp(varargin{1}, 'chance')
        chance=varargin{2}
    else
        error('unknown parameter as argument')
    end
end

%just for subject 1, figure out the relevant discriminations (this assumes
%all subjects have same discriminations
       subjectID=subjectlist{1};
       subjdir=[studydir subjectID '/'];
       cd(subjdir)
       disc=load([subjdir 'mvpa_ptb/discriminations_' task '_' runsincluded]);
       discnames=fieldnames(disc) % disname input must match one of these
       numDisc=length(discnames);
%go through the discrimination

     discname=contrastdetails(discnum).name
     selectors=contrastdetails(discnum).selectors
     partitions=contrastdetails(discnum).partitions
            for s=1:length(selectors)
                selector=selectors{s}
            for p=1:length(partitions) 
                partition=partitions{p};
           filestring=[discname '.binarizedreg_' selector '_train*' partition '*.img'];
%load mask
maskfileinfo=spm_vol(maskfile);
mask=spm_read_vols(maskfileinfo);            

%go through each subject and make their crossfold avg .imgs
subjcount=0;
for s=1:length(subjectlist)
       subjectID=subjectlist{s};
       subjdir=[studydir subjectID '/'];
       cd(subjdir)
            mvpadir=[subjdir 'mvpa_ptb/' classfolder '/'];
            cd(mvpadir)
            imgnames=dir(filestring);
            numfolds=size(imgnames,1)/2;
            if numfolds>0
                subjcount=subjcount+1;
            imgstring=imgnames(1).name(1:end-11);
            p=spm_select('list', mvpadir, ['^' imgstring '.*\.img$']); %% not sure why you need this special filter instead of swrf*.img, but you do

                files = spm_vol(p);
                for i=1:numfolds
                   i
                   files(i).fname
                if files(i).fname(end-19:end-14) ~='minus0'
                data(:,:,:,i)=spm_read_vols(files(i));
                data(data==0)=NaN;
                end
                end
                voxelwiseAvg(:,:,:,subjcount)=mean(data,4).*mask; % 
                voxelwiseDiffFromChance(:,:,:,subjcount)=(mean(data,4)-chance).*mask;
                clearvars data
                
                
            writeTemplate=files(1); % getting template from a single fold classification .img, keeping pinfo
            classificationpinfo=writeTemplate.pinfo
            writeTemplate.dt = [spm_type('float64') spm_platform('bigend')];
            writeTemplate.fname = [subjectID '_' discname '_' selector '_train' partition '_crossfoldMEANACC.img'];
            avg_singleoutput.fname = [subjectID '_' discname '_' selector '_train' partition '_crossfoldMEANACC.img'];
            avg_singleoutput=spm_create_vol(writeTemplate);
            avg_singleoutput = spm_write_vol(avg_singleoutput, voxelwiseAvg(:,:,:,subjcount));        
            copyfile([subjectID '_' discname '_' selector '_train' partition '_crossfoldMEANACC.img'], [groupdir 'IND_' subjectID '_' discname '_' selector '_train' partition '_crossfoldMEANACC.img'])
            copyfile([subjectID '_' discname '_' selector '_train' partition '_crossfoldMEANACC.hdr'], [groupdir 'IND_' subjectID '_' discname '_' selector '_train' partition '_crossfoldMEANACC.hdr'])
           
            writeTemplate.fname = [subjectID '_' discname '_' selector '_train' partition '_crossfoldDiffChance.img'];
            avg_singleoutput.fname = [subjectID '_' discname '_' selector '_train' partition '_crossfoldDiffChance.img'];
            avg_singleoutput=spm_create_vol(writeTemplate);
            avg_singleoutput = spm_write_vol(avg_singleoutput, voxelwiseDiffFromChance(:,:,:,subjcount));        
            copyfile([subjectID '_' discname '_' selector '_train' partition '_crossfoldDiffChance.img'], [groupdir 'IND_' subjectID '_' discname '_' selector '_train' partition '_crossfoldDiffChance.img'])
            copyfile([subjectID '_' discname '_' selector '_train' partition '_crossfoldDiffChance.hdr'], [groupdir 'IND_' subjectID '_' discname '_' selector '_train' partition '_crossfoldDiffChance.hdr'])
 
            end
            
end

%now do group analysis if there is at least one subject
if runRFX
if subjcount>1;
cd(groupdir)

if nonparametric
dim=size(voxelwiseAvg)
groupW=zeros(dim(1:3));
 for x=1:dim(1)
     for y=1:dim(2)
         for z=1:dim(3)
             for subj=1:dim(4)
                 vector(subj)=voxelwiseAvg(x,y,z,subj);
             end
             if sum(isnan(vector))==0
             [p, h, stat]=signrank(vector, .5);
             stat.signedrank
             groupW(x,y,z)=stat.signedrank;
             end
         end
     end
 end
end
                groupAvg=nanmean(voxelwiseAvg,4);
                groupSTD=nanstd(voxelwiseDiffFromChance,0,4);
                groupSTD(groupSTD==0)=NaN;
                groupAvg(groupAvg==0)=NaN;
                groupSE=groupSTD./sqrt(subjcount);
                maskgroup=mask(:,:,:,1);

                
                groupSE=groupSE.*mask;
                groupDiff=groupAvg-chance;
                groupT=groupDiff./groupSE;
                groupDiff=groupDiff.*mask;
                groupT=groupT.*mask;
                
            inputfiles={'groupAvg', 'groupDiff', 'groupT', 'groupSE',}; 
            outputfiles={'AVG_output', 'Diff_output', 'T_output', 'SE_output',};  
            outputfilenames={'_crosssubjMEANACC.img','_crosssubjACCminus50.img', '_crosssubjT.img', '_crosssubjSE.img',};
            for x=1:length(outputfiles)
            outputfile=outputfiles{x};    
            writeTemplate.fname = [discname '_' selector '_train' partition '_' outputfilenames{x}];
            name=[discname '_', selector '_train' partition '_' outputfilenames{x}];
            eval([outputfile '.fname = name;'])
            eval([outputfile, '=spm_create_vol(writeTemplate);'])
            eval([outputfile,' = spm_write_vol(' outputfile ', ',inputfiles{x},'(:,:,:));']) 
            
            end
end
end
            end
           
end

