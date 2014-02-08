function run_slice_timing(study, tasklist, subjectlist)
%% created by AES 4/7/13
%% conducts slice timing on per study basis for all subjects in subject list. 
% currently to be implemented following realignment and normalization (e.g.
% looks for swrf images), but should make generalizable to different
% preprocessing orders

rootdir=['/mindhive/saxelab2/' study];
addpath(rootdir)
addpath('/mindhive/saxelab/scripts/aesscripts');
numslices=32;
sliceorder=[xxxxx]; % slice acquisition order, vector of integers, each integer referring to the slice number in the image file
%                (1=first), and the order of integers representing their temporal acquisition order
%% get this from nick/sheeba... just leave this ghettocoded since all studies using 32 generic will be same

refslice=[xxxxx];   %% slice for time 0 (ideally middle but I might do first so that everything is aligned to start of TR)

TR=2;
TA=2-(2/numslices); %time between last slice and next volume
timing(1)= TA/(numslices-1);    %additional information for sequence timing, timing(1) = time between slices, timing(2) = time between last slices and next volume
timing(2)= TR-TA;

prefix='a'     %filename prefix for corrected image files, defaults to 'a'


for t=1:length(tasklist)
task=tasklist{t};

runlist=listbolds(task, subjectlist);

for subject=1:length(subjectlist)
    subjectID=subjectlist{subject};
    runs=runlist{subject};
    subjdir=[rootdir '/' subjectID '/'];
    for r=1:length(runs)
       bold=runs{r};
       bolddir=[subjdir,'bold/0' num2str(bold)]
       cd(bolddir)
       p=spm_select('list', bolddir, '^swrf.*\.img$'); %nimages x ? Matrix with filenames
       %% not sure why you need this special filter instead of swrf*.img, but internets says you do
       ips=size(p); ips=ips(1);
       
       spm_slice_timing_amy(p, sliceorder, refslice, timing, prefix)
        
    end




end

end