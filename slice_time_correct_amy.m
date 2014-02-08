function slice_time_correct_amy(study, subjectlist)
rootdir='/mindhive/saxelab2/'
studydir=[rootdir study '/']
imagetype='f' %% what kind of image are you looking for (r, rf, swrf?)

for s=1:length(subjectlist)
subject=subjectlist{s}
subjectdir=[studydir subject '/'];
cd(subjectdir)
bdirs=adir('bold/0*');
    
    fprintf('working on slice timing correction...');
    
    numslices=32; %change this to get actual dimension from an image rather than hardcoding in
    sliceorder=[2 4 6 8 10 12 14 16 18 20 22 24 26 28 30 32 1 3 5 7 9 11 13 15 17 19 21 23 25 27 29 31]; %% temporarily made up sliceorder=[xxxxx]; % slice acquisition order, vector of integers, each integer referring to the slice number in the image file
%                (1=first), and the order of integers representing their temporal acquisition order. just leave this hardocoded since all studies using 32 generic will be same

    refslice=2; %%slice to align to (ideally middle but I might do first so that everything is aligned to start of TR)

TR=2;
TA=2-(2/numslices); %time between last slice and next volume
timing(1)= TA/(numslices-1);    %additional information for sequence timing, timing(1) = time between slices, timing(2) = time between last slices and next volume
timing(2)= TR-TA;

prefix='a'     %filename prefix for corrected image files, defaults to 'a'

for i = 1:length(bdirs)
    fprintf('timing correcting...');
    fprintf('%s ',bdirs{i});
    P=spm_select('list', bdirs{i}, ['^' imagetype '.*\.img$']);
    IPS=size(P); IPS=IPS(1)
    for p=1:IPS
    P2(p,:)=[subjectdir bdirs{i} '/' P(p,:)];
    end
    spm_slice_timing_amy(P2, sliceorder, refslice, timing, prefix)
end
end
end
