function deleteContrasts(maxVal, resultsdir, subjects)
%e.g. deleteContrasts(38, 'EIB_main_results_normed', makeIDs('EIB', [1:13, 15:20]))
%created by AES 2/13
% if you made some mvpa contrasts and want to get rid of them, call this script
% with the maximum value of contrasts you want to keep (i.e. calling with
% 38 will delete everything higher than 38


rootdir='/mindhive/saxelab2/EIB/'
if ~iscell(subjects)
subjects={subjects};
end
numSubj=length(subjects);

for i=1:numSubj
 subjectName=subjects{i}
 cd([rootdir,subjectName,'/results/', resultsdir])
  load('SPM.mat')
  SPM.xCon=SPM.xCon(1:maxVal)
  save('SPM.mat','SPM')
 
spmTs=dir([pwd, '/spmT_*']);
cons=dir([pwd, '/con_*']);

for s=1:length(spmTs)
    fileName=spmTs(s).name;
    number=str2num(fileName(end-7:end-4));
    if number > maxVal
        delete(fileName);
    end
end

for s=1:length(cons)
    fileName=cons(s).name;
    number=str2num(fileName(end-7:end-4));
    if number > maxVal
        delete(fileName);
    end
end
 
 
end
end
    
    
