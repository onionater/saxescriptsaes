function runs=listbolds(task, subjects)
%% created by AES 3/26/13
%%takes as input a task and a cell array of subjects, prints cell array of
%%relevant bold directories
%% meant to be called as argument for other modeling scripts

load('/mindhive/saxelab2/EIB/EIB_subject_taskruns.mat')
numSubjects=size(s);
numSubjects=numSubjects(2);

for i=1:length(subjects)
   
    subjID= subjects{i};
    for x=1:numSubjects
        subjIDmatch=s(x).ID;
        if strcmp(subjID,subjIDmatch)
           r=eval(['s(x).' task]);
           if ~isempty(r)
           runs{i}=r;
            else
            disp(['runs found not for for task ' task ' for subject ' subjID])
           end
        end
    
    end

end
end