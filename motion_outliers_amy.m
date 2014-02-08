function [outliers, mean_outliers] = motion_outliers()


%subj, average per run, total outliers


subjects = {};


outliers = [];
mean_outliers = [];

for subj =1:length(subjects)
rootdir = '/mindhive/saxelab2/EIB'
 cd(fullfile(rootdir,subjects{subj},'results/EIB_main_results_normed'));


 load SPM.mat
 bolds = {};
for i = 1:length(SPM.xY.VY)
    tmp = regexp(SPM.xY.VY(i).fname,'/','split');
    bolds{end+1} = tmp{7};
end
bolds = unique(bolds);
cwd = pwd;
tmp = regexp(cwd,'/','split');
root = fullfile('/',tmp{1:5});
for i = 1:length(bolds)
    k = adir(fullfile(root,'bold',bolds{i},'*outliers_s*.mat'));
    load(k{1});
    fprintf('Run %s has %i outliers\n',bolds{i},size(R,2));

    participant(i,1) = subj; 
       participant(i,2) = str2num(bolds{i});
       participant(i,3) = size(R,2); 
end

participant_mean(1,1) = subj;
participant_mean(1,2) = mean(participant(:,3));
participant_mean(1,3) = sum(participant(:,3));

outliers = vertcat(outliers, participant);
mean_outliers = vertcat(mean_outliers, participant_mean)
end 

end
