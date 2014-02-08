function [motion_all] = motion_vector()


%subj, average per run, total outliers


subjects = {};



motion_all = [];

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
motion = [];
for i = 1:length(bolds)
    k = adir(fullfile(root,'bold',bolds{i},'*rp*.txt'));
    temp = textread(k{1});
    motion = vertcat(motion,abs(temp));
end

participant_mean = mean(motion,1);
participant_mean = [subj participant_mean];
motion_all = vertcat(motion_all, participant_mean);
end 



end 

