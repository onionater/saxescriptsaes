function [outliers, mean_outliers] = intensity_outliers()



subjects = {};

 

outliers = [];
mean_outliers = [];

for subj =1:length(subjects)
rootdir = '/mindhive/saxelab2/EIB'
 cd(fullfile(rootdir,subjects{subj},'results/EIB_main_results_normed'));

m = spm_vol('mask.img');
M = spm_read_vols(m);
M = logical(M);
load SPM.mat
blds = {};
for i = 1:length(SPM.xY.VY)
       tmp = regexp(SPM.xY.VY(i).fname,'/','split');
       blds{end+1} = tmp{7};
end
bolds = unique(blds);
cd(fullfile('/',tmp{1:6}));
int_arts = [];
nstds = 2;
for bold = 1:length(bolds)
       fprintf('working on: %s\n',bolds{bold});
       imgs = adir(fullfile(bolds{bold},'swrf*.img'));
       y = spm_vol(char(imgs));
       gb = 0;
       for im = 1:length(imgs)
               fprintf(repmat('\b',1,gb));
               gb = fprintf('%i/%i',im,length(imgs));
               Y = spm_read_vols(y(im));
               tp(im) = nanmean(reshape(Y(M),[],1,1));
       end
       fprintf('\n');
       tp = detrend(tp);
       int_arts(bold) = sum(tp>(std(tp)*nstds));
end
fprintf('Sequence    Artifacts\n');
for i = 1:length(bolds)
       fprintf('%-13s%i\n',bolds{i},int_arts(i));
       participant(i,1) = subj; 
       participant(i,2) = str2num(bolds{i});
       participant(i,3) = int_arts(i); 
end

participant_mean(1,1) = subj;
participant_mean(1,2) = mean(int_arts);
participant_mean(1,3) = sum(int_arts);

outliers = vertcat(outliers, participant);
mean_outliers = vertcat(mean_outliers, participant_mean)
end 

end
