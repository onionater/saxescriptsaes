function add_contrast_to_behav_amy(study, task, subjects, contrastName, runs, values)

%% aes 12/12/2012

%%forgot to put that contrast in your original code? fix it...

rootdir=['/mindhive/saxelab2/' study '/behavioural/']; %% don't forget to SAXify this
cd(rootdir);
s=size(subjects);
numSubj=s(2);
for subj=1:numSubj
    for i=1:length(runs)
    load([subjects{subj} '.' task '.' num2str(runs(i)) '.mat']);
      n=size(con_info);
      origNumContrasts=n(2);   
      con_info(origNumContrasts+1).name=contrastName;  
      con_info(origNumContrasts+1).vals=values;
      save([subjects{subj} '.' task '.' num2str(runs(i)) '.mat'],'-append','con_info');
    
    end
end
end
