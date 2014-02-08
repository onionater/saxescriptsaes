function fix_onsets(study, task, subjects, runs)

%% originally called first 0-2 seconds TR0 (i.e. onset at 12 seconds= TR6). this changes it to align with saxelab style of labeling that as TR1 (i.e. onset of 12 seconds = TR7)

rootdir=['/mindhive/saxelab2/' study '/behavioural/']; 
cd(rootdir);
s=size(subjects);
numSubj=s(2);
for subj=1:numSubj
    disp(subjects{subj})
    for i=1:length(runs)
        disp(i)
    load([subjects{subj} '.' task '.' num2str(runs(i)) '.mat']);
      s=size(spm_inputs);
      numConds=s(2);
      for c=1:numConds
      old=spm_inputs(c).ons'  %% just to display
      spm_inputs(c).ons=spm_inputs(c).ons+1
      new=spm_inputs(c).ons' %%just to display
      end
      save([subjects{subj} '.' task '.' num2str(runs(i)) '.mat'],'-append','spm_inputs');
    end
end
end
