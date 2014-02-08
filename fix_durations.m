function fix_changenames(study, task, subjects, runs, duration)

%% originally modeled durations as just movie period, but want to change it to include response window (5.75 sec/2.875 TR)

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
      spm_inputs(c).dur=duration
      end
      save([subjects{subj} '.' task '.' num2str(runs(i)) '.mat'],'-append','spm_inputs');
    end
end
end
end
