function add_folder(study, subjects, foldername)

%% rename results directories so you can rerun analyses without losing existing

rootdir=['/mindhive/saxelab2/' study '/']; 
cd(rootdir);
s=size(subjects);
numSubj=s(2);
for subj=1:numSubj
    thisSubj=subjects{subj}
    subjdir=[rootdir thisSubj '/'];
    cd(subjdir)
	mkdir(foldername);

end
end
