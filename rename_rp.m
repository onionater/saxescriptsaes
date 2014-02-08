function rename_rp(study, subjectlist)

numSubj=length(subjectlist)
for s=1:numSubj
   subject=subjectlist{s}
   subjdir=['/mindhive/saxelab2/' study '/' subject '/bold/'];
   cd(subjdir)
   bolddirs=dir;
   bolddirs=bolddirs(3:end)
   numbolds=length(bolddirs)
   for b=1:numbolds
      bolddir=[subjdir bolddirs(b).name '/']
      bolddiroutput=['/mindhive/saxelab2/EIB/' subject '/bold/' bolddirs(b).name '/']
      copyfile([bolddir 'rp*'],bolddiroutput)
   end
   
end

end