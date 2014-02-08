function mvpaData2csv(fn,fo)
% written by AES to facilitate access to results of correlation based mvpa analyses
% EIB specific but should generalize easily
%% takes mvpa .mat files and makes them into excel-able .csv files
%e.g. mvpaData2xlsx('MVPA_data_DMPFC_tomloc.mat', 'DMPFC_tomloc.csv')

%%% inputs
datatypes={'Rdata'; 'Rrawdata'; 'Zdata'; 'Zrawdata'; 'nvoxels'};
IDstring='SAX_EIB';

%%
mat=load(fn,datatypes{:}); 

    f=fopen(fo,'w');
    fprintf(f, '%s %s %s %s %s %s %s %s %s %s %s %s', 'subjID', 'numVoxels','datatype', 'discrimination', 'within1', 'within2', 'between1', 'between2', 'withinAvg', 'betweenAvg');
    fprintf(f,'\n');
    
subjVoxels=mat.nvoxels;    
subjects=fieldnames(subjVoxels);
N=length(subjects);
voxels=[];
for count=1:N
    voxelVal=eval(['subjVoxels.' subjects{count}]);
    voxels=[voxels; voxelVal];
end

for s=1:4
    structName=datatypes{s};
    dataT=eval(['mat.' structName]);
    discriminations=fieldnames(dataT)
    numDis=length(discriminations);
for i=1:numDis
  
  data=eval(['dataT.' discriminations{i}]);
  sizedata=size(data);
  numSubj=sizedata(4);
  disc=discriminations{i}
  groupWithin=0;
  groupBetween=0;
  for n=1:numSubj
subjID=subjects{n};
subjVox=voxels(n);
      within1=data(1,1,1,n,1);
      within2=data(2,2,1,n,1);
      between1=data(1,2,1,n,1);
      between2=data(2,1,1,n,1);
      withinAvg=(within1+within2)/2;
      betweenAvg=(between1+between2)/2;
      groupWithin=groupWithin+withinAvg;
      groupBetween=groupBetween+betweenAvg;
      
      fprintf(f, '%s %i %s %s %4.8f %4.8f %4.8f %4.8f %4.8f %4.8f', subjID, subjVox, structName, disc, within1, within2, between1, between2, withinAvg, betweenAvg);
      fprintf(f,'\n');
      
      
  end
  
    groupWavg=groupWithin/n;
    groupBavg=groupBetween/n;
    fprintf(f, '%s %s %s %s %s %s %s %s %4.8f %4.8f', 'groupAverages', 'NaN', structName, disc, 'NaN', 'NaN', 'NaN', 'NaN', groupWavg, groupBavg);
      fprintf(f,'\n');
  
end 
end
fclose(f) 
end
