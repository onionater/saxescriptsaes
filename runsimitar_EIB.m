function [structureScoreMap, volume]=runsimitar_EIB(task, measureType, subjectlist)
% e.g. runsimitar_EIB('tomloc', 'euclidean', makeIDs('EIB', [1:5, 7:13, 16:20])
%%created by AES 4/14/13
%% runs searchlight over whole brain looking for pattern that matches similarity structure specified for task

rootdir='/mindhive/saxelab2/EIB/';
mvpadir=[rootdir 'EIB_mvpa/'];
simitardir=[mvpadir 'simitar'];

numSubj=length(subjectlist)

for s=1:numSubj
    subject=subjectlist{s}
    
    datamat=['simdata_' subject '.mat'];

load(datamat)

measure   = measureType;

if strcmp(task,'EIB_main')
structure = [[1 -1 1 -1 1 1 1 -1]
             [1- 1 -1 1 -1 1 -1 1]
             [1 -1 1 -1 1 1 1 -1]
             [1- 1 -1 1 -1 1 -1 1]
             [1 -1 1 -1 1 1 1 -1]
             [1- 1 -1 1 -1 1 -1 1]
             [1 -1 1 -1 1 1 1 -1]
             [1- 1 -1 1 -1 1 -1 1]];
else if strcmp(task,'tomloc')
        structure = [[1 -1]
                     [-1 1]]; 
    end
end

[structureScoreMap] = computeSimilarityStructureMap(measure,examples,labels,examples,labels,'meta',meta,'similarityStructure',structure);



      
end
end