function run_grid_permutations(numIter, subjectlist)
%e.g. run_grid_permutations(5, makeIDs('EIB', [1:3]))

for subj=1:length(subjectlist)
    subject=subjectlist{subj};
    for iter=1:numIter   
%command=['run_classification_EIB(''EIB'', ''EIB_main'', {''' subject '''}, ''1to8'', {''folderprefix'', ''permutation_'', ''scramble'', 1, ''suffix'', ' num2str(iter) '})']   
command=['run_classification_EIB(''EIB'', ''EIB_main'', {''' subject '''}, ''1to8'')']   
name= ['fixclass_' subject]
%name=[subject '_perm_' num2str(iter)]
gridSubmitAES(command, name)
    end

end