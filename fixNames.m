function fixNames()
rootdir='/mindhive/saxelab2/EIB/'
resultsdir='/results/EIB_main_results_normed'
for i=1:9
    
 subjectName=['SAX_EIB_0',num2str(i)]
 cd([rootdir,subjectName,resultsdir])
 load('SPM.mat')
 SPM.xCon(51).name='EVEN_fake_pos';
 SPM.xCon(52).name='EVEN_fake_neg';
 SPM.xCon(53).name='ODD_fake_pos';
 SPM.xCon(54).name='ODD_fake_neg';
 save('SPM.mat','SPM')
end
end
    
    
