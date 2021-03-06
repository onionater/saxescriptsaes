cd(directory)

 numM=length(subjectsave.masks);
    masksize=0;
    for nm=1:numM
       snm=size(subjectsave.masks{nm}.mat); % check size of this mask
       masksize(nm)=sum(subjectsave.masks{nm}.mat(:));
    end

if uselotsofspace
p=imagesc([subjectsave.regressors{1}.mat' subjectsave.regressors{2}.mat' subjectsave.regressors{3}.mat']);
    plotptbselectors=gcf;
    saveas(plotptbselectors, [mvpadir mask_description '_' printregressor '_' xvalselector '_regressors_noavg.fig']);
    clear gcf
    close(gcf)
    
    p=imagesc([subjectsave.regressors{4}.mat']);
    plotavgselectors=gcf;
    saveas(plotavgselectors, [mvpadir mask_description '_' printregressor '_' xvalselector '_regressors_avg.fig']);
    clear gcf
    close(gcf)
    
    notaveraged=[];
    averaged=[];
    for x=1:length(subjectsave.selectors)
       name=subjectsave.selectors{x}.name;
       if isempty(strfind(name, 'avg'))
           notaveraged=[notaveraged (subjectsave.selectors{x}.mat/max(subjectsave.selectors{x}.mat))'];
       else 
           averaged=[averaged (subjectsave.selectors{x}.mat/max(subjectsave.selectors{x}.mat))'];
       end
    end
    
    p=imagesc(notaveraged);
    noavgselectors=gcf;
    saveas(noavgselectors, [mvpadir mask_description '_' printregressor '_' xvalselector 'selectors_noavg.fig']);
    clear gcf
    close(gcf)
    
    p=imagesc(averaged);
    avgselectors=gcf;
    saveas(avgselectors, [mvpadir mask_description '_' printregressor '_' xvalselector 'selectors_avg.fig']);
    clear gcf
    close(gcf)
    
    disp(xvalselector)
  
    p=bar(masksize);
   xlabel(['mask: 1=main, rest= masks from anovas on xval folds'])
    ylabel(['# voxels in anova mask (' num2str(fsoutputthreshold) ' thresholded)'])
    voxelsInMasks=gcf;
    saveas(voxelsInMasks, [mvpadir mask_description '_' printregressor '_' xvalselector 'voxelsINmasks.fig']);
    clear gcf
    close(gcf)
end

    save([mvpadir mask_description '_' printregressor '_' xvalselector 'voxelsINmasks.mat'], 'masksize'); 
    

    %turns out you'll use a lot of space even if you just save the full
    %subjstructure for every mask. copy and delete out masks from structure
    %that is saved to dir and just save it once
    subjectsave2dir=subjectsave;
    subjectsave2dir.masks=[]; % to replace a given mask m into a structure later on just set subjectsave2dir.masks=savemasks{m}


    %% save the important stuff no matter what
    save([mvpadir mask_description '_' printregressor '_' xvalselector '_classification.mat'], 'results');
    if m==minmask %update: only save this once since it's the same for every mask
    save([mvpadir printregressor '_' xvalselector '_subjstructure.mat'], 'subjectsave2dir');
    onemaskdone=1;
    end