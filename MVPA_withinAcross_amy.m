function MVPA_withinAcross_amy(rois, groupORind)
% change the names to match your experiment 
%cd to directory where .mat files are stored 

% rois = {'RTPJ','LTPJ','PC','DMPFC'};

%March 2012: edited by AES to have slightly different labeling of output
%folders (to keep summary file for group and ind rois separate)

data_all = {};
for y= 1:length(rois)

    roi = rois{y};
    load(['MVPA_data_' roi '.mat']);  

    %names of contrasts 
    names = fieldnames(Zrawdata);
    for n = 1:length(names)
        
        %assumes that you are doing only a two-way comparision 
        %(ODD_x, ODD_y, EVEN_x, EVEN_y) 
        data.(rois{y}).(names{n}).within1 = squeeze(Zrawdata.(names{n})(1,1,:,:,1));
        data.(rois{y}).(names{n}).within2 = squeeze(Zrawdata.(names{n})(2,2,:,:,1));
        data.(rois{y}).(names{n}).across1 = squeeze(Zrawdata.(names{n})(1,2,:,:,1));
        data.(rois{y}).(names{n}).across2 = squeeze(Zrawdata.(names{n})(2,1,:,:,1));
        
        data.(rois{y}).(names{n}).within = nanmean([data.(rois{y}).(names{n}).within1 data.(rois{y}).(names{n}).within2],2);
        data.(rois{y}).(names{n}).across = nanmean([data.(rois{y}).(names{n}).across1 data.(rois{y}).(names{n}).across2],2);
        
    

        %one tailed t-test for within vs across 
        [a, b, c, d ]= ttest(data.(rois{y}).(names{n}).within,data.(rois{y}).(names{n}).across, .05, 'right');

        meanWithin = nanmean(data.(rois{y}).(names{n}).within);
        meanAcross = nanmean(data.(rois{y}).(names{n}).across);
        steWithin = ste(data.(rois{y}).(names{n}).within);
        steAcross = ste(data.(rois{y}).(names{n}).across);
        Difference = data.(rois{y}).(names{n}).within - data.(rois{y}).(names{n}).across;
        steDifference = ste(Difference);
        meanDifference = nanmean(Difference);
        
      %  sprintf('Region = %s',roi)
      %  sprintf('Contrast = %s',names{n})
      %  sprintf('Across = %0.2g(%1.2f)',meanAcross,steAcross)
      %  sprintf('Within = %0.2g(%1.2f)',meanWithin,steWithin)
      %  sprintf('Difference: t(%d)=%0.2g, p=%0.2g',d.df,d.tstat,b)

        within = num2cell(data.(rois{y}).(names{n}).within);
        across = num2cell(data.(rois{y}).(names{n}).across);

        IDs = fieldnames(Data_part1);
        
        data.(rois{y}).(names{n}).IDs = IDs;

        for x = 1:length(IDs)
            roinames{x,1} = roi;
            contrast{x,1} = names{n};
        end

        statsum = sprintf('Difference = %0.2g(%1.2f); t(%d)=%0.2g, p=%0.2g \n',meanDifference,steDifference,d.df,d.tstat,b);
        totalsum = sprintf('within = %0.2g(%1.2f), across = %0.2g(%1.2f), t(%d)=%0.2g, p=%0.2g \n',meanWithin,steWithin,meanAcross,steAcross,d.df,d.tstat,b);
       % eval(['mvpa_stats.' roi '.' (names{n}) '= statsum;']);
        eval(['mvpa_stats.' roi '.' (names{n}) '= totalsum;']);

 
        datatemp = horzcat(IDs, roinames, contrast, within, across);
        data_all= vertcat(data_all, datatemp);
    
    end 

clear datatemps;
clear names;
clear roinames;
clear contrast;

end 
time=clock;
save(['MVPAdata_',date,'_',num2str(time(4)),num2str(time(5)),groupORind,'.mat'],'mvpa_stats','data_all', 'data');
end
