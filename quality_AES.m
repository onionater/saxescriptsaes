
function files=quality_AES(study, task, subjectlist)
%% created by aes march 2013
%% shitshow but should generalize
%%e.g. quality_AES('EIB','tomloc', makeIDs('EIB', [1:13, 15:20]));
%% this script provides several measures to assess quality of a run
%Calculates the voxel-wise mean, SD, and mean/SD (ie: temporal SNR) &
%prints image/ps file
%Calculates global mean, SD, and tSNR and prints to txt file
%Calculates global mean timecourse and correlates with conditions
%correlates condition regressors with 6 movement parameters and with
%artifact regressors

%%notes:
%% 1) correlations are currently based on condition regressors convolved with HDR (from SPM mat taken from random results folder)
%% 2) translation and rotation vectors in task_rot and task_trans correlations are average of absolute value of each of the three dimensions (which is not the same as the actual nuisance regressor in the task model)
%% 3) correlation with artifact regressor is based on sum of all artifact regressors. should build in something to calculate total number of artifacts per condition
%% 4) global_task correlation is based on simple correlation between convolved regressors and the global mean. should instead use regression and get betas for each condition if want to do stats on global signal confounds


addpath('/mindhive/saxelab2/EIB')
resultsString=[task '*results*']; %% this specifies the directory to look for SPM.mat file in (shouldn't matter which you pick since cond regressors will be the same)
runs=listbolds(task, subjectlist); %% get task-relevant bold dirs for each subject
numSubj=length(subjectlist); %% how many subjects?
rrange=[-.5 .6]; %% how big should the yaxis range for r values be?

for count=1:length(subjectlist)
subjectlabels(count)=str2double(subjectlist{count}(end-1:end));
end


for subject=1:numSubj 
 

subjectID=subjectlist{subject};
boldlist=runs{subject}; % get list of bolds for this task for this subject
if ~isempty(boldlist)
numruns=length(boldlist);
rootdir=['/mindhive/saxelab2/' study '/',subjectID,'/'];
cd([rootdir 'results'])
resultfolder=dir(resultsString);
cd ..
load([rootdir 'results/' resultfolder(1).name '/SPM.mat']) %load SPM.mat containing condition regressors for this task

for r=1:numruns %% go through each bold directory

bold=boldlist(r);
bolddir=[rootdir,'bold/0' num2str(bold)]
cd(bolddir)

time = SPM.Sess(r).row; %relevant timepoints from SPM.mat
condColumns = SPM.Sess(r).col(1:length(SPM.Sess(r).U)); % extracts only effects of interest (no covariates) from design matrix
numConds=size(condColumns); numConds=numConds(2);
conditions=SPM.xX.X(time,condColumns); %% convolved condition regressors



%% this section calculates task-correlated motion (must have already run art)
matfile=adir('art_*movement*.mat');
load(matfile{1})
runtrans=R(:,end-5:end-3);
runrot=R(:,end-2:end);
avgtrans=mean(abs(runtrans),2);
avgrot=mean(abs(runrot),2);
runarts=R(:,1:end-6);
runarts=sum(runarts,2);

   for c=1:numConds
       length(conditions(:,c));
       correltrans=corrcoef(avgtrans, conditions(:,c));
       task_avgtrans_corr(r,c)=correltrans(1,2);
       correlrot=corrcoef(avgrot, conditions(:,c));
       task_avgrot_corr(r,c)=correlrot(1,2);
       correlrunarts=corrcoef(runarts, conditions(:,c));
       task_runarts_corr(r,c)=correlrunarts(1,2);
   end
  



%% this section calculates voxel-wise mean, SD, and SNR, global mean/SD/SNR, and also global-task correlations

p=spm_select('list', bolddir, '^swrf.*\.img$'); %% no sure why you need this special filter instead of swrf*.img, but you do
ips=size(p); ips=ips(1);

brainmask=[rootdir '/3danat/skull_strip_mask.img'];
    %calc voxel-wise mean, sd, and snr for preprocessed BOLDS
    disp(['Calculating voxel-wise snr on ',subjectID,'s preprocessed data'])
    %%%get files and mask and apply mask to data
    files = spm_vol(p);
    maskfile=spm_vol(brainmask);
    data=spm_read_vols(files(1));
    maskzeros=spm_read_vols(maskfile);
    data=maskzeros.*data; %%mask data
    voxelsum=zeros(size(data));
    sum_sq_deviations=zeros(size(data));
    timepoints=size(files,1);
    disp('Calculating mean signal in each voxel:')
    for i=1:timepoints
        data=spm_read_vols(files(i));
        data=maskzeros.*data;
        voxelsum=voxelsum+data;
        globaltimecourse(i)=mean(data(:)); %% for task-global correlations
    end
    
    %%task-global correlations
    for c=1:numConds
       correlglobal=corrcoef(globaltimecourse, conditions(:,c));
       task_global_corr(r,c)=correlglobal(1,2);
       condNames{c}=SPM.Sess(1).U(c).name;
    end

    
    voxelavg=voxelsum/timepoints;
    disp('Calculating standard deviation over time series in each voxel:')
    for i=1:timepoints,
        data=spm_read_vols(files(i));
        data=maskzeros.*data;
        sum_sq_deviations=sum_sq_deviations+(voxelavg-data).^2;
    end
    sd=sqrt(sum_sq_deviations/(timepoints-1));
    snr=voxelavg./sd;
    snr=maskzeros.*snr; %%mask snr (ermm, just to be sure?)
    snr(isnan(snr))=0;
    
    voxels=numel(voxelavg);
    zerocount=(voxelavg==0);
    numZeros=sum(zerocount(:));
    %%%average voxel-wise values to get a global SNR score for each subject
    globalSD=sum(sd(:))/(voxels-numZeros);
    globalAVG=sum(voxelavg(:))/(voxels-numZeros);
    globalSNR=sum(snr(:))/(voxels-numZeros);
    
%% make quality dir if it doesn't exist    
cd(rootdir)
s=size((dir('quality')));
if s(1)>0
mkdir('quality')
end
cd([rootdir,'quality'])

%s=size((dir('snr_images')));
%if s(1)>0
mkdir('snr_images')
disp('made directory snr images')
%end

    
%% print global scores for run
    fid = fopen([task '_tSNRoutput.txt'],'a');
    fprintf(fid,'%s \n', ['global stats for run ' r]);
    fprintf(fid,'%s %d  \n', 'global mean (over time series):', globalAVG);
    fprintf(fid,'%s %d  \n', 'global SD (deviations over time):', globalSD);
    fprintf(fid,'%s %d  \n', 'global SNR (temporal):', globalSNR);
    fclose(fid);
    
    %Clean up snr varaible
    snr(isnan(snr))=0;
    snr(snr>5000)=0; %eliminates the absurdly high values that can occur outside the brain
  
%%% make single run images
    %output files to the quality dir
    avg_output=files(1);
    sd_output=files(1);
    snr_output=files(1);
    cd('snr_images')
    avg_output.fname    = [task,'_average_run',num2str(r),'.img'];
    sd_output.fname    = [task,'_sd_run',num2str(r),'.img'];
    snr_output.fname    = [task,'_snr_run',num2str(r),'.img'];
    
    
    %the following method keeps the scaling factor set to 1
    avg_output=spm_create_vol(avg_output);
    sd_output=spm_create_vol(sd_output);
    snr_output=spm_create_vol(snr_output);
    disp('Writing out volumes:')
    for i=1:avg_output.dim(3);
        avg_output=spm_write_plane(avg_output,voxelavg(:,:,i),i);
        sdoutput=spm_write_plane(sd_output,sd(:,:,i),i);
        snr_output=spm_write_plane(snr_output,snr(:,:,i),i); 
        fprintf ('.')
    end
      cd ..


end

%% for each subject, go through each run and make .ps with slices from snr for each run
%% make run-averaged snr images for each subject
%% in group_quality dir make .ps showing run-averaged snr slices for each subject
for r=1:numruns

%##########################################################
currentplot = 1;
        %load files
        cd('snr_images')
        files_avg=spm_vol([task,'_average_run',num2str(r),'.img']);
        files_sd=spm_vol([task,'_sd_run',num2str(r),'.img']);
        files_snr=spm_vol([task,'_snr_run',num2str(r),'.img']);
        data_avg=spm_read_vols(files_avg);
        data_sd=spm_read_vols(files_sd);
        data_snr=spm_read_vols(files_snr);
        cd ..
        %% this is for making run-averaged snr images
        if r==1
        sumOfRuns_data_avg=data_avg;
        sumOfRuns_data_sd=data_sd;
        sumOfRuns_data_snr=data_snr;
        else
        sumOfRuns_data_avg=data_avg+sumOfRuns_data_avg;
        sumOfRuns_data_sd=data_sd+sumOfRuns_data_sd;
        sumOfRuns_data_snr=data_snr+sumOfRuns_data_snr;
        end
        %%
        
        %make and print run-specific slices
        slice1_avg=squeeze(data_avg(:,:,20));
        slice1_sd=squeeze(data_sd(:,:,20));
        axial1_snr=squeeze(data_snr(:,:,20));
        sagittal_snr=squeeze(data_snr(26,:,:));
        coronal_snr=squeeze(data_snr(:,32,:));
        if currentplot == 1
            %Do Figure
            width=8.5;
            height=11;
            % Get the screen size in inches
            set(0,'units','inches')
            scrsz=get(0,'screensize');
            % Calculate the position of the figure
            position=[scrsz(3)/2-width/2 scrsz(4)/2-height/2 width height];
            figure(1), clf;
            h=figure(1);
            set(h,'units','inches')
            % Place the figure
            set(h,'position',position)
            % Do not allow Matlab to resize the figure while printing
            set(h,'paperpositionmode','auto')
            % Set screen and figure units back to pixels
            set(0,'units','pixel')
            set(h,'units','pixel')
            %Set colors
            set(gcf,'color',[1 1 1])
            colormap(hot)
        end
        %Start Plotting

        %Plots
        subplot(5,1,1)
           axis off
       subplot(6,1,(2))
            set(gca,'position',[0.05,(0.75),0.2,0.2])
            imagesc(flipud(slice1_sd'),[2, 20])
            hold on
            axis equal
            axis off
            title('SD','fontweight','bold','position',[27,0.5])

       subplot(6,1,(3))
            set(gca,'position',[0.20,(0.75),0.2,0.2])
            imagesc(flipud(slice1_avg'),[10, 900])
             hold on
             axis equal
             axis off
             title('Avg','fontweight','bold','position',[27,0.5])
       subplot(6,1,(4))
       set(gca,'position',[0.38,(0.75),0.2,0.2])
            imagesc(flipud(coronal_snr'),[10,350])
            hold on
            axis equal
            axis off
            title('SnR Coronal','fontweight','bold','position',[27,0.5])
       subplot(6,1,(5))
       set(gca,'position',[0.58,(0.75),0.2,0.2])
            imagesc(flipud(sagittal_snr'),[10,350])
            hold on
            axis equal
            axis off
            title('SnR Sagittal','fontweight','bold','position',[27,0.5])
       subplot(6,1,6)
             set(gca,'position',[0.76,(0.75),0.2,0.2])
 imagesc(flipud(axial1_snr'),[10,350])
            hold on
            axis equal
            axis off
            title('SnR Axial','fontweight','bold','position',[27,0.5])
       
       %Title
       ttl = ['Run ',num2str(r)];
       tax = axes('Position',[0.01,.8,1,1]);
       tmp= text(0,0,ttl);
       set(tax,'xlim',[0,1],'ylim',[0,1])
       set(tmp,'FontSize',10,'HorizontalAlignment','left','FontWeight','bold')
       axis off
       %Plot checks

               %Print, close and return
               prnstr = ['print -dpsc2 -painters -append ',[subjectID,'_' task '_snr.ps']];
               eval(prnstr);
               disp('SNR output printed to file')
               close (1);           


clear correltrans
clear correlrot
clear correlrunarts
clear correlglobal

task_avgtrans_corr(isnan(task_avgtrans_corr))=0;
task_avgrot_corr(isnan(task_avgrot_corr))=0;
task_runarts_corr(isnan(task_runarts_corr))=0;
task_global_corr(isnan(task_global_corr))=0;

for currcondition=1:numConds
    seriesavg_task_avgtrans_corr(1,currcondition)=mean(task_avgtrans_corr(:,currcondition));
    seriesavg_task_avgrot_corr(1,currcondition)=mean(task_avgrot_corr(:,currcondition));
    seriesavg_task_runarts_corr(1,currcondition)=mean(task_runarts_corr(:,currcondition));
    seriesavg_task_global_corr(1,currcondition)=mean(task_global_corr(:,currcondition));
end


end

%% run-average data
seriesavg_AVG=sumOfRuns_data_avg/numruns;
seriesavg_SD=sumOfRuns_data_sd/numruns;
seriesavg_SNR=sumOfRuns_data_snr/numruns;

%% write these run-averaged volumes as images for individual subject
pwd
  %output files to the quality dir
    printseries_avg_output=files_avg(1);
    printseries_sd_output=files_sd(1);
    printseries_snr_output=files_snr(1);
    printseries_avg_output.fname    = [task,'_average_seriesavg.img'];
    printseries_sd_output.fname    = [task,'_sd_seriesavg.img'];
    printseries_snr_output.fname    = [task,'_snr_seriesavg.img'];
    
    
    %the following method keeps the scaling factor set to 1
    printseries_avg_output=spm_create_vol(printseries_avg_output);
    printseries_sd_output=spm_create_vol(printseries_sd_output);
    printseries_snr_output=spm_create_vol(printseries_snr_output);
    disp('Writing out volumes:')
    for i=1:avg_output.dim(3);
        printseries_avg_output=spm_write_plane(printseries_avg_output,seriesavg_AVG(:,:,i),i);
        printseries_sd_output=spm_write_plane(printseries_sd_output,seriesavg_SD(:,:,i),i);
        printseries_snr_output=spm_write_plane(printseries_snr_output,seriesavg_SNR(:,:,i),i); 
        fprintf ('.')
    end

%%make slices and print to subject .ps
        series_slice1_avg=squeeze(seriesavg_AVG(:,:,20));
        series_slice1_sd=squeeze(seriesavg_SD(:,:,20));
        series_axial1_snr=squeeze(seriesavg_SNR(:,:,20));
        series_sagittal_snr=squeeze(seriesavg_SNR(26,:,:));
        series_coronal_snr=squeeze(seriesavg_SNR(:,32,:));
        
                if currentplot == 1
            %Do Figure
            width=8.5;
            height=11;
            % Get the screen size in inches
            set(0,'units','inches')
            scrsz=get(0,'screensize');
            % Calculate the position of the figure
            position=[scrsz(3)/2-width/2 scrsz(4)/2-height/2 width height];
            figure(1), clf;
            h=figure(1);
            set(h,'units','inches')
            % Place the figure
            set(h,'position',position)
            % Do not allow Matlab to resize the figure while printing
            set(h,'paperpositionmode','auto')
            % Set screen and figure units back to pixels
            set(0,'units','pixel')
            set(h,'units','pixel')
            %Set colors
            set(gcf,'color',[1 1 1])
            colormap(hot)
        end
        %Start Plotting

        %Plots
        subplot(5,1,1)
           axis off
       subplot(6,1,(2))
            set(gca,'position',[0.05,(0.75),0.2,0.2])
            imagesc(flipud(series_slice1_sd'),[2, 20])
            hold on
            axis equal
            axis off
            title('SD','fontweight','bold','position',[27,0.5])

       subplot(6,1,(3))
            set(gca,'position',[0.20,(0.75),0.2,0.2])
            imagesc(flipud(series_slice1_avg'),[10, 900])
             hold on
             axis equal
             axis off
             title('Avg','fontweight','bold','position',[27,0.5])
       subplot(6,1,(4))
       set(gca,'position',[0.38,(0.75),0.2,0.2])
            imagesc(flipud(series_coronal_snr'),[10,350])
            hold on
            axis equal
            axis off
            title('SnR Coronal','fontweight','bold','position',[27,0.5])
       subplot(6,1,(5))
       set(gca,'position',[0.58,(0.75),0.2,0.2])
            imagesc(flipud(series_sagittal_snr'),[10,350])
            hold on
            axis equal
            axis off
            title('SnR Sagittal','fontweight','bold','position',[27,0.5])
       subplot(6,1,6)
             set(gca,'position',[0.76,(0.75),0.2,0.2])
            imagesc(flipud(series_axial1_snr'),[10,350])
            hold on
            axis equal
            axis off
            title('SnR Axial','fontweight','bold','position',[27,0.5])
       
       %Title
       ttl = [task ' all runs'];
       tax = axes('Position',[0.01,.8,1,1]);
       tmp= text(0,0,ttl);
       set(tax,'xlim',[0,1],'ylim',[0,1])
       set(tmp,'FontSize',10,'HorizontalAlignment','left','FontWeight','bold')
       axis off
       %Plot checks

               %Print, close and return
               prnstr = ['print -dpsc2 -painters -append ',[subjectID,'_snr.ps']];
               eval(prnstr);
               disp('SNR output printed to file')
               close (1); 
        
        
        
        
%% save subject's run-wise task-motion correlations
save([task '_task_correlations.mat'], 'condNames','task_avgtrans_corr', 'task_avgrot_corr', 'task_runarts_corr', 'task_global_corr', 'seriesavg_task_avgtrans_corr', 'seriesavg_task_avgrot_corr', 'seriesavg_task_runarts_corr', 'seriesavg_task_global_corr')
       
print_task_avgtrans_corr=[task_avgtrans_corr; seriesavg_task_avgtrans_corr];
print_task_avgrot_corr=[task_avgrot_corr; seriesavg_task_avgrot_corr];
print_task_runarts_corr=[task_runarts_corr; seriesavg_task_runarts_corr];
print_task_global_corr=[task_global_corr; seriesavg_task_global_corr];

for iii=1:length(condNames)
    legendconds(iii)=condNames{iii};
end
boldlabels=[boldlist 99];
width=8*(numruns*numConds/8);
height=12;
hFig = figure(1);
set(hFig,'units','inches')
set(hFig, 'Position', [10 10 width height])
subplot(4,1,1);bar(print_task_avgtrans_corr(:,:));legend(legendconds, 'location', 'NorthEastOutside');ylim(rrange);ylabel('r value');title('task-trans corr  ');set(gca,'XTick',[])
subplot(4,1,2);bar(print_task_avgrot_corr(:,:));legend(legendconds, 'location', 'NorthEastOutside');ylim(rrange);ylabel('r value');title('task-rot corr  ');set(gca,'XTick',[])
subplot(4,1,3);bar(print_task_runarts_corr(:,:));legend(legendconds, 'location', 'NorthEastOutside');ylim(rrange);ylabel('r value');title('task-art corr  ');set(gca,'XTick',[])
subplot(4,1,4);bar(print_task_global_corr(:,:));legend(legendconds, 'location', 'NorthEastOutside');ylim(rrange);ylabel('r value');title('task-global corr  ');set(gca,'XTick',[])
set(gca,'XTick',1:numruns+1)
xlabel(['run # (run ' num2str(boldlabels(end)) ' = average across runs)'])

p=gcf;
saveas(p, [rootdir 'quality/' task '_task-motion_plots_']);
hold off
clear gcf



for c=1:numConds

%compute run-averaged task-motion correlations for the subject

             
       Group_seriesavg_task_avgtrans_corr(subject,c)=seriesavg_task_avgtrans_corr(:,c);
       Group_seriesavg_task_avgrot_corr(subject,c)=seriesavg_task_avgrot_corr(:,c);
       Group_seriesavg_task_runarts_corr(subject,c)=seriesavg_task_runarts_corr(:,c);
       Group_seriesavg_task_global_corr(subject,c)=seriesavg_task_global_corr(:,c);
end
end

subjectlabels(subject)=str2double(subjectlist{subject}(end-1:end));

end
subjectlabels=[subjectlabels 99];

cd .. 
cd ..
save(['group_quality/' task '_group_task_correlations.mat'], 'condNames', 'Group_seriesavg_task_avgtrans_corr', 'Group_seriesavg_task_avgrot_corr', 'Group_seriesavg_task_runarts_corr', 'Group_seriesavg_task_global_corr')

%print task-confound correlations to group directory

print_Group_seriesavg_task_avgtrans_corr=[Group_seriesavg_task_avgtrans_corr; mean(Group_seriesavg_task_avgtrans_corr)];
print_Group_seriesavg_task_avgrot_corr=[Group_seriesavg_task_avgrot_corr; mean(Group_seriesavg_task_avgrot_corr)];
print_Group_seriesavg_task_runarts_corr=[Group_seriesavg_task_runarts_corr; mean(Group_seriesavg_task_runarts_corr)];
print_Group_seriesavg_task_global_corr=[Group_seriesavg_task_global_corr; mean(Group_seriesavg_task_global_corr)];

width=8;
height=12;
hFig = figure(1);
set(hFig,'units','inches')
set(hFig, 'Position', [10 10 width height])
subplot(4,1,1);bar(print_Group_seriesavg_task_avgtrans_corr(:,:));legend(legendconds, 'location', 'NorthEastOutside');ylim(rrange/2);ylabel('r value');title('task-trans corr  ');set(gca,'XTick',[])
subplot(4,1,2);bar(print_Group_seriesavg_task_avgrot_corr(:,:));legend(legendconds, 'location', 'NorthEastOutside');ylim(rrange/2);ylabel('r value');title('task-rot corr  ');set(gca,'XTick',[])
subplot(4,1,3);bar(print_Group_seriesavg_task_runarts_corr(:,:));legend(legendconds, 'location', 'NorthEastOutside');ylim(rrange/2);ylabel('r value');title('task-art corr  ');set(gca,'XTick',[])
subplot(4,1,4);bar(print_Group_seriesavg_task_global_corr(:,:));legend(legendconds, 'location', 'NorthEastOutside');ylim(rrange/2);ylabel('r value');title('task-global corr  ');set(gca,'XTick',[])
set(gca,'XTick',1:numSubj+1)
set(gca,'XTickLabel',subjectlabels)
xlabel(['subject # (subject ' num2str(subjectlabels(end)) ' = average across subjects)'])
p=gcf;
saveas(p, ['group_quality/' task '_task-motion_plots_seriessummary']);
hold off
clear gcf



%% print individual subject's run-averaged snr images to group directory
if currentplot == 1
            %Do Figure
            width=8.5;
            height=11;
            % Get the screen size in inches
            set(0,'units','inches')
            scrsz=get(0,'screensize');
            % Calculate the position of the figure
            position=[scrsz(3)/2-width/2 scrsz(4)/2-height/2 width height];
            figure(1), clf;
            h=figure(1);
            set(h,'units','inches')
            % Place the figure
            set(h,'position',position)
            % Do not allow Matlab to resize the figure while printing
            set(h,'paperpositionmode','auto')
            % Set screen and figure units back to pixels
            set(0,'units','pixel')
            set(h,'units','pixel')
            %Set colors
            set(gcf,'color',[1 1 1])
            colormap(hot)
        end
        %Start Plotting

        %Plots
        subplot(5,1,1)
           axis off
       subplot(6,1,(2))
            set(gca,'position',[0.05,(0.75),0.2,0.2])
            imagesc(flipud(series_slice1_sd'),[2, 20])
            hold on
            axis equal
            axis off
            title('SD','fontweight','bold','position',[27,0.5])

       subplot(6,1,(3))
            set(gca,'position',[0.20,(0.75),0.2,0.2])
            imagesc(flipud(series_slice1_avg'),[10, 900])
             hold on
             axis equal
             axis off
             title('Avg','fontweight','bold','position',[27,0.5])
       subplot(6,1,(4))
       set(gca,'position',[0.38,(0.75),0.2,0.2])
            imagesc(flipud(series_coronal_snr'),[10,350])
            hold on
            axis equal
            axis off
            title('SnR Coronal','fontweight','bold','position',[27,0.5])
       subplot(6,1,(5))
       set(gca,'position',[0.58,(0.75),0.2,0.2])
            imagesc(flipud(series_sagittal_snr'),[10,350])
            hold on
            axis equal
            axis off
            title('SnR Sagittal','fontweight','bold','position',[27,0.5])
       subplot(6,1,6)
             set(gca,'position',[0.76,(0.75),0.2,0.2])
            imagesc(flipud(series_axial1_snr'),[10,350])
            hold on
            axis equal
            axis off
            title('SnR Axial','fontweight','bold','position',[27,0.5])
       
       %Title
       ttl = subjectID;
       tax = axes('Position',[0.01,.8,1,1]);
       tmp= text(0,0,ttl);
       set(tax,'xlim',[0,1],'ylim',[0,1])
       set(tmp,'FontSize',07,'HorizontalAlignment','left','FontWeight','bold')
       axis off
       %Plot checks

               %Print, close and return
               prnstr = ['print -dpsc2 -painters -append ',['group_quality/' task,'_snr.ps']];
               eval(prnstr);
               disp('SNR output printed to file')
               close (1); 


end