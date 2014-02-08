function roi_batch_amy_with_plots(subjects,roi_name,task_dir, loc_dir, win_secs, onsetdelay, highpass, meanwin, group_roi)
%% edited by AES, march 2013
%% now can be called by get_PSC_from_ROI (including for group ROIs)
%% plots PSC in each ROI
%% %% AES added round function to tmp_idx= line because there were decimals for onsets in certain .spm files


subjects,roi_name,task_dir, loc_dir, win_secs, onsetdelay, highpass, meanwin


temp       = strread(task_dir,'%s','delimiter','/');
task       = temp{end};

temp       = strread(loc_dir,'%s','delimiter','/');
loc        = temp{end};

temp       = strread(subjects{1},'%s','delimiter','/');
study_dir  = [temp{1} '/' temp{2} '/' temp{3} '/' temp{4}];

meanwin    = strread(meanwin,'%s','delimiter',';');
row=3;meanrow=2;
wb = waitbar(0,'Processing. . . ');
roi_xyz=0;

% New functionality lets one apply a single ROI to all subjects
if group_roi==1;
    f     =['/mindhive/saxelab/roi_library/' roi_name '.mat'] %% AES edit: takes group ROI as input from get_PSC_from_ROI
    load(f);if roi_xyz ==0;roi_xyz = xY.XYZmm';end % if VOI_*mat group ROI
    temp  = strread(f,'%s','delimiter','/');temp = temp{end};
    temp  = strread(temp,'%s','delimiter','.'); roi_name = temp{1};

%shorten name
[path, roi_name, ext]=fileparts(roi_name);
roi_name=roi_name(5:end-4)
end


for s = 1:length(subjects)
    waitbar((s/length(subjects)),wb);
    fprintf(['Now working on subject: ' subjects{s} '. . . ']);
    %try
    disp(['searching for ' fullfile(subjects{s},task_dir,'SPM.mat')]);
    if (exist((fullfile(subjects{s},task_dir,'SPM.mat')))== 0) 
        disp('task not found')
    else
        load(fullfile(subjects{s},task_dir,'SPM.mat'));
        clear temp
        if group_roi==0;
                disp(['searching for ' fullfile(subjects{s},'roi',['ROI_' roi_name '_' loc '*xyz.mat'])]);
            temp = dir(fullfile(subjects{s},'roi',['ROI_' roi_name '_' loc '*xyz.mat']));
            if length(temp)>1
                disp(['extracting average response in ' roi_name ' for subject ' subjects{s}])
                f = spm_select(1,'mat','Choose a ROI xyz file','',fullfile(subjects{s},'roi'),'xyz.mat',1);
                load(f);temp = strread(f,'%s','delimiter','_');
                newfilt = ['*' temp{end-1} '_' temp{end}];
                temp = dir(fullfile(subjects{s},'roi',['ROI_' roi_name '_' loc newfilt]));
            end
            load(fullfile(subjects{s},'roi',temp(1).name));
            
        else % if group ROI, still have to do this part
            vinv_data = inv(SPM.xY.VY(1).mat);
            ROI.XYZ   = vinv_data(1:3,:)*[roi_xyz'; ones(1,size(roi_xyz',2))];
            ROI.XYZ   = round(ROI.XYZ);  
        end
%     tempfile = dir(fullfile(subjects{s},'roi',['ROI_' roi_name '_' loc '*xyz.mat']));
%     if length(tempfile) == 0
%         fprintf([ subjects{s} ' has no ROI!']); 
%     else
%         fprintf([ subjects{s} ' has ROI!']);
%         load(fullfile(subjects{s},task_dir,'SPM.mat'));
%         clear temp
%         if group_roi==0;
%             temp = dir(fullfile(subjects{s},'roi',['ROI_' roi_name '_' loc '*xyz.mat']));
%             if length(temp)>1
%                 f = spm_select(1,'mat','Choose a ROI xyz file','',fullfile(subjects{s},'roi'),'xyz.mat',1);
%                 load(f);temp = strread(f,'%s','delimiter','_');
%                 newfilt = ['*' temp{end-1} '_' temp{end}];
%                 temp = dir(fullfile(subjects{s},'roi',['ROI_' roi_name '_' loc newfilt]));
%             end
%             load(fullfile(subjects{s},'roi',temp(1).name));
%             
%         else % if group ROI, still have to do this part
%             vinv_data = inv(SPM.xY.VY(1).mat);
%             ROI.XYZ   = vinv_data(1:3,:)*[roi_xyz'; ones(1,size(roi_xyz',2))];
%             ROI.XYZ   = round(ROI.XYZ);
%         end

        [Cond, V, RT] = jc_get_design(SPM);
        window_length = round(win_secs / RT);
        V             = SPM.xY.VY;% Timepoints x 3D Images
        %%
        if s==1 %% AES edit
        groupAvg_PSC= zeros(length(Cond),window_length+2); %% AES edit:make matrix for group average PSC
        condNames=cell(length(Cond),1); %% AES edit
        counter=zeros(length(Cond),1)'; %% AES edit
        end %% AES edit

        %start the excel cell array
        if exist('notes','var')== 0
            notes        = cell(length(subjects)+2,window_length+5);
            if group_roi==1
                if highpass==1 notes(1,1)   = {['PSC averages for ' num2str(length(subjects)) ' Subjects with an offset delay of ' num2str(onsetdelay) ', Highpass filtered']};
                else           notes(1,1)   = {['PSC averages for ' num2str(length(subjects)) ' Subjects with an offset delay of ' num2str(onsetdelay)]};
                end
            else
                if highpass==1 notes(1,1)   = {['PSC averages for ' num2str(length(subjects)) ' Subjects localized by ' loc ' with an offset delay of ' num2str(onsetdelay) ', Highpass filtered']};
                else           notes(1,1)   = {['PSC averages for ' num2str(length(subjects)) ' Subjects localized by ' loc ' with an offset delay of ' num2str(onsetdelay)]};
                end
            end
            notes(2,1:3) = {'Subject' 'Condition' 'Flag'};
            for i=-1:window_length
                notes(2,i+5) = {(i*RT)-RT};
            end

            meanamp = cell(length(subjects)+1,window_length+5);
            meanamp(1,1:3) = {'Subject' 'Window' 'Flag'};

            for i=1:length(Cond)
                meanamp(1,i+3) = Cond(i).name;
            end
        end

        % Extract values for timepoints x voxels
        for t = 1:length(V)
            ROI.Y(t,:) = spm_sample_vol(V(t), ROI.XYZ(1,:), ROI.XYZ(2,:), ROI.XYZ(3,:),1);
        end

        % This is a single timecourse for the ROI
        Y  = mean(ROI.Y,2);

        % High Pass Filtering to remove slow trends
        if highpass==1
            clear K;
            for  ss = 1:length(SPM.Sess)
                K(ss) = struct('HParam', 128,'row', SPM.Sess(ss).row,'RT', RT);
            end;
            K = spm_filter(K); Y = spm_filter(K,Y);
        end

        % warning for over 5% signal change
        Y2 = Y/mean(Y)*100;Y2 = Y2-100;
        if max(abs(Y2))>5
            notes(row,3) = {'Z > 5!'};meanamp(meanrow,2) = {'Z > 5!'};
        end

        % Create event-related responses for each condition
        % Collapsing across trials
        for c = 1:length(Cond)
            for t=1:window_length+2 % include -1 and 0 trs
                tmp_idx          = Cond(c).onidx + (t-3);
                tmp_idx          = round(tmp_idx(find(tmp_idx<=length(Y)))); %% AES added round() function because there were decimals for onsets in certain .spm files
                Cond(c).Y_avg(t) = mean(Y(tmp_idx));
            end
        end

        % Grab baseline from rest periods
        onlist = 1;
        for c=1:length(Cond)
            for t=Cond(c).onidx'
                onlist = [onlist t:t+Cond(c).num_scans+round(onsetdelay/RT)-1];
            end
        end
        offlist  = setdiff(1:length(Y),onlist);

        % make new offset list, omitting the first two trs
        for f = 1:length(SPM.nscan)
            nscan2(f) = sum(SPM.nscan(1:f));
        end
        croplist = sort([(nscan2-SPM.nscan(1)+1) (nscan2-SPM.nscan(1)+2)]);
        offlist  = setdiff(offlist,croplist);
        baseline = mean(Y(offlist));

        % Convert from raw to PSC
        PSC = zeros(length(Cond),window_length);
        for c = 1:length(Cond)
            for t = 1:window_length+2
                PSC(c,t) = 100*(Cond(c).Y_avg(t) - baseline)/baseline;         
            end
            notes(row,1:2) = [subjects{s} Cond(c).name];
            notes(row,4:window_length+5) = num2cell(PSC(c,:)); row=row+1;
        
        groupAvg_PSC(c,:)=groupAvg_PSC(c,:)+PSC(c,:); %% AES edit: add individual subject to group average
        condNames(c)=Cond(c).name; %% AES edit: doing this redundantly but whatever 
        counter(c)=counter(c)+1; %% AES edit
        end
        % calculate mean windows
        for i=1:length(meanwin)
            meanamp(meanrow,1) = {subjects{s}};
            meanamp(meanrow,2) = {['''' meanwin{i} ]};
            for c=1:length(Cond)
                temp = eval(meanwin{i});
                clear x;
                clear y;
                x=round((temp(1)/RT)+1);y=round((temp(end)/RT)+1);
                meanamp(meanrow,c+3) = {mean(PSC(c,x+2:y+2))};
            end
            meanrow=meanrow+1;
        end
        save(fullfile(subjects{s},'roi', ['ROI_' roi_name '_' task '_' date '_psc.mat']), 'Cond','Y', 'PSC','offlist','-mat');
        clear Cond Y PSC offlist baseline croplist tmp_idx nscan2 V RT ROI vinv_data
    end
     %catch
     %    warndlg(['Error with subject ' subjects{s}])
     %    notes(row,1:3) = {subjects{s} '' 'Error with this subject'};row=row+1;
     %    meanamp(meanrow,1:3) = {subjects{s} '' 'Error with this subject'};meanrow=meanrow+1;
     %end

    fprintf('Done \n');
    
end% subject loop

if exist('notes','var') 
    if ~iscell(notes)
        notes = {notes};
    end
for count=1:length(condNames) %% AES edit
    groupAvg_PSC(count,:)=groupAvg_PSC(count,:)/counter(count); %% AES edit: divide to get average PSC
end %% AES edit
    timevector=notes(2,4:window_length+5); %% AES edit grab x axis of timepoints
    timevector=cell2mat(timevector); %% AES edit
    plotPSC(study_dir, roi_name, task, subjects, timevector, groupAvg_PSC, condNames) %% AES edit
    
    %
    %
    cell2csv(fullfile(study_dir,'ROI', ['PSC_' roi_name '_' task '_' num2str(length(subjects)) '_subs.csv']),notes,',','2000');
    cell2csv(fullfile(study_dir,'ROI', ['means_' roi_name '_' task '_' num2str(length(subjects)) '_subs.csv']),meanamp,',','2000');
end

cd(fullfile(study_dir,'ROI'));
waitbar(1,wb,'Finished!');

end


%% AES edit: this function added to plot PSC in ROI whenever they are extracted.
function plotPSC(study_dir, ROI, result, subjects, timevector,condmatrix, condnames)
%% e.g. plotPSC([-4 -2 0 2 4 6 8 10], matrix, {'fh', 'fu'}) 
%%
s=size(condmatrix);
numConds=s(1);
if numConds>8
    disp('this script can only deal with first 8 conditions') %% stop being ghetto and fix this!!
end
numTimePoints=s(2);

%%currently can only deal with 8 conditions, but could make this flexibly
%%generate maximally discriminating colors for the number of conditions
%%required
markers={
    '-or'
    '-+g'
    '-*b'
    '->c'
    '-xm'
    '-s'
    '-dk'
    '-hw'
    };
colors={
    [1 0 0];
    [0 1 0];
    [0 0 1];
    [0 1 1];
    [1 0 1];
    [1 .6 .25];
    [.1 .1 .1];
    [.25 0 .63];
    };


for i=1:numConds
vector=condmatrix(i,:);
plot(timevector, vector, markers{i}, 'Color', colors{i});
hold on
end

legend(condnames, 'location', 'northwest');
title(['PSC in ' ROI ' for ' result]);
p=gcf;
saveas(p, [fullfile(study_dir,'ROI',['plot_' ROI '_' result '_' num2str(length(subjects)) '_subs'])]);
hold off
end
