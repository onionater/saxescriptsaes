function autoROI_picker_AES(study,subjects,task,con,selectedroi, uconfig)
%called by AES's pickROIs_iterative()
%
% autoROI_picker(study,subjects,task,con,uconfig)
%
% Constructs a job structure, then performs ROI picking. The job structure
% may be re-loaded into the function to repeat the picking.
% 
% job will consist of:
%   config - STRUCT, configuration parameters. 
%   study - CELL, containing study name
%   task - CELL, containing task name
%   subjects - CELL, containing subject names / RFX directories
%   contrast - INT, the contrast number to pick from
%   conname - INT, the contrast name
%   rois - CELL, containing XYZmm, the ROI coordinates, for each ROI
%   roi_names - CELL of ROI names
%   roi_files - CELL of ROI files
%   roi_suffix - CHAR, full roi suffix (with _xyz, etc)
%   log_location - where to write the picker log to
%   log_fname - the log filename
%
% - the job file will be saved in <study>/autoROI_jobs/*
%
% - rois are saved under 'autoROI' with the name:
%       <ROI_name>_<task>_con<contrast number>_<suffix_>xyz.*
%
% options:
%
%   roi_radius
%       [numeric]
%       DEFAULT: 9 (mm)
%       Specifies the radius of the sphere to be drawn about the peak voxel
%       found in a given subject's contrast. 
%
%   op_order
%       [numeric]
%       DEFAULT: 1 
%       If 1, the picker will mask the image with the ROI, and then
%       threshold, and then select clusters that are greater than
%       clust_size. If 2, it will select clusters that are greater than
%       clust_size, THEN mask with the ROI. This will result in voxels
%       from outside the ROI being included in the peak.
%
%   old_style 
%       [boolean]
%       DEFAULT: 0
%       If 1, the script will pick the ROIs in the old style. If 0, then it
%       will pick ROIs in the current style.
%   
%   dilate_roi
%       [numeric]
%       DEFAULT: 1.00 
%       This must be a value between 1 and 0; if 1, it will take the ROI
%       mask as-is. If less than 1, will use the weighted ROI images (if it
%       can find them) to dilate the ROI, taking only the top
%       (dilate_roi*100)% of voxels. 
%
%   p_val_thresh
%       [numeric]
%       DEFAULT: 0.001 (alpha)
%       p_val_thresh is the p-value threshold to be used in computing the
%       minimum T-value required for a voxel to be considered
%       suprathreshold.
%
%   clust_size
%       [numeric]
%       DEFAULT: 10 (voxels)
%       Specifies the minimum size of a contiguous cluster required for a
%       suprathreshold voxel to be included in the map. 
%
%   custom_suffix
%       [char]
%       DEFAULT: ''
%       Defining a custom suffix allows the user to add a custom string to
%       all ROIs picked in the current session. This is useful if there
%       will be multiple iterations of ROI picking, perhaps from the same
%       task but using different contrasts. Or the user may add in a date
%       of ROI picking, for instance. The custom suffix is inserted before
%       the _xyz.mat but after all other parts of the filename, or at the
%       very end if the _xyz suffix is omitted (see incl_xyz_stuff)
%
%   group_analysis
%       [boolean]
%       DEFAULT: 0
%       Permits extraction of group ROIs. Automatically omits the task.
%       This option, if enabled, will alter the behavior of autoROI_picker.
%       Instead of selecting subjects, you will select the directory
%       containing the, for instance, random effects analysis (typically
%       found within RandomEffects). The autoROI directory is created in
%       the identified random effects results directory. 
%
%   render 
%       [boolean]
%       DEFAULT: 1
%       If 1, will render images of the ROIs as they are produced, as well
%       as saving them in <subject/RFX_directory>/report/*, as PDF files. 
%
%   whole_cluster
%       [boolean]
%       DEFAULT: 0
%       If 1, will select the whole cluster of the peak, as opposed to just
%       a sphere of the specified radius around it.
%
%   points
%       [numeric -- ARRAY]
%       DEFAULT: [] (empty)
%       If nonempty, will not prompt you for ROIs, but will instead pick
%       ROIs based on the mm-coordinate points specified in the Nx3 matrix
%       'points.' Note that points should really only be used for GROUP
%       analyses, but it is ultimately unimportant. 
%   
%   point_names
%       [CELL of CHAR]
%       DEFAULT: {} (empty)
%       The names to assign 'points.' If 'points' is nonempty, then
%       point_names must also be nonempty and of the same length as
%       size(points,1)
%-------------------------------------------------------------------------%
cwd             = pwd;
close all force;

%% Check if the function is to run an existing job or create a new one.
% Case input is just a .mat file containing an autoROI job struct
if nargin == 1 && ischar(study) && exist(study,'file') && ~exist(study,'dir')
    fprintf('Loading job\n');
    load(study);
% Case input is just an autoROI job struct
elseif nargin == 1 && isstruct(study)
    fprintf('Job accepted\n');
    job = study;
% Case the input is the string 'load', opens a SPM selection Dialogue to
% select the .mat file containing the job.
elseif nargin == 1 && strcmpi(study,'Load')
    load(spm_select(1,'mat','Select job'));
else
%% Constract a new job
%% Config setting
% First set all configs to default value.
    config.roi_radius       = 9;
    config.op_order         = 1;
    config.old_style        = 0;
    config.dilate_roi       = 1;
    config.p_val_thresh     = 0.001;
    config.clust_size       = 10;
    config.custom_suffix    = '';
    config.group_analysis   = 0;
    config.render           = 1;
    config.whole_cluster    = 0;
    config.points           = [];
    config.point_names      = {};
    
% Check if there's a uconfig cell array (for costume config pairs)
    if exist('uconfig','var')&&~iscell(uconfig)
        fprintf('\n\tWARNING: Your user configuration is not a cell array!\n');
    end
% Case there is uconfig, check that it obeys the config pairs structure and
% all the fields refer to available configs.
    if exist('uconfig','var')&&iscell(uconfig) % then they've included a custom configuration
        uco = fieldnames(config); % uco = user-configurable options
        for ovs = 1:2:length(uconfig)
            cov = uconfig([ovs,ovs+1]); % current configuration parameter-value pair
            dne = 0; % if 1, 'do not evaluate'
            if ~ischar(cov{1})
                warning('Extra variable %i is not a string',(ovs+1)/2);
                dne = 1;
            end
            % validation step 2: ensure that cov{1} is a variable that can be 
            % changed
            if ~dne&&~any(strcmp(cov{1},uco))
                warning('%s is not a user-configurable parameter',cov{1});
                dne = 1;
            end
            if ~dne
                config.(cov{1}) = cov{2};
            end
        end
    end
% Check that configs that expect a certain range/form of input are actually
% being set into that range/ form (validation of user input)
    if ~isempty(config.points)
        if ~size(config.points,2)==3
            error('"points" must be specified as a Nx3 array of mm-coordinate points\n');
        end
        if length(config.point_names)~=size(config.points,1)
            error('You must have a single name for each point you specify!');
        end
        fprintf('You have defined seed points, setting min cluster size to 0.\n');
        config.clust_size = 0;
    end
% Write the configs into the job struct
    job.config = config;
    
%% Choose Study
    if (~nargin||(exist('study','var')&&isempty(study)))
        % no study has been specified
        study = inputdlg('What is the study name?');
        if isempty(study)
            study = '*';
        end
        pos_stu = adir(fullfile('/mindhive','saxelab*',['*' study{:} '*']));
        if ~iscell(pos_stu)
            pos_stu = adir(fullfile('/*','saxelab*',study{:}));
            if ~iscell(pos_stu)
                error('Could not locate study.\n');
            end
        end
        if length(pos_stu) > 1
            % the user must disambiguate which study they wish
            study = pos_stu(listdlg('ListString',pos_stu,'SelectionMode','single','PromptString','Multiple studies found, please choose one','ListSize',[500,300]));
            if isempty(study)
                error('No study chosen.\n');
            end
        else
            study = pos_stu;
        end
    else
        if ~iscell(study)
            study = {study};
        end
        pos_stu = adir(fullfile('/mindhive','saxelab*',['*' study{:} '*']));
        if ~iscell(pos_stu)
            pos_stu = adir(fullfile('/*','saxelab*',study{:}));
            if ~iscell(pos_stu)
                error('Could not locate study.\n');
            end
        end
        if length(pos_stu) > 1
            % the user must disambiguate which study they wish
            study = pos_stu(listdlg('ListString',pos_stu,'SelectionMode','single','PromptString','Multiple studies found, please choose one','ListSize',[500,300]));
            if isempty(study)
                error('No study chosen.\n');
            end
        else
            study = pos_stu;
        end
    end
    job.study = study;
    
 %% Choose subjects and task/ Group Analysis
    cd(study{1});
    if ~config.group_analysis&&((nargin > 1)&&~isempty(subjects))
        if ~iscell(subjects)
            pos_sub = adir(['*' subjects '*']);
            if ~iscell(pos_sub)
                error('Could not locate subject(s)\n');
            elseif length(pos_sub) > 1
                subjects = pos_sub(listdlg('ListString',pos_sub,'SelectionMode','multiple','PromptString','Multiple subjects found, please choose desired subjects','ListSize',[300,300]));
            else
                subjects = pos_sub;
            end
        end
    elseif config.group_analysis
        %subjects = {''};
        subjects = {spm_select(1,'dir','Specify the folder containing the Random Effects directory','',study{1})};
    else
% find subjects folder in the study to be candidates for job
        pos_sub = adir('*/results/*/SPM.mat');
        pos_sub = unique(cellfun(@(x) x{1}, regexp(pos_sub,'/','split'),'UniformOutput',false));
        if ~iscell(pos_sub)
            error('Could not locate subject(s)\n');
        elseif length(pos_sub) > 1
            subjects = pos_sub(listdlg('ListString',pos_sub,'SelectionMode','multiple','PromptString','Multiple subjects found, please choose desired subjects','ListSize',[300,300]));
        else
            subjects = pos_sub;
        end
    end
% decide on the task 
    if ~config.group_analysis&&(nargin > 2&&~isempty(task))
        task = strrep(task,'/','');
        pos_task = adir(['*/results/*' task '*']);
        pos_task = unique(cellfun(@(x) x{end},regexp(pos_task,'/','split'),'UniformOutput',0));
        if ~iscell(pos_task)
            error('Could not locate task\n');
        elseif length(pos_task) > 1
            task = pos_task(listdlg('ListString',pos_task,'SelectionMode','single','PromptString','Please select the task','ListSize',[500,300]));
        else
            task = pos_task;
        end
    elseif ~config.group_analysis
        pos_task = adir('*/results/*');
        pos_task = unique(cellfun(@(x) x{end},regexp(pos_task,'/','split'),'UniformOutput',0));
        task = pos_task(listdlg('ListString',pos_task,'SelectionMode','single','PromptString','Please select the task','ListSize',[500,300]));
    else
        task = {''};
    end
    job.task = task;
% check which of the subjects were modeled for the desired task and choose
% only those as the subjects for the job
    if ~config.group_analysis
        % check the subjects, to ensure that each one has the desired task
        missing = [];
        notmiss = [];
        for i = 1:length(subjects)
            if ~exist(fullfile(subjects{i},'results',task{1},'SPM.mat'))
                missing(end+1) = i;
            else
                notmiss(end+1) = i;
            end
        end
        if ~isempty(missing)
            fprintf('WARNING: Could not find results directories for the following subjects:\n');
            for i = missing
                fprintf('\t%s\n',subjects{i});
            end
            fprintf('They will be excluded.\n');
            subjects = subjects(notmiss);
        end
    end
    job.subjects = subjects;
%% Choose contrast    
%load the SPM.mat for the desired job (either the RFX or a subject SPM.mat) to see available contrasts 
    if ~config.group_analysis
        load(fullfile(subjects{1},'results',task{1},'SPM.mat'));
    else
        load(fullfile(subjects{1},task{1},'SPM.mat'));
    end
    if nargin < 4 || isempty(con)
        contrast = listdlg('ListString',{SPM.xCon.name},'SelectionMode','single','PromptString','Please select the contrast','ListSize',[500,300]);
    else
        if ~isnumeric(con)
            error('Contrast number must be an integer!\n');
        end
        contrast = con;
    end
    conname = SPM.xCon(contrast).name; % do we need con?
    job.contrast = contrast;
    job.conname = conname;
    
%% Choose hypotheses space ROIs / points (if using the points config) 
    if ~isempty(selectedroi)
        roif=cellstr(selectedroi) % added this option to provide specific ROI name
    else
    if isempty(config.points)
        roif = cellstr(spm_select([1 Inf],'mat','Select ROI xyz files'));
    else
        roif = {{}};
    end
    end
    rois = {};
    names = {};
    if ~isempty(roif{1})&&isempty(config.points)
        for i = 1:length(roif)
            tmp = load(roif{i});
            tmpf = fieldnames(tmp);
% some hypotheses ROI files contain the xY struct and others don't, depends
% on the ROI library/ file that is used
            if any(strcmpi('xY',tmpf))  
                tmp4 = load(roif{i},'xY');
                rois{end+1} = tmp4.xY;
                [null name null] = fileparts(roif{i});
                name = strrep(name,'_xyz','');
                names{end+1} = name;
            elseif any(strcmpi('roi_xyz',tmpf))
                tmp4 = load(roif{i},'roi_xyz');
                tmp4.xY.XYZmm = reshape(tmp4.roi_xyz,3,[]);
                rois{end+1} = tmp4.xY;
                [null name null] = fileparts(roif{i});
                name = strrep(name,'_xyz','');
                names{end+1} = name;
            else
                fprintf('Your hypothesis spaces maybe be ''old-style'', and are not able to be used\n');
                error('Unrecognized ROI type!');
            end
        end
    elseif ~isempty(config.points)
        for p = 1:size(config.points,1)
            rois{p}.XYZmm = config.points(p,:)';
            names{p} = config.point_names{p};
        end
    else
        error('No ROIs selected!');
    end
    job.rois = rois;
    job.roi_names = names;
    job.roi_files = roif;
% determine if there exists weighted rois, which are the same as ROIs,
% only with a 'wxyz' intead of 'xyz'
    if config.dilate_roi < 1
        wfiles = {};
        for rfile = 1:length(job.roi_files)
            null = regexp(job.roi_files{rfile},'_xyz.mat','split');
            wfile{rfile} = [null{1} '_wxyz.img'];
            if ~exist(wfile{rfile},'file')
                fprintf('\n\tWarning: Could not find weighted file for some of the ROIs, skipping dilation if requested\n');
                wfile = {};
                break
            end
        end
        if ~isempty(wfile)
            fprintf('Dilating ROIs\n');
            for rfile = 1:length(job.roi_files)
                wr = spm_vol(wfile{rfile});
                [WR XYZ] = spm_read_vols(wr);
                tthresh = prctile(WR(~~WR),100-(config.dilate_roi*100)); % tthresh is the minimum t-value threshold to dilate the ROI appropriately
                rois{rfile} = XYZ(:,find(WR)>=tthresh);
            end
            job.rois = rois;
            job.roi_wfiles = wfile;
        else
            fprintf('While ROI dilation was requested, the weighted ROI files could not be located and so this step is being skipped.\n');
        end
    end
%% Assign location and suffix for the resulting ROI files and create the directory where the results will be created
    if ~config.group_analysis
        roi_suffix = ['_' job.task{1} '_' sprintf('con%02i',job.contrast) '_'];
        if config.custom_suffix
            roi_suffix = [roi_suffix config.custom_suffix '_'];
        end
        roi_suffix = [roi_suffix 'xyz'];
    else
        if config.custom_suffix
            roi_suffix = ['_' config.custom_suffix '_xyz'];
        else
            roi_suffix = '_xyz';
        end
    end
    job.roi_suffix = roi_suffix;
    job.log_location = fullfile(job.study{1},'ROI');
    if ~isempty(job.roi_names)
        for i = 1:length(job.roi_names)
            job.log_fname{i} = ['autoROI_picker_log_' job.roi_names{i} '_' job.task{1} '_' num2str(length(job.subjects)) 'subjects'];
            if ~isempty(job.config.custom_suffix)
                job.log_fname{i} = [job.log_fname{i} '_' job.config.custom_suffix];
            end
            job.log_fname{i} = [job.log_fname{i} '.csv'];
            f=fopen(fullfile(job.log_location,job.log_fname{i}),'w');
            fprintf(f,'ROIs chosen for %s contrast %s at p = %G with %imm radius sphere\n',job.roi_names{i},job.conname,job.config.p_val_thresh,job.config.roi_radius);
            fprintf(f,'subject,peak x,peak y,peak z,n voxels,max T value\n');
            fclose(f);
        end
    end
    mkdir(fullfile(study{1},'autoROI_jobs'));
    save(fullfile(study{1},'autoROI_jobs',['autoROI_job_' strrep(strrep(strrep(datestr(clock),' ','_'),':','_'),'-','_') '.mat']),'job');
end

%% Process the job - Pick ROIs for each subject/ the group
cur_count = 0;
for s = 1:length(job.subjects)
    job = pick_ROIs(job.study{1},job.subjects{s},job.task{1}, ...
                       job.contrast,job.rois,job.roi_names,job.config, ...
                       job.roi_suffix,cur_count,job);
    cur_count = cur_count + length(job.rois);
end
cd(cwd);
end

%% The ROI picking procedure
function job = pick_ROIs(study,subject,task,contrast,rois,names,config,roi_suffix,cur_count,job)
%% Prepeartion activities before processing
% create the directory for the ROIs and navigate to the folder where the model results are. 
if ~config.group_analysis
    cd(study);
    cd(subject)
    mkdir('autoROI');
    targ = fullfile(pwd,'autoROI');
% end
% if ~config.group_analysis
    cd(fullfile('results',task));
else
    cd(fullfile(subject));
    mkdir('autoROI');
    targ = fullfile(pwd,'autoROI');
end
% report to command window which subject is being processed
tmp = regexp(subject,'/','split');
if ~config.group_analysis
    try
        subject = tmp{find(cellfun(@(x) ~isempty(x),tmp),1,'last')};
    catch
        subject = tmp{1}; % I'm not sure how this will behave in a group analysis; this will have to be examined.
    end
else
    subject = strrep(subject,[study '/'],'');
    subject = strrep(subject,study,'');
end
fprintf('%s...\n',subject);
% Create folder for reports and name for report file
pfilename = fullfile(study,subject,'report');mkdir(pfilename);
pfilename = fullfile(pfilename,'ROI_picking');mkdir(pfilename);
filename = fullfile(pfilename,['ROI_picking_' date]);
%% Loading images and prepare ROIs
% load up the T-image
if ~config.group_analysis
    t = spm_vol(fullfile(study,subject,'results',task,sprintf('spmT_%04i.img',contrast)));
else
    t = spm_vol(fullfile(study,subject,sprintf('spmT_%04i.img',contrast)));
end
[T XYZ] = spm_read_vols(t);

% threshold the spmT image
null = regexp(t.descrip,'[[]]','split');
t_crit = tinv(1-config.p_val_thresh,str2num(null{2})); %set the T value threshold corresponding to the p thresh chosen
T(~isfinite(T))=0; %Turns all NaNs to 0
T(T<t_crit)=0; %Turns all values under the threshold to 0
if isempty(find(T))  %check that there were any voxels above T-Threshold
    fprintf('No suprathreshold voxels!\n');
    return
end

% transform the ROIs from millimeter space into index space
MAT    = t.mat;
IMAT   = inv(MAT);
for r = 1:length(rois)
    null = rois{r};null=null.XYZmm;
    null(4,:) = 1;
    null = IMAT*null;
    null = sub2ind(t.dim,null(1,:),null(2,:),null(3,:));
    roi_inds{r} = null;
end

%% Process peaks and ROI picking according to the requested order of actions
if config.op_order == 2||config.old_style
    % eliminate clusters smaller than the critical cluster size NOW, prior
    % to masking. 
    L = bwlabeln(~~T);
    clust_IDs = unique(L(~~L));
    for cID = clust_IDs'
        if numel(find(L==cID))<config.clust_size
            L(L==cID)=0;
        end
    end
    T = T.*~~L;
    if isempty(find(T))
        fprintf('No sufficiently sized clusters!\n'); %in all the image, regardless of ROIs
        return
    end
end

% find all the peaks, in case you need them later
peaks_ind = find(imregionalmax(T));
[px py pz] = ind2sub(size(T),peaks_ind);
peaks_xyz = [px py pz];

for r = 1:length(rois)
    % begin iterating through the rois
    stop = 0;
    cur_count = cur_count+1;
    fprintf('\tROI: %s\n',names{r});
    % create an ROI mask -- unless you have points defined!
    if isempty(config.points)
        RM = zeros(t.dim);
        RM(roi_inds{r}) = 1;
        mT = T.*RM;
    else
        % mask with the cluster containing the roi point
        tL = bwlabeln(~~T);
        mT = T.*(tL==tL(roi_inds{r}));
    end
    if isempty(find(mT))
        fprintf('No suprathreshold voxels within ROI!\n');
        stop = 1;
    end
    % mT is now the ROI masked T-image
  
    % Check that the cluster size is larger than min cluster size required (if regular order) 
    if config.op_order == 1&&~config.old_style
        L = bwlabeln(~~mT);
        clust_IDs = unique(L(~~L));
        for cID = clust_IDs'
            if numel(find(L==cID))<config.clust_size
                L(L==cID)=0;
            end
        end
        mT = mT.*(~~L); % Eliminates small clusters
        if isempty(find(mT))
            fprintf('No sufficiently sized clusters within ROI!\n');
            stop = 1;
        end
    end
    if ~stop
    % find the max
    max_ind = find(mT==max(max(max(mT))));
    [mx my mz] = ind2sub(t.dim,max_ind);
    max_xyz = [mx my mz];
    if config.old_style
        % then you need to jump to the nearest peak
        % compute the distance to the true peaks
        peak_dist = (sum(bsxfun(@minus,peaks_xyz,max_xyz).^2,2).^(1/2));
        max_ind = peaks_ind(find(peak_dist == min(peak_dist)));
        [mx my mz] = ind2sub(t.dim,max_ind);
        max_xyz = [mx my mz];
    end
    clust_id = L(max_ind);
    mT = mT.*(L==clust_id);  %Eliminates all clusters except for the one with max peak
    %limit the cluster to a sphere around the peak voxel with the configed
    %radius (unless configured not too)
    stv_inds = find(mT);
    [sx sy sz] = ind2sub(t.dim,stv_inds);
    stv_xyz = [sx sy sz];
    if ~config.whole_cluster
        stv_dist = (sum(bsxfun(@minus,stv_xyz,max_xyz).^2,2).^(1/2));
        clust_inds = stv_inds(stv_dist<=(config.roi_radius/2)); % divide by two, because we're in voxel space
    else
        clust_inds = stv_inds;
    end
    %collect the picked ROI data into variables and save ROI .mat file
    peak_T = mT(max_ind);
    n_vox = size(clust_inds,1);
    peak_xyz_vox = max_xyz;
    peak_xyz_ind = max_ind;
    peak_xyz_mm = XYZ(:,max_ind)';
    [rx ry rz] = ind2sub(t.dim,clust_inds);
    roi_XYZ = [rx ry rz];
    roi_XYZmm = XYZ(:,clust_inds)';
    roi_XYZind = clust_inds;
    if config.whole_cluster
        xY.descrip = 'cluster';
        xY.str = 'cluster';
    else
        xY.descrip = 'sphere';
        xY.str = sprintf('%.1fmm sphere',config.roi_radius);
    end
    xY.xyz = peak_xyz_mm;
    xY.M = t.mat;
    xY.XYZmm = roi_XYZmm';
    save(fullfile(targ,[names{r} roi_suffix '.mat']),'xY','config','peak_T','n_vox', ...
         'peak_xyz_vox','peak_xyz_ind','peak_xyz_mm','roi_XYZ','roi_XYZmm','roi_XYZind');         
    try
    	f=fopen(fullfile(job.log_location,job.log_fname{r}),'a');
        fprintf(f,'%s,%i,%i,%i,%i,%G\n',subject,peak_xyz_mm(1),peak_xyz_mm(2),peak_xyz_mm(3),n_vox,peak_T);
        fclose(f);
    end
    % if you're supposed to render stuff, now is the time!
    if config.render
        render_images(study,subject,task,contrast,names{r},roi_suffix,filename);
    end
    rh = t;
    rh.fname = fullfile(targ,[names{r} roi_suffix '.img']);
    R = zeros(rh.dim);
    R(roi_XYZind) = 1;
    spm_write_vol(rh,R);
    end
end
% convert the .ps file, if it exists!
if exist([filename '.ps'],'file')
    ps2pdf('psfile',[filename '.ps'],'pdffile',[filename '.pdf']);
    system(sprintf('rm -rf %s',[filename '.ps']));
end
end


%% Render image function
function render_images(study,subj,task,con,roi,roi_suffix,filename)
%-------------------------------------------------------------------------%
% render_images(study,subj,task,con,roi,n2r,c2r)
% study = string, study
% subj = string, subject
% task = string, task
% con = int, contrast number from which to obtain the T
% roi = string, the ROI name
% n2r = [int, int], number of images to render simultaneously
% c2r = int, current image to render (index into the subplot)
%-------------------------------------------------------------------------%
% this script renders the ROI of interest, overlayed on the participant's
% anatomical and functional activation. The rendering is done in a subplot,
% the size of which is given by the 
%subplot(n2r(1),n2r(2),c2r);

thresh = 0.001;
func_trans = 0.75;
ROI_trans = 1;
color_allocation = [  1 200; ...
                    201 300; ...
                    301 301;]; % this is the color range allocated to the anatomical, functional, and ROI images respectively
                
root = char(adir(fullfile(study,subj)));
try
    a = spm_vol(char(adir(fullfile(root,'3danat','ws*.img'))));
catch
    a = spm_vol(char(adir('/mindhive/saxelab/scripts/template/rT1.nii')));
end
A = spm_read_vols(a); % anatomical
try
    f = spm_vol(char(adir(fullfile(root,'results',task,sprintf('spmT_%04i.img',con)))));
catch
    f = spm_vol(char(adir(fullfile(root,task,sprintf('spmT_%04i.img',con)))));
end
F = spm_read_vols(f); % functional T-map 
r = char(adir(fullfile(root,'autoROI',[roi roi_suffix '.mat'])));
r = load(r);
R = zeros(size(F));R(r.roi_XYZind) = 1;

% threshold the functional T-map
% => aquire degrees of freedom
null = regexp(f.descrip,'[[]]','split');
dof = str2num(null{2});
% => compute the critical T-statistic
t_crit = tinv(1-thresh,dof);
% => filter & threshold T-map
F(~isfinite(F))=0;
F(F<t_crit)=0;

% Determine the peak T-value
MAT    = f.mat;
IMAT   = inv(MAT);
xyz = r.xY.xyz';
xyz(4) = 1;
xyz = IMAT*xyz;xyz=xyz(1:3); % xyz is the center of the roi

% produce appropriate image slices
Ax=squeeze(A(xyz(1),:,:));Fx=squeeze(F(xyz(1),:,:));Rx=squeeze(R(xyz(1),:,:));
Ay=squeeze(A(:,xyz(2),:));Fy=squeeze(F(:,xyz(2),:));Ry=squeeze(R(:,xyz(2),:));
Az=squeeze(A(:,:,xyz(3)));Fz=squeeze(F(:,:,xyz(3)));Rz=squeeze(R(:,:,xyz(3)));
Ax=rot90(Ax);Fx=rot90(Fx);Rx=rot90(Rx);
Ay=rot90(Ay);Fy=rot90(Fy);Ry=rot90(Ry);

Ay=flipdim(Ay,2);Fy=flipdim(Fy,2);Ry=flipdim(Ry,2);
Az=flipdim(Az,1);Fz=flipdim(Fz,1);Rz=flipdim(Rz,1);
% compute alpha mapping
Axa = ones(size(Ax));Aya = ones(size(Ay));Aza = ones(size(Az));
Fxa = zeros(size(Fx));Fxa(find(Fx)) = func_trans;
Fya = zeros(size(Fy));Fya(find(Fy)) = func_trans;
Fza = zeros(size(Fz));Fza(find(Fz)) = func_trans;
Rxa = zeros(size(Rx));Rxa(find(Rx)) = ROI_trans;
Rya = zeros(size(Ry));Rya(find(Ry)) = ROI_trans;
Rza = zeros(size(Rz));Rza(find(Rz)) = ROI_trans;

% now, store the images in a cell array to make indexing easier
% imgs{a,b,c} >
%       a, image (anat, func, roi)
%       b, axis (x, y, z)
%       c, type (image, alpha map)

% oh right, and the y-images have to be flipped Left to Right while the z
% images have to be flipped up > down

imgs{1,1,1} = Ax;imgs{1,1,2} = Axa;
imgs{1,2,1} = Ay;imgs{1,2,2} = Aya;
imgs{1,3,1} = Az;imgs{1,3,2} = Aza;
imgs{2,1,1} = Fx;imgs{2,1,2} = Fxa;
imgs{2,2,1} = Fy;imgs{2,2,2} = Fya;
imgs{2,3,1} = Fz;imgs{2,3,2} = Fza;
imgs{3,1,1} = Rx;imgs{3,1,2} = Rxa;
imgs{3,2,1} = Ry;imgs{3,2,2} = Rya;
imgs{3,3,1} = Rz;imgs{3,3,2} = Rza;

% oh right, and the y-images have to be flipped Left to Right while the z
% images have to be flipped up > down


% now we need to generate a colormap; to force it to use multiple
% colormaps, we have to catenate a bunch of colormaps together
% the anatomical will be displayed in GREY
% ... guh...
% actually, this isn't that hard. Let's allocate the range 0-200 to the
% anatomical, 201-300 to the functional, and 301 to the ROI!
% so let's set all the images to their appropriate ranges
ca=color_allocation; %to make it easier to type
for axis = 1:3
    for img = 1:3
        n=reshape(imgs{img,axis,1},[],1);n=n+(-1*min(n));n=n/max(n);
        n=n*(ca(img,2)-ca(img,1));
        n=n+ca(img,1);
        imgs{img,axis,1}=round(reshape(n,size(imgs{img,axis,1})));
    end
end
CMap = [gray(length(ca(1,1):ca(1,2))); hot(length(ca(2,1):ca(2,2))); 0 0 1];
colormap(CMap);
% let's not do the subplot thing; it's useless. why not catenate the images
% together instead
%dims = a.dim;
imsz = [size(Ax,1)+size(Az,1) size(Ax,2)+size(Ay,2)];
im{1,1} = zeros(imsz);im{1,2} = zeros(imsz);
im{2,1} = zeros(imsz);im{2,2} = zeros(imsz);
im{3,1} = zeros(imsz);im{3,2} = zeros(imsz);
col_inds = [1 size(imgs{1,1,1},2); size(imgs{1,1,1},2)+1 size(imgs{1,1,1},2)+size(imgs{1,2,1},2); 1 size(imgs{1,3,1},2)];
row_inds = [1 size(imgs{1,1,1},1); 1 size(imgs{1,2,1},1); size(imgs{1,1,1},1)+1 size(imgs{1,1,1},1)+size(imgs{1,3,1},1)];
for imz = 1:size(imgs,1)
    for ax = 1:size(imgs,2)
        im{imz,1}(row_inds(ax,1):row_inds(ax,2),col_inds(ax,1):col_inds(ax,2)) = imgs{imz,ax,1};
        im{imz,2}(row_inds(ax,1):row_inds(ax,2),col_inds(ax,1):col_inds(ax,2)) = imgs{imz,ax,2};
    end
end
h(1) = image(im{1,1});
hold on
h(2) = image(im{2,1},'AlphaData',im{2,2});
h(3) = image(im{3,1},'AlphaData',im{3,2});
set(h(1),'CDataMapping','direct');
set(h(2),'CDataMapping','direct');
set(h(3),'CDataMapping','direct');
set(gca,'XTick',[]);
set(gca,'YTick',[]);

%-------------------------------------------------------------------------%
% LABELING
% let's draw lines? 
% it seems that the horizontal line is given by the z-coordinate, or rather
% its compliment
line([1 size(im{1,1},2)],[a.dim(3)-xyz(3) a.dim(3)-xyz(3)],'Color','r');
% the veritcal line is merely the y-coordinate
line([xyz(2) xyz(2)],[1 size(im{1,1},1)],'Color','r');
hold off
s2p = sprintf('%s\n%s\n%s\nSize: %i\nPeak T: %.3f\nCoords: %i %i %i\n',subj,task,roi,size(r.xY.XYZmm,2),F(xyz(1),xyz(2),xyz(3)),r.xY.xyz);
s2p=strrep(s2p,'_','\_');
text(size(Ax,2)+10,round((size(im{1,1},1))*2/3),s2p,'Color','w');
drawnow;
print(gcf,filename,'-r150','-append','-dpsc2')
end