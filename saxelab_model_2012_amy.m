function saxelab_model_2012_amy(varargin)
%%% march 2013: edited by AES for various reasons
%% changes made so far:
%%% added option to include temporal and/or dispersion derivative for the HRF 
%%% adds relevant parameters to output folder name (e.g. wglobal vs.
%%% nglobal, and prefix indicating preprocessing steps)
%% changes to be made:




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% HELP
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% saxelab_model_2012(varargin)
%
% This is the primary Saxelab batch function for creating and estimating a
% model. 
% 
% --------------------------BASIC FUNCTION--------------------------
% The script requires at least four arguments: 
% 
% saxelab_model_2012('study','subject','task',[bold])
%
% STUDY   - is a string, i.e., 'BLI'
% SUBJECT - is a string, i.e., 'SAX_BLI_01'
% TASK    - is a string, i.e., 'fba' such that there exist behavioral files
%           in the study's 'behavioural' directory, with the name 
%           <subject>.<task>.run#.mat
% BOLD    - is a numberic array, specifying the bold directories. 
%
% NOTE: similar to the old script, you can iterate over subjects by
% wrapping multiple subject strings in a CELL array (i.e., with curly
% braces). If this is the case, there must be either as many BOLD inputs as
% their are subject inputs (also wrapped in curly braces), or there must be
% exactly one--and the script will assume that each subject's analysis will
% use the same bold directories. 
%
% EXAMPLE ARGUMENTS:
%   saxelab_model_2012('BLI','SAX_BLI_01','fba',[5 9 14])
%   saxelab_model_2012('BLI',{'SAX_BLI_01','SAX_BLI_02','SAX_BLI_03'},'fba',{[5 9 14],[5 12 14],[7 9 11]})
%   saxelab_model_2012('BLI',{'SAX_BLI_01','SAX_BLI_02','SAX_BLI_03'},'fba',[5 9 14])
%   saxelab_model_2012('BLI',{'SAX_BLI_01','SAX_BLI_02','SAX_BLI_03'},'fba',[5 9 14],{'normed',0,'add_art',1, 'custom_prefix', 'swarf'})
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% --------------------------ADVANCED: OPTIONS--------------------------
% The script supports a number of options which allow the analysis itself
% to be configured by the user. 
%
% These options are passed to the script as a series of parameter/value
% pairs in a cell array. It's not as hard as it sounds. The fifth argument
% to the function are where this occurs, and it consists of a series of
% such parameter/value pairs like:
%
%   {'param1',value,'param2',value,...,'paramN',value}
%
% For instance, if you're using unnormalized data and want to include
% artifact regressors, the fifth argument would look like:
%
%   {'normed',0,'add_art',1}
%
% and the entire function call would be
%
% saxelab_model_2012('BLI',{'SAX_BLI_01','SAX_BLI_02','SAX_BLI_03'},'fba',[5 9 14],{'normed',0,'add_art',1})
%
% What follows is a description of options that are currently configurable
% by the user (default value in parenthesis):
%
%   TR (2)
%       > number (seconds)
%       > the temporal resolution
%   normed (1)
%       > 1 or 0
%       > if 1, will use normalized data, if 0, will use unnormalized data
%   mask_type (1)
%       > 1 or 0
%       > if 1, will use genmask (see: saxelab_genmask)
%   filter_frequency (128)
%       > number (max seconds / cycle)
%       > highpass filter cutoff frequency
%   add_art (0)
%       > 1 or 0
%       > if 1, will include artifact regressors in the model
%   add_mot (0)
%       > 1 or 0
%       > if 1, will include motion regressors in the model
%   global_mean (1)
%       > 1 or 0
%       > if 1, will use global mean as a criterion to find outliers
%   global_threshold (3)
%       > number (standard deviations)
%       > global signal threshold used to define outliers
%   motion_threshold (2)
%       > number (mm)
%       > motion threshold used to define outliers
%   use_diff_motion (1)
%       > 1 or 0
%       > if 1, use motion difference between scans to define artifacts
%   use_diff_global (0)
%       > 1 or 0
%       > if 1, use global signal difference between scans to define
%         artifacts
%   skip_art (0)
%       > 1 or 0
%       > if 1, will skip artifact analysis (unless add_art or force_art is
%         1)
%   force_art (0)
%       > 1 or 0
%       > if 1, will always perform artifact analysis
%   clobber (0)
%       > 1 or 0
%       > if 1, will remove old results directory without prompting
%   auto_corr_correct (0)
%       > 1 or 0
%       > if 1, will use an autoregressive model to remove unexplained
%         correlations
%   global_normalize (1)
%       > 1 or 0
%       > if 1, will account for global signal
%   force_mask (0)
%       > 1 or 0
%       > if 1, will always regenerate the mask (via saxelab_genmask),
%         unless mask_type is 0.
%   rename_old_res (0)
%       > 1 or 0
%       > if 1, will rename old directories instead of overwriting them.
%   custom_name ('')
%       > string
%       > will name the results directory custom_name
%   custom_prefix ('')
%       > string
%       > custom prefix for searching for images (i.e., 'srf')
%   timing ('scans')
%       > string
%       > 'scans' or 'secs'
%       > switches between treating the timing information as TRs ('scans')
%         or seconds ('secs')
%   inc_user_regs (0)
%       > 1 or 0
%       > if 1, will not attempt to include user regressors -- which can be
%         a problem in older studies where the user regressors field is not
%         properly specified. 
%   ignore_ips (0)
%       > 1 or 0
%       > if 1, will not validate the number of IPS on a run-by-run basis.
%         In some cases (like Bubbles3) the number of images in a run seems
%         to be at odds with the distribution of IPS for run, but the
%         images seems to sum to the correct number. Thus, by enabling
%         this, you can still model the data.
%   HRF_derivatives ([0 0])
%       > added by AES to allow use of temporal derivative as means of slice timing
%       correction
%       > first= time derivative, second = dispersion derivative
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% STEP 1: Parse the initial inputs
% If there are no input arguments, display the help file
if nargin == 0
    help saxelab_model_2012
    return
end
% if there are less than 4 arguments, error out
if nargin<4
    disp('error: saxelab_model_2012 requires a minimum of 4 arguments');
    disp('''study'', ''subjIDs'', ''tasks'' & ''boldirs'' required');
    return
end
% Check that the study <study> is type CHAR

study = varargin{1};
if iscell(study)
    if length(study)>1
        error('saxelab_model_2012 does not support iterating over studies');
    else
        study = study{1};
    end
elseif ~ischar(study)
    error('STUDY must be a STRING');
end
% Check that the subject <subjs> is type CHAR or CELL
subjs = varargin{2};
if ~iscell(subjs)
    if ischar(subjs)
        subjs = {subjs};
    else
        error('SUBJS must be a STRING or CELL array of STRINGs');
    end
end
% Check that the task <task> is type CHAR
task = varargin{3};
if iscell(task)
    if length(task)>1
        error('saxelab_model_2012 does not support iterating over tasks');
    else
        task = task{1};
    end
elseif ~ischar(task)
    error('TASK must be a STRING');
end
% Check that the bold directories <bolds> are type DOUBLE or CELL
bolds = varargin{4};
if ~isnumeric(bolds)&&~iscell(bolds)
    error('BOLDS must be an array of DOUBLES or CELL array of arrays of DOUBLEs');
end
if isnumeric(bolds)
    bolds = {bolds};
end
if iscell(bolds)
    if length(bolds)==1
        bolds = repmat(bolds,1,length(subjs));
    elseif length(bolds)~=length(subjs)
        error('Number of sets of bold directories specified does not match the number of subjects');
    end
end

%% STEP 2.1: Serialize the processing of subjects
if length(subjs)>1
    for s = 1:length(subjs)
        if length(varargin)>4
            saxelab_model_2012_amy(study,subjs{s},task,bolds{s},varargin{5:end});
        else
            saxelab_model_2012_amy(study,subjs{s},task,bolds{s});
        end
    end
    return
end
tic;

%% STEP 2.2: Do some more variable processing
% Extract the variables
subj  = subjs{1};
bolds = bolds{1};
% make sure bolds is in the correct orientation, just in case
bolds = reshape(bolds,1,[]);

%% STEP 3: Initial check that everything is in order
% check that you can find the study
study_dir = adir(['/mindhive/saxelab*/' study]);
if ~iscell(study_dir)
    error('Could not locate study');
end
study_dir = study_dir{1};
% check that you can find the subject
subj_dir = adir(fullfile(study_dir,subj));
if ~iscell(subj_dir)
    error('Could not locate subject');
end
subj_dir = subj_dir{1};
% generate the results and report directory
mkdir(fullfile(subj_dir,'results'));
mkdir(fullfile(subj_dir,'report'));
% open the log
fclose all;
f=fopen(fullfile(subj_dir,'report',['saxelab_model_2012_' strrep(datestr(clock),' ','_') '.txt']),'w');
[~, user_name] = system('whoami');
fprintf(f,'Analysis by %s\n',user_name);

% check that you can find the bold directories
bdirs = {};
for i = bolds
    if ~iscell(adir(fullfile(subj_dir,'bold',sprintf('%03d',i))))
        error(sprintf('Could not locate bold directory %03d',i));
    end
    bdirs(end+1) = adir(fullfile(subj_dir,'bold',sprintf('%03d',i)));
end
% check that you can find the same number of behavioral files as bold directories
behavs = adir(fullfile(study_dir,'behavioural',[subj '.' task '.*.mat']));
if ~iscell(behavs)
    error('Could not find any behavioral files');
elseif length(behavs)~=length(bolds)
    error('Number of bold directores does not match the number of behavioral files.');
end

% re-sort the behavioral to ensure they're in the correct order (an issue
% that can crop up when you have more than 9 runs.
[a__ b__] = sort(cellfun(@(x) str2num(x{end-1}),regexp(behavs,'\.','split')));
behavs = behavs(b__);

%% STEP 4: Parse remaining inputs
% define default values
def.TR                  = 2;    % TR in seconds
def.normed              = 1;    % Assume normalization
def.mask_type           = 1;    % if 1, use genmask. if 0, use SPM default
def.filter_frequency    = 128;  % default frequency (in seconds) for highpass filter
def.add_art             = 0;    % if 1, will add artifact regressors
def.add_mot             = 0;    % if 1, will add motion regressors
def.global_mean         = 1;    % if 1, will use global mean to identify outliers
def.global_threshold    = 3;    % global signal threshold used to define outliers, in # standard devs from mean
def.motion_threshold    = 2;    % motion threshold used to define outliers, in mm
def.use_diff_motion     = 1;    % 
def.use_diff_global     = 0;    %
def.skip_art            = 0;    % if 1, will not run artifact detection
def.force_art           = 0;    % if 1, will re-perform art forcibly
def.clobber             = 0;    % if 1, will attempt to remove old results directories (if found)
def.auto_corr_correct   = 0;    % if 1, will correct for autocorrelations
def.global_normalize    = 1;    % if 1, will perform global intensity normalization by scaling
def.force_mask          = 0;    % if 1, will forcibly regenerate the mask
def.rename_old_res      = 0;    % if 1, will attempt to rename old directories
def.custom_name         = '';   % permits you to specify a custom name (a string) as a results dir
def.custom_prefix       = '';   % permits you to specify a custom previx for the images that are being looked for (i.e., 'rf')
def.timing              = 'scans'; % permits switching between treating the timing units as seconds or TRs ('scans')
def.inc_user_regs       = 0;    % if 1, will include behavior-file specified user regressors
def.ignore_ips          = 0;    % if 1, will not perform by-run IPS validation
def.HRF_derivatives     = [0 0]; % if [1 0], will include temporal derivative, if [0 1] dispersion derivative, if [0 0] will include both temporal and dispersion derivative (added by AES) 
fprintf(f,'Study: %s\n',study);
fprintf(f,'Subject: %s\n',subj);
fprintf(f,'Task: %s\n',task);
fprintf(f,'Bolds: ');fprintf(f,'%03i ',bolds);fprintf(f,'\n');
fprintf(f,'Behavs: ');fprintf(f,'%s ',behavs{:});fprintf(f,'\n');
% user_config_vars is a list of user-configurable variables from the 
% command line--i.e., these are variables which are appropriate to modify
% by passing an executable string to saxelab_model_2012

% evaluate all strings passed to elaborate them into variable values
%   > if these do not work, throw a warning, but do not error out
user_config_vars    = fieldnames(def);
if length(varargin) > 4
    othervars = varargin{5};
    for ovs = 1:2:length(othervars)
        % othervars is a series of param / value pairs
        cov = othervars([ovs,ovs+1]); %"current other varargin"
        dne = 0; % dne = 'do not evaluate'
        % validation step: ensure that cov{1} is a string
        if ~ischar(cov{1})
            warning('Extra variable %i is not a string',(ovs+1)/2);
            dne = 1;
        end
        % validation step 2: ensure that cov{1} is a variable that can be 
        % changed
        if ~dne&&~any(strcmp(cov{1},user_config_vars))
            warning('%s is not a user-configurable variable',cov{1});
            dne = 1;
        end
        if ~dne&&~isnumeric(cov{2})
            warning('%s is assigned a non-numeric value',cov{1});
        end
        if ~dne
            def.(cov{1}) = cov{2};
        end
    end
end

if ~any(strcmp({'scans','secs'},def.timing))
    warning('''%s'' is not an acceptable timing unit. Using ''scans''',def.timing);
    def.timing = 'scans';
end

% here we print more details about the configuration of the script to the
% report file...
for param = 1:length(user_config_vars)
    p_val = def.(user_config_vars{param}); % parameter value
    if isnumeric(p_val)
        if p_val==uint8(p_val)
            fprintf(f,'%s = %i\n',user_config_vars{param},p_val);
        else
            fprintf(f,'%s = %.4f\n',user_config_vars{param},p_val);
        end
    elseif ischar(p_val)
        if isempty(p_val)
            p_val = 'NONE/UNDEFINED';
            fprintf(f,'%s = %s\n',user_config_vars{param},p_val);
        else
            fprintf(f,'%s = %s\n',user_config_vars{param},p_val);
        end
    end
end

% remove possible conflicts from arguments
if def.add_art||def.force_art
    def.skip_art = 0;
end
if ~def.mask_type
    def.force_mask = 0;
end

%% Step 5: Validate remaining requirements
% validate the behavioral file data against what's in the bold directories
images = {};
if def.normed
    prefix = 'swrf';
else
    prefix = 'srf';
end
if def.custom_prefix
    prefix = def.custom_prefix;
end
for b = 1:length(behavs)
    k = load(behavs{b});
    if ~isfield(k,'ips')
        fprintf(f,'\n\nERROR: IPS not defined for behavioral %s',behavs{b});
        fclose(f);
        error('IPS not defined for behavioral %s',behavs{b});
    end
    if ~isfield(k,'spm_inputs')
        fprintf(f,'\n\nERROR: spm_inputs are not specified -- cannot proceed');
        fclose(f);
        error('spm_inputs are not specified -- cannot proceed');
    end
    comp_cons = 1;
    if ~isfield(k,'con_info')
        warning('con_info not specified -- will not attempt to generate contrasts');
        comp_cons = 0;
    end
    images{b} = adir(fullfile(bdirs{b},[prefix '*.img']));
    if ~iscell(images{b})
        if def.normed
            fprintf(f,'\n\nERROR: data from run %s appear to be unnormalized',bdirs{b});
            fclose(f);
            error(sprintf('data from run %s appear to be unnormalized',bdirs{b}));
        else
            fprintf(f,'\n\nERROR: data from run %s appear to be unnormalized',bdirs{b});
            fclose(f);
            error(sprintf('data from run %s appear to be missing',bdirs{b}));
        end
    end
    if length(images{b})~=k.ips&&~def.ignore_ips
        fprintf(f,'\n\nERROR: either too few or too many images in bold directory %s',bdirs{b});
        fclose(f);
        error(sprintf('either too few or too many images in bold directory %s',bdirs{b}));
    end
end

% Check that the MASK is present
if def.normed
    mask = adir(fullfile(subj_dir,'3danat','skull_strip_mask.img'));
else
    mask = adir(fullfile(subj_dir,'3danat','unnormalized_skull_strip_mask.img'));
end

% if it does not exist, attempt to generate it
if ~iscell(mask)||def.force_mask
    if ~iscell(mask)
        fprintf(f,'Could not locate an appropriate mask.\n');
        fprintf('Could not locate an appropriate mask.\n');
    elseif def.force_mask
        fprintf(f,'Force mask enabled, regenerating mask.\n');
        fprintf('Force mask enabled, regenerating mask.\n');
    end
    if def.mask_type
        fprintf('Will attempt to generate one using saxelab_genmask.\n');
        % check to see if the mask directory exists
        if iscell(adir(fullfile(subj_dir,'mask')))
            fprintf('Old mask directory found. Removing it now.\n');
            tmp_msk_dir = adir(fullfile(subj_dir,'mask'));
            system(sprintf('rm -rf %s',tmp_msk_dir{1}));
        end
        try 
            saxelab_genmask(study,subj);
        catch
            fprintf(f,'\n\nERROR: Mask generation has failed.');
            fclose(f);
            error('Mask generation has failed.');
        end
        if def.normed
            mask = adir(fullfile(subj_dir,'3danat','skull_strip_mask.img'));
        else
            mask = adir(fullfile(subj_dir,'3danat','unnormalized_skull_strip_mask.img'));
        end
        if ~iscell(mask)
            fprintf(f,'\n\nERROR: Masking has failed, despite not throwing an error');
            fclose(f);
            error('Masking has failed, despite not throwing an error');
        end
    else
        fprintf('Mask is to be generated by SPM system anyway.\n');
    end
end

% check for the presence of artifact files
art_files = {};
missing = 0;
for i = 1:length(bdirs)
    art_file_tmp = adir(fullfile(bdirs{i},['art_regression_outliers_and_movement_' prefix '*.mat']));
    if ~iscell(art_file_tmp)
        missing = 1;
        break
    end
    art_files(i) = art_file_tmp(1);
    if ~iscell(art_file_tmp)
        missing = 1;
        break
    end
end
if ~missing
    def.skip_art = 1; % if you can find all the artifact files, you can skip art
end

%% Step 6: Run Art
if def.force_art ||~def.skip_art
    % make an artifact directory
    mkdir(fullfile(subj_dir,'art'));
    mkdir(fullfile(subj_dir,'art',task));
    cd(fullfile(subj_dir,'art',task));
    % construct a session file
    af = fopen('art_config001.cfg','w');
    fprintf(af,'sessions: %i\n',length(bdirs));
    fprintf(af,'global_mean: %i\n',def.global_mean);
    fprintf(af,'global_threshold: %.6f\n',def.global_threshold);
    fprintf(af,'motion_threshold: %.6f\n',def.motion_threshold);
    fprintf(af,'motion_file_type: 0\n');
    fprintf(af,'motion_fname_from_image_fname: 1\n');
    fprintf(af,'use_diff_motion: %i\n',def.use_diff_motion);
    fprintf(af,'use_diff_global: %i\n',def.use_diff_global);
    fprintf(af,'use_norms: 1\n');
    if iscell(mask)
        fprintf(af,'mask_file: %s\n',mask{1});
    end
    fprintf(af,'output_dir: %s\n',fullfile(subj_dir,'art',task));
    fprintf(af,'end\n');
    for i = 1:length(bdirs)
        fprintf(af,'session %i image ',i);
        fprintf(af,'%s ',images{i}{:});
        fprintf(af,'\n');
    end
    fprintf(af,'end\n');
    fclose(af);
    % run art 
    art('sess_file',fullfile(pwd,'art_config001.cfg'));
    close all
end

% now, re-aquire all the motion / artifact / motion + artifact regressors
art_mot_files = {};
art_files = {};
mot_files = {};
for i = 1:length(bdirs)
    if def.add_art||def.add_mot
        file_tmp = adir(fullfile(bdirs{i},['art_regression_outliers_and_movement_' prefix '*.mat']));
        art_mot_files(i) = file_tmp(1);
        file_tmp = adir(fullfile(bdirs{i},['art_regression_outliers_' prefix '*.mat']));
        art_files(i) = file_tmp(1);
        try
            file_tmp = adir(fullfile(bdirs{i},'rp_f*.txt'));
            mot_files(i) = file_tmp(1);
        catch
            file_tmp = adir(fullfile(bdirs{i},'rp_rf*.txt'));
            mot_files(i) = file_tmp(1);
        end
    end
end

%% Step 7: Specify a model directory
% construct a name for the model
model_dir = [task '_' prefix]; %% AES added prefix to model dir name
if def.add_art && def.add_mot
    model_dir = [model_dir '_with_art_and_mot_reg'];
elseif def.add_art
    model_dir = [model_dir '_with_art_reg'];
elseif def.add_mot
    model_dir = [model_dir '_with_mot_reg'];
end
model_dir = [model_dir '_results'];
if def.normed
    model_dir = [model_dir '_normed'];
else
    model_dir = [model_dir '_unnormed'];
end
% AES add to output name whether data were global normlaized
if def.global_normalize
    model_dir = [model_dir '_wglobal'];
else 
    model_dir = [model_dir '_nglobal'];
end
% end of AES edit
if ~isempty(def.custom_name)
    model_dir = def.custom_name;
end
% target dir
target_dir = fullfile(subj_dir,'results',model_dir);
if exist(target_dir,'dir')
    if def.rename_old_res
        % assign the current results directory the name
        % [target_dir '_old_' #] 
        % where # is the number of old results directories + 1
        sofar = adir([target_dir '_old_*']);
        if iscell(sofar)
            count = length(sofar)+1;
        else
            count = 1;
        end
        system(sprintf('mv %s %s',target_dir,[target_dir '_old_' sprintf('%02i',count)]));
        if exist(target_dir,'dir')
            fprintf(f,'\n\nERROR: Unable to relocate old results folder, bailing out to prevent overriding the old results directory!\n');
            fclose(f);
            fprintf('Unable to relocate old results folder, bailing out to prevent overriding the old results directory!\n');
            return
        end
    elseif def.clobber
        fprintf('Clobber is enabled, removing old directory...\n');
        system(sprintf('rm -rf %s',target_dir));
    else
        removeDir = questdlg('Old results directory found! Delete?','Dir with same name found','Yes','No','No');
        if strcmpi(removeDir,'Yes');
            system(sprintf('rm -rf %s',target_dir));
        else
            return;
        end
    end
end

% create the directory
mkdir(target_dir);

%% Step 8: Specify a model
spm_get_defaults('mask.thresh',-Inf); % set the mask threshold to -Inf to prevent it from thresholdin
fmri_spec.dir = {target_dir};
fmri_spec.timing.units = def.timing;
fmri_spec.timing.RT = def.TR;
fmri_spec.timing.fmri_t = 16; % number of time bins per scan
fmri_spec.timing.fmri_t0 = 1; % reference time bin
fmri_spec.fact = struct('name', {}, 'levels', {});
fmri_spec.bases.hrf.derivs = [0 0]; %% position 1= time derivative, position 2= dispersion derivative
fmri_spec.volt = 1; % OPTIONS: 1|2 = order of convolution; 1 = no Volterra
if def.global_normalize
    fmri_spec.global = 'Scaling';
else
    fmri_spec.global = 'None';
end
if def.auto_corr_correct
    fmri_spec.cvi = 'AR(1)';
else
    fmri_spec.cvi = 'none';
end
if def.mask_type
    fmri_spec.mask = {[mask{1} ',1']};
end

% iterate over the behavioral files
for i = 1:length(behavs)
    k = load(behavs{i});
    fmri_spec.sess(i).scans = images{i};
    % iterate over the spm_inputs field
    for j = 1:length(k.spm_inputs)
        fmri_spec.sess(i).cond(j).name = k.spm_inputs(j).name;
        fmri_spec.sess(i).cond(j).onset = k.spm_inputs(j).ons;
        fmri_spec.sess(i).cond(j).duration = k.spm_inputs(j).dur;
        fmri_spec.sess(i).cond(j).tmod = 0;
        fmri_spec.sess(i).cond(j).pmod = struct('name', {}, 'param', {}, 'poly', {});
        if isfield(k.spm_inputs(j),'pmod')&~isempty(k.spm_inputs(j).pmod)
            try
                fmri_spec.sess(i).cond(j).pmod(1).name = k.spm_inputs(j).pmod.name;
                fmri_spec.sess(i).cond(j).pmod(1).param = k.spm_inputs(j).pmod.param;
                try
                    fmri_spec.sess(i).cond(j).pmod(1).poly = k.spm_inputs(j).pmod.poly;
                catch
                    fmri_spec.sess(i).cond(j).pmod(1).poly = 1;
                end
            catch
                fmri_spec.sess(i).cond(j).pmod = struct('name', {}, 'param', {}, 'poly', {});
            end
        end
    end
    % iterate over the user_regressors field
    if isfield(k,'user_regressors')&&def.inc_user_regs
        for j = 1:length(k.user_regressors)
            try
                fmri_spec.sess(i).regress(j).name = k.user_regressors(j).name;
                fmri_spec.sess(i).regress(j).val = k.user_regressors(j).val;
            catch
                fmri_spec.sess = rmfield(fmri_spec.sess,'regress');
                break
            end
        end
    end
    % decide what regressors to use for motion / artifacts
    if def.add_mot||def.add_art
        k = load(art_mot_files{i});
        artrs = k.R(:,1:end-6);
        motrs = k.R(:,end-5:end);
        if def.add_art
            if ~isfield(fmri_spec.sess(i),'regress')
                r_ind = 1;
            else
                r_ind = length(fmri_spec.sess(i).regress)+1;
            end
            for artr = 1:size(artrs,2)
                fmri_spec.sess(i).regress(r_ind).name = ['artifact_' num2str(artr) '_run_' num2str(i)];
                fmri_spec.sess(i).regress(r_ind).val = artrs(:,artr);
                r_ind = r_ind+1;
            end
        end
        mot_n = {'x_trans','y_trans','z_trans','roll','pitch','yaw'};
        if def.add_mot
            if ~isfield(fmri_spec.sess(i),'regress')
                r_ind = 1;
            else
                r_ind = length(fmri_spec.sess(i).regress)+1;
            end
            for motr = 1:size(motrs,2)
                fmri_spec.sess(i).regress(r_ind).name = [mot_n{motr} '_run_' num2str(i)];
                fmri_spec.sess(i).regress(r_ind).val = motrs(:,motr);
                r_ind = r_ind+1;
            end
        end
    end
end

%% Step 9: Create the model specification
matlabbatch{1}.spm.stats.fmri_spec = fmri_spec;
save(fullfile(target_dir,'batch.mat'),'matlabbatch');
fprintf(f,'\nModel specification has begun. \n');
try
    spm_jobman('run',matlabbatch);
catch perror
    fprintf('\n\nERROR: Model specification has failed -- consult report.\n');
    fprintf(f,'\n\nERROR: Model specification has failed.\n');
    fprintf(f,'\n\nERROR Returned:\n%s\n',perror.message);
    fclose(f);
    return
end
cd(target_dir)

%% Step 10: Estimate the model
fprintf(f,'\nModel specification complete.\n');
load SPM.mat
fprintf(f,'\nModel estimation has begun. \n');
try
    spm_spm(SPM);
catch perror
    fprintf('\n\nERROR: Model estimation has failed -- consult report.\n');
    fprintf(f,'\n\nERROR: Model estimation has failed.\n');
    fprintf(f,'\n\nERROR Returned:\n%s\n',perror.message);
    fclose(f);
    return
end
fprintf('Model estimation complete.\n');

%% Step 11: Generate the contrasts
% in contrast to previous scripts, this script will not assume that 
% contrasts are the same between runs, instead it will be done on a run-by-
% run basis. 
if ~comp_cons
    fprintf('Not computing contrasts. Finished!\n');
    fprintf('Compute time: %02ih',floor(toc/3600));
    fprintf(' %02im',floor(rem(toc,3600)/60));
    fprintf(' %.2fs\n',rem(toc,60));
    fprintf(f,'\n\nNot computing contrasts (missing con_info). Finished!\n');
    fprintf(f,'Compute time: %02ih',floor(toc/3600));
    fprintf(f,' %02im',floor(rem(toc,3600)/60));
    fprintf(f,' %.2fs\n',rem(toc,60));
    fclose(f);
    return;
end

clear k
for b = 1:length(behavs)
    k(b) = load(behavs{b},'con_info'); 
    conlen = length(k(b).con_info);
    if b==1
        pconlen = conlen;
    end
    if pconlen~=conlen
        warning('There are an irregular number of contrasts across runs!');
        fprintf('Not computing contrasts. Finished!\n');
        fprintf('Compute time: %02ih',floor(toc/3600));
        fprintf(' %02im',floor(rem(toc,3600)/60));
        fprintf(' %.2fs\n',rem(toc,60));
        fprintf(f,'\n\nNot computing contrasts (irregular number of contrasts across runs). Finished!\n');
        fprintf(f,'Compute time: %02ih',floor(toc/3600));
        fprintf(f,' %02im',floor(rem(toc,3600)/60));
        fprintf(f,' %.2fs\n',rem(toc,60));
        fclose(f);
        return
    end
    pconlen= conlen;
end

load(fullfile(target_dir,'SPM.mat'));
% now that we've validated the contrast images, begin contrast estimation
for con = 1:length(k(1).con_info)
    cols = [];
    vals = [];
    c = zeros(1,size(SPM.xX.X,2));
    for run = 1:length(k)
        pos_ind = find(k(run).con_info(con).vals > 0);
        neg_ind = find(k(run).con_info(con).vals < 0);
        cols    = [cols SPM.Sess(run).col(pos_ind) SPM.Sess(run).col(neg_ind)];
        vals    = [vals k(run).con_info(con).vals(pos_ind) k(run).con_info(con).vals(neg_ind)];
    end
    c(cols) = vals;
    %c = c - mean(c);
    if ~isfield(SPM,'xCon')||isempty(SPM.xCon)
        SPM.xCon = spm_FcUtil('Set', k(1).con_info(con).name, 'T', 'c', c',SPM.xX.xKXs);
    else
        SPM.xCon(end+1) = spm_FcUtil('Set', k(1).con_info(con).name, 'T', 'c', c',SPM.xX.xKXs);
    end
end
spm_contrasts(SPM);

fprintf('Computing contrasts complete.\n');
fprintf('Compute time: %02ih',floor(toc/3600));
fprintf(' %02im',floor(rem(toc,3600)/60));
fprintf(' %.2fs\n',rem(toc,60));
fclose all;
end