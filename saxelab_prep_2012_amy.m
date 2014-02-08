function saxelab_prep_2012_amy(varargin)
%%% march 2013: edited by AES for various reasons
%% changes made so far:
%%% option to do slice timing correction (assumes TR=2, slices=32, and 32
%%% generic protocol)
% added 6 to each of the relevant prep_seq options
%% changes to be made:


% saxelab_prep_2012(STUDY,[{]SUBJECT[,SUBJECT2, ...}][,preprocessing integer])
% This script is the primary method of preprocessing one or more subjects
% for analysis in SPM.
%
% A) At its simplest it may be called with one argument to
% indicate 'study', for which it will make the following assumptions:
%       1) All subjects in the directory 'study' are to be processed
%       2) Subject directories are given by the identifier 'SAX'
%       3) All preprocessing steps (realign, normalize, smooth) are to be
%       performed
%       E.G. % saxelab_prep_2012('MOR4')
%
% B) second & third arguments may be given to override these assumptions:
%       - Use a string to specify a subject or different subject filter
%       E.G. % saxelab_prep_2012('MOR4','SAX_MOR4_01')
%       E.G. % saxelab_prep_2012('MOR3','KAN')
%
%       -Specify multiple subjects with a cell of strings:
%       E.G. % saxelab_prep_2012('MOR3',{'KAN_MOR3_07','KAN_MOR3_11'})
%
%       - Specify preprocessing steps with an integer:
%           (all preprocessing sequences are given by additive combinations
%            of the integers that represent each step)
%               1* = realign
%               2  = realign & normalize
%               3* = smooth
%               4  = realign & smooth
%               5  = realign, normalize, & smooth (default)
%               6  = realign, slice timing, normalize & smooth (added by
%               AES)
%       E.G. % saxelab_prep_2012('MEM',4)
% Note that the order of the third and fourth arguments is irrelevant,
%  the code is only sensitive to the class of its inputs.
%       Examples with 3 arguments:
%       E.G. % saxelab_prep_2012('MOR4',4,'SAX_MOR4_01')
%       E.G. % saxelab_prep_2012('CHAR2',{'SAX_CHAR2_03','SAX_CHAR2_04',...
%       'SAX_CHAR2_07','SAX_CHAR2_09'},4)

% ============================ Usage Notes ===========================
% There is a common tweak for preprocessing that didn't deserve
% "arguement" status:
%
%       Which functional data to look for when running preprocessing steps
%   independently.  That is, for prep_seq values of 2, 4, or 6 we need to
%   make an assumption about whether or not realignment (and normalization)
%   have been done.  The most common reasons for this will be either matlab
%   crashes after realignment is complete, or smoothing un-normalized data
%   for a within-subjects study.  Control of these assumptions is simply
%   achieved by toggling a boolean for the variables "realigned" &
%   "normalized" in the preprocess function (see lines ~210  and ~230).
%       -Default: smooth rrun* data for prep_seq=4, wrrun* for prep_seq>5
%
%

% ====================================================================
% To port this script to another workstation environment you can change the
% root directory here.
% This script expects the following directory structure:
%                       EXPERIMENT_ROOT_DIR
%                   /                        \
%           <experiment(s)>                     <analysis_tools>
%               |                                       |
%           <subject(s)>                              <spm>
%               ^                                       |
% <bold> <dicom> <3danat> <results> <scout> <roi>     <spm2>
%   ^                         ^               |         |
% <001...00n>             <task(s)>          ...       ...
%
% ====================================================================
% modified on 8/18/12 to motion correct to the first volume, then apply the
% first volumes's coregistration to the anatomical to all subsequent
% volumes, across runs.
% ====================================================================
prep_seq = 5; % code for preprocessing steps
% ====================================================================

if nargin<1
    help saxelab_prep_2012;
    return;
end
%cd('/');
study = adir(sprintf('/mindhive/saxe*/%s',varargin{1}));
if ~iscell(study)
    terror('Could not locate study.');
elseif length(study) > 1
    study = adir(sprintf('-t /mindhive/saxe*/%s',varargin{1}));
    study = study{1};
    warning('Found more than one study with that name!')
    warning(sprintf('Will analyze most recent one: %s',study));
else
    study = study{1};
end
if nargin==1
    % find all subjects which match.
    search_string = fullfile(study,'SAX*');
    subjects = adir(search_string);
    if ~iscell(subjects)
        terror('Could not locate any subjects.');
    end
    fprintf('Found %i subjects to analyze.\n',length(subjects));
elseif nargin>=2
    subjects = {};
    for i=2:nargin
        switch class(varargin{i})
            case 'char'
                subjects = {fullfile(study,varargin{i})};
            case 'cell'
                for j = 1:length(varargin{i})
                    tmp = adir(fullfile(study,varargin{i}{j}));
                    if ~iscell(tmp)
                        warning(sprintf('Could not find subject %s, will skip this subject\n',varargin{i}{j}));
                    else
                        subjects{end+1} = tmp{1};
                    end
                end
            case 'double'
                prep_seq = varargin{i};
        end
    end
end
for subject = 1:length(subjects)
    tmp = regexp(subjects{subject},'/','split');
    fprintf('______________________________\n');
    fprintf('==============================\n');
    fprintf('Working on %s\n',tmp{end});
    fprintf('==============================\n');
    preprocess(subjects{subject},prep_seq);
end
end % Main Body

function preprocess(subject,prep_seq)
% conducts preprocessing
% ======================================================================= %
% Get Ready
% ======================================================================= %
cd(subject)

% attempt to restore the original anatomical
origdir = fullfile(subject,'3danat','orig');
if isdir(origdir)
else
    warning('No guarentee of unmodified anatomical--proceeding anyway...');
end 
comd = rlc();
repo_dir = fullfile(subject,'report');mkdir(repo_dir);
save(fullfile(subject,'report','prep_command'),'comd');
fwhm=8;
if prep_seq==2||prep_seq==5||prep_seq==6,fwhm=1;end %% extended to include 6 (AES) changed fwhm from 5 to 1
orig_prep_seq = prep_seq;
defaults=spm_defaults_alek;
bdirs=adir('bold/0*');
DOP = strrep(datestr(clock),' ','_'); % timestamp of preprocessing
bolds = {};
fprintf('Acquiring Data...\n');
for i = 1:length(bdirs)
    bolds{i} = adir(fullfile(bdirs{i},'f*.img'));
end
 anat = adir('3danat/s0*.img',1);
spm('FnUIsetup','Preprocessing')
% ======================================================================= %
% Realign
% ======================================================================= %
if any(prep_seq==[1 2 4 5 6]) %% added 6 (AES)
fprintf('==============================\n');
fprintf('REALIGN\n');
fprintf('Note: Images are realigned in-place.\n');
realign_flags = struct('quality',defaults.realign.estimate.quality,'fwhm',1,'rtm',0); % changed fwhm from 5 to 1 realign, do not reslice
fprintf('Realigning functionals...\n');
for i = 1:length(bolds)
    fprintf('%s ',bdirs{i});
    spm_realign({char(bolds{i})},realign_flags);
    spm_print(fullfile(repo_dir,sprintf('motion_run_%s_%s',bdirs{i}(end-2:end),DOP)));
end
fprintf('\n');
end
% ======================================================================= %
% Coregister Func -> Anat
% ======================================================================= %
fprintf('==============================\n');
fprintf('FUNC -> ANAT COREG\n');
job.ref = {[anat{1} ',1']};
for i = 1:length(bolds)
    fprintf('%s ',bdirs{i});
    job.source = {[bolds{i}{1} ',1']};
    job.other = bolds{i}(2:end);
    job.eoptions = defaults.coreg.estimate;
    job.roptions = defaults.coreg.write;
    job.roptions.prefix = '';
    spm_run_coreg_estimate(job);
    spm_print(fullfile(repo_dir,sprintf('func2anat_coregistration_run_%s_%s',bdirs{i}(end-2:end),DOP)));
end
% ======================================================================= %
% Coregister Anat -> Template
% ======================================================================= %
fprintf('==============================\n');
fprintf('FUNC, ANAT -> TEMPLATE COREG\n');
job.ref = {'/software/spm8/templates/T1.nii,1'};
job.source = {[anat{:} ',1']};
job.other = [bolds{:}];
job.eoptions = defaults.coreg.estimate;
job.roptions = defaults.coreg.write;
% will not be using spm_run_coreg_estimate, since we *don't* want to
% reslice to the source....
spm_run_coreg_estimate(job);
spm_print(fullfile(repo_dir,sprintf('anat2temp_coregistration_%s',DOP)));
% ======================================================================= %
% Reslice Func
% ======================================================================= %
fprintf('==============================\n');
fprintf('RESLICING FUNCTS\n');
reslice_flags.interp = defaults.realign.write.interp;
reslice_flags.wrap = defaults.realign.write.wrap;
reslice_flags.mask = defaults.realign.write.mask;
reslice_flags.which = 2;
reslice_flags.mean = 1;
reslice_flags.prefix = 'r';
spm_reslice([bolds{:}],reslice_flags); % IMAGE WRITTEN


%for i = 1:length(bolds)
%    fprintf('%s ',bdirs{i});
%    spm_reslice([bolds{i}],reslice_flags); % IMAGE WRITTEN
%end
fprintf('\n');
% ======================================================================= %
% Slice timing correction (added by AES...)
% ======================================================================= %
if prep_seq==6
    
%rootdir='/mindhive/saxelab2/'
%studydir=[rootdir study '/']
imagetype='rf' %% what kind of image are you looking for (r, rf, swrf?)

%subjectdir=pwd
%cd(subjectdir)
bdirs=adir('bold/0*');
    
    fprintf('working on slice timing correction...');
    
    numslices=32; %change this to get actual dimension from an image rather than hardcoding in
    sliceorder=[2 4 6 8 10 12 14 16 18 20 22 24 26 28 30 32 1 3 5 7 9 11 13 15 17 19 21 23 25 27 29 31]; %% temporarily made up sliceorder=[xxxxx]; % slice acquisition order, vector of integers, each integer referring to the slice number in the image file
%                (1=first), and the order of integers representing their temporal acquisition order. just leave this hardocoded since all studies using 32 generic will be same

    refslice=2; %%slice to align to (ideally middle but I might do first so that everything is aligned to start of TR)

TR=2;
TA=2-(2/numslices); %time between last slice and next volume
timing(1)= TA/(numslices-1);    %additional information for sequence timing, timing(1) = time between slices, timing(2) = time between last slices and next volume
timing(2)= TR-TA;

prefix='a'     %filename prefix for corrected image files, defaults to 'a'

for i = 1:length(bdirs)
    fprintf('timing correcting...');
    fprintf('%s ',bdirs{i});
    P=spm_select('list', bdirs{i}, ['^' imagetype '.*\.img$']);
    IPS=size(P); IPS=IPS(1)
    for p=1:IPS
    P2(p,:)=[bdirs{i} '/' P(p,:)];
    end
    spm_slice_timing_amy(P2, sliceorder, refslice, timing, prefix)
end

    
end
% end of slice timing correction (AES)

% ======================================================================= %
% Normalize Anat
% ======================================================================= %
if any(prep_seq==[2 5 6])
fprintf('==============================\n');
fprintf('NORMALIZING & SEGMENTING ANAT\n');
results = spm_preproc(anat{1});
fprintf('Saving normalization parameters...\n');
spm_prep2sn(results);
p = spm_prep2sn(results);
fprintf('==============================\n');
fprintf('Normalizing anatomicals...\n');
spm_preproc_write(p);
% write out transformed anatomical
spm_write_sn(anat{1},p,defaults.normalise.write);
end
% ======================================================================= %
% Normalize Func
% ======================================================================= %
if any(prep_seq==[2 5 6])
fprintf('==============================\n');
fprintf('NORMALIZING FUNC\n');
bolds = {};
fprintf('Acquiring Data...\n');
for i = 1:length(bdirs)
    if any(prep_seq==[6]) %% AES added option to take arf.imgs)
    bolds{i} = adir(fullfile(bdirs{i},'arf*.img'));
    else
    bolds{i} = adir(fullfile(bdirs{i},'rf*.img'));
    end
end
for i = 1:length(bdirs)
    fprintf('Bold directory %s\n',bdirs{i}(end-2:end));
    bhdr = spm_vol(char(bolds{i}));
    spm_write_sn(bhdr,p,defaults.normalise.write);
end
end
% ======================================================================= %
% Smooth Func
% ======================================================================= %
if any(prep_seq==[3 4 5 6])
% begin smoothing!
fprintf('Acquiring data...\n');
if prep_seq==3
    func_images = adir('bold/*/rf*.img');
elseif prep_seq==4
    func_images = adir('bold/*/rf*.img');
elseif prep_seq==5
    func_images = adir('bold/*/wrf*.img');
elseif prep_seq==6
    func_images = adir('bold/*/warf*.img');
end
func_images = char(func_images);
fprintf('==============================\n');
fprintf('Smoothing functionals...\n');
for j = 1:size(func_images, 1)
    curr = deblank(func_images(j,:));
    [path,name,ext] = fileparts(curr);
    smoothed_name = fullfile(path,['s' name ext]);
    spm_smooth(curr,smoothed_name,fwhm);
    spm_progress_bar('Set',j);
end
if any(prep_seq==5) %% AES replaced line: prep_seq == 5
    sinfo=regexp(subject,'/','split');
    saxelab_checkreg(sinfo{end-1},sinfo{end});
    spm_print(fullfile(repo_dir,sprintf('registration_check_%s',DOP)));
    % AES added to look for swarf*
else if any(prep_seq==6)
        sinfo=regexp(subject,'/','split');
    saxelab_checkreg(sinfo{end-1},sinfo{end}, 'swarf');
    spm_print(fullfile(repo_dir,sprintf('registration_check_%s',DOP)));
    end
    % AES added above
end
end
% ======================================================================= %
% Cleanup
% ======================================================================= %
fprintf('==============================\n');
fprintf('Deleting unneeded images...\n');
if any(prep_seq==[2 3 4 5]) %aes can add 6
    system('rm -rf bold/*/rf*');
end
%% ^leaving rf if prep_seq==6 so that I could then run just smoothing and normalization to those if I want non-slice time corrected data
%% ^put 6 back in this step to delete rf (might want to have that be default since there isn't normalize/smooth only option) 
 if any(prep_seq==[6])
     system('rm -rf bold/*/arf*');
 end
if any(prep_seq==[5])
    system('rm -rf bold/*/wrf*');
end
if any(prep_seq==[6])
    system('rm -rf bold/*/warf*');
end
%^commented out so that I can go back and smooth with larger kernel later
%(currently not smoothing)
tz = regexp(subject,'/','split');
% ======================================================================= %
% Generate Mask
% ======================================================================= %
fprintf('==============================\n');
fprintf('Generating mask...\n');
fprintf('Attempting to remove old mask, if any\n');
system(sprintf('rm -rf %s',fullfile(subject,'mask')));
if any(prep_seq==[2 5 6])
    saxelab_genmask(tz{end-1},tz{end});
else
    saxelab_genmask(tz{end-1},tz{end},0);
end
fprintf('==============================\n');
fprintf('Done.\n');
fprintf('==============================\n');
end

function [result, char_results] = adir(search_str,full_file)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% this is an adaptation of MATLAB's 'dir' function ('advanced dir')
% that wraps the ls command and parses its output.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% search_str is the search string, i.e., a the pattern to match.
% full_file is a boolean, either 1 or 0 (or omitted); if 1, the
% function will return the full file + location.
%
%		e.g.,
%
%				results = adir('*/*SAX*')
%				results = adir('*SAX*',1)
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% results is a cell array of file/directory names. char_results is that
% same set of names but expressed as a char array.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
command = ['ls -dm ' search_str];
[status,result] = system(command);
% exclude any letters that are *not* in the proper range, i.e., return
% characters, etc
if status~=0
    result = -1;
    char_results = -1;
    return
end
rnge = [double(' ')-1 double('~')+1];
result = result(double(result) > rnge(1) & double(result) < rnge(2));
% split by comma-spaces
result = regexp(result,', *','split');
if exist('full_file','var') && full_file
    % add in the full working directory
    result = cellfun(@(x) {fullfile(pwd,x)},result);
end
char_results = char(result);
end

function terror(text)
error(sprintf('\n==========================================\n%s\n==========================================\n',text));
end