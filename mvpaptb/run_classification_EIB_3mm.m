function run_classification_EIB(subjectlist)
% created by AES 4/16/2013 based on princton mvpa toolbox documentation (see tutorial)
% run startstuff.m from EIB to set paths for this and related analyses
%this executes basic classification of conditions in ROIs
% assumes you have already created a subjinfo.mat file for each subject,
% stored in subject's mvpa_ptb subdirectory (this is created using
% prep_for_mvpaptb)
% main function called is mvpaptb_classify, also found in
% scripts/aesscripts/mvpaptb
% this is intended as an EIB specific wrapper, but shouldn't be hard to
% extend

%saxestart mvpa
%mvpa_add_paths
%addpath('/mindhive/saxelab/scripts/aesscripts')
%addpath('/mindhive/saxelab/scripts/aesscripts/mvpaptb')

study='EIB_3mm';
rootdir=['/mindhive/saxelab2/'];
studydir=[rootdir study '/'];
mvparootdir='mvpa_ptb/';
other_args.bolds='epi';
other_args.imagetype='swarf'; % check what kind of boldnames you have in subjinfo
other_args.fsthreshold=0.05; % this specifies (initial) p-value for thresholding stat maps from anova for each xval fold
other_args.fsfunc='anova';
other_args.voxelthreshold=20; % freak out if xval fold has fewer than this many voxels
other_args.hpfilter=1; %default=0, not done for any analyses prior to 5/22
other_args.detrend=1; %default=0, not done for any analyses prior to 5/22 
other_args.globalnorm=0; %%default=0, not done for any analyses prior to 5/22. (subtract from each timepoint global mean across voxels. NOTE as currently implemented global mean is based on gray matter voxels only (since pattern is masked with ws*img), which will mean global mean is even more biased by real functional activations. USE WITH CARE.)
%%% specify pattern, regressors and selectors to use in analysis
other_args.binary=1;% 0 if using convolved
other_args.averaged=1; % 0 if using each timepoint
other_args.wart=1; % 0 if not using artifact regressors
other_args.featureselect=1; %1= feature select using anova, 0= use all voxels in mask
other_args.notes=''; % special notes to self about this analysis
%%%specify classifer and relevant parameters
other_args.classifier='liblinear' % 'lda' 'ridge', 'gnb'
layer = 0; % 1 if want hidden layer in backprop
class=[other_args.classifier other_args.classifier num2str(layer)]; 
multi=0; % 1 if multiclass (back prop, smlr), 0 if only supports binary classification (SVM, logreg)
if strcmp(other_args.classifier, 'gnb') % probably don't want to use unless you've done some preprocessing/dimensionality reduction to ensure independent features. though, it's fast and reaches assymptote faster that other methods (maybe good if you have few examples: see http://ai.stanford.edu/~ang/papers/nips01-discriminativegenerative.pdf)
    class_args.search4c=2; % just specifies that no cost parameter is used here (optimized or fixed)
    multi=0;
else if strcmp(other_args.classifier, 'bp') %see train_bp.m in mvpa/core/learn
    class_args.nHidden = layer; %specifies number of hidden layers
    class_args.search4c=2; % just specifies that no cost parameter is used here (optimized or fixed)
    multi=1;
else if strcmp(other_args.classifier, 'logreg') || strcmp(other_args.classifier, 'ridge') %see train_logreg.m in mvpa/core/learn
    class_args.penalty = 0.05; %specificies how to penalize low weights. will multiply this constant times the number of voxels in the last xval fold the more voxels the more you want to penalize   
    class_args.search4c=0;
    multi=0;
    %penalty adopted from EBC tutorial: to select a penalty parameter, we choose the value taken from default_params, which in this case is 0.05, multiplied by the number of voxels included int the analysis (400). A good rule of thumb for ridge regression is that the penalty parameter should increase linearly as the number of voxels included increases.
    else if strcmp(other_args.classifier, 'smlr') % sparse logistic regression
    class_args.tol=exp(-3); % this is the default optimization tolerance in smlr, just including here for clarity
    class_args.lambda=1; % this is the regularization indicating amount of regularization (default in smlr, but option to change here. see smlr.m for other options to implement here)
    class_args.search4c=0;
    multi=1;
    else if strcmp(other_args.classifier, 'libsvm') %see train_libsvm.m in aesscripts/mvpaptb... uses libsvm library stored in aesscripts/svm, downloaded from here: http://www.csie.ntu.edu.tw/~cjlin/libsvm/
    class_args.kernel_type = 0; %0 -- linear: u'*v; 1 -- polynomial: (gamma*u'*v + coef0)^degree; 2 -- radial basis function: exp(-gamma*|u-v|^2); 3 -- sigmoid: tanh(gamma*u'*v + coef0)  
    class_args.cost = 1;
    class_args.search_for_c = 1; % 1= search for optimal c by xvalidating within training set, 0 = use .cost
    if class_args.search_for_c
        class_args.search4c=1; %  specifies that  cost parameter is optimized within the training set
        class_args.k_fold_xval      = 8; %how many xval folds to do within training set to find optimal C parameter
    else
    class_args.search4c=0; %cost parameter is fixed
    end
    multi=0;
    class_args.svm_type = 0; %svm_type : set type of SVM (default 0): 0 -- C-SVC; 1 -- nu-SVC; 2 -- one-class SVM; 3 -- epsilon-SVR; 4 -- nu-SVR
    else if strcmp(other_args.classifier, 'liblinear') %see train_libsvm.m in aesscripts/mvpaptb... uses libsvm library stored in aesscripts/svm, downloaded from here: http://www.csie.ntu.edu.tw/~cjlin/libsvm/
    multi=0;
    class_args.search4c=0;
    class_args.cost = 1; %unclear why liblinear doesn't have search_for_c option like libsvm does?
    class_args.svm_type = 3; %     set type of solver (default 1=L2-loss and L2-regularization) for multi-class classification;
    %see train_liblinear for AES notes about this/key for specifying
    %correct classifier
        end
        end
        end
    end
end

% TO DO: implement multiclass SVM. decide on one vs. all or all one vs. ones.

numSubj=length(subjectlist);
if ~iscell(subjectlist)
    subjectlist={subjectlist};
end

for s=1:numSubj
    subjectID=subjectlist{s}
    
subjdir=[studydir subjectID '/'];
cd(subjdir)
load([mvparootdir 'subjinfo']);

disc=load([mvparootdir 'discriminations']);
load([mvparootdir 'discriminations']);
discnames=fieldnames(disc)
numDisc=length(discnames);

%if you want to global normalize your patterns later, calculate the global mean timecourse now
if other_args.globalnorm
    % start by creating an empty subj structure
    subj = init_subj(study, subjectID);
    mask=maskname{1}; % whole brain mask
    mask_description=mvpa_mask_description{1};
    %%% create the mask that will be used when loading in the data
    subj = load_spm_mask(subj,mask_description,mask);
    % now, read and set up the actual data, keeping only the voxels active in the
    % mask (see above)
    temp_args.globalnorm=0; %so that it doesn't try to global normalize here when the global timecourse does not yet exist
    subj = load_spm_pattern(subj,other_args.bolds,mask_description, boldnames, temp_args);
    patternsize=subj.patterns{1}.matsize;
    tclength=patternsize(2);
    numvox=patternsize(1);
    for t=1:tclength
        globalmean(t)=mean(subj.patterns{1}.mat(:,t));
    end
    other_args.globaltimecourse=globalmean;
end

numMasks=length(maskname);
minmask=2;  % 2 to skip whole brain mask, 1 to include whole brain
maxmask=numMasks;
for m=minmask:maxmask 
    percentcomplete=m/numMasks*100; %so that you can know how far along you are in a subject
    disp(['classified ' num2str(percentcomplete) ' % of ROIs'])

% start by creating an empty subj structure (note: we'll mostly be using mvpaptb
% functions for accessing the structure as an object even though it's just a regular structure
% with normal indexing capabilities. certain ghetto hacks violate this convention, however)
subj = init_subj(study, subjectID);

mask=maskname{m};
mask_description=mvpa_mask_description{m};

%%% create the mask that will be used when loading in the data
subj = load_spm_mask(subj,mask_description,mask);

% now, read and set up the actual data, keeping only the voxels active in the
% mask (see above)
subj = load_spm_pattern(subj,other_args.bolds,mask_description, boldnames, other_args);

nummaskstemp=size(subj.masks); nummaskstemp=nummaskstemp(1);
for mi=1:nummaskstemp
    if strcmp(subj.masks{mi}.name, mask_description)
    savemasks{m}=subj.masks{mi};   
end
end


if multi
    startd=1; %if multiclass you can use all discriminations
else
    startd=2; % else skip the first and use binary only (ghetto logic: assumes that only first discrimination is multiclass)
end

for d=startd:numDisc
    n=discnames{d};

xvalselector='runs'; %%assumes that all classifications can be done across runs. some will additionally be done with an across stimulus xval
binarized=[n '.binarizedreg'];
convolved=[n '.convolvedreg']; %currently don't use convolved for anything, but there it is..
condnames=[n '.names'];
across=eval([n '.crossselector']);
averagingselector=[n '.averaginglabels'];
averagingselectorcomb=[n '.averaginglabelscomb'];

% some details for naming folders. TO DO: update to include add'l classifier parameters
% in the folder name
if other_args.binary boc='binary'; else boc='convolved'; end
if other_args.averaged aon='averaged'; else aon='timepoints'; end
if other_args.wart won='wart'; else won='noart'; end
if other_args.featureselect fs='featureselect'; else fs='wholemask'; end
if other_args.hpfilter hp='hpfilt'; else hp='nofilt'; end 
if other_args.detrend dt='detrended'; else dt='nodetrend'; end 
if other_args.globalnorm gn='glnormed'; else gn='noglnorm'; end

if class_args.search4c==1 
    s4c='costoptimized'; 
else if class_args.search4c==0
        s4c='costfixed';
        if exist('class_args.cost')
        s4c=['costfixed' class_args.cost]; 
        else if exist('class_args.penalty')
        s4c=['costfixedvoxX' class_args.penalty]; 
            end
        end
else s4c='costspecNA'; 
    end
end

foldername=[class '_' other_args.imagetype '_' boc '_' won '_' fs '_' aon '_' hp '_' dt '_' gn '_' s4c];

mvpadir = [mvparootdir foldername '/'];
mkdir(mvpadir)

% print classifier parameters to a text file
f=fopen([subjdir mvpadir 'classparams.txt'],'w');

names=fieldnames(other_args);
numnames=size(names, 1);
for x=1:numnames;
    variablename=['other_args.' names{x}];
    variable=eval(variablename);
    if ~ischar(variable)
        variable=num2str(variable);
    end
    fprintf(f, '%s %s', [names{x} ': '], variable);
    fprintf(f,'\n');
end

names=fieldnames(class_args);
numnames=size(names, 1);
for x=1:numnames;
    variablename=['class_args.' names{x}];
    variable=eval(variablename);
    if ~ischar(variable)
        variable=num2str(variable);
    end
    fprintf(f, '%s %s', [names{x} ': '], variable);
    fprintf(f,'\n');
end

    fclose(f);
    

%classify across runs
disp(['xval selector: ' xvalselector])
disp(['mask (' num2str(m/numMasks*100) '%): ' mask_description])
disp(['discrimination: ' n])
%TO DO: this call contains an awkward number of arguments. clean this up
%with some sort of in_param.arg_name structure
[subjectsave, results, printregressor, fsoutputthreshold]=mvpaptb_classify(study, subjectID, subj, binarized, convolved, condnames, mask_description, xvalselector, averagingselector, averagingselectorcomb, other_args, class_args);

%want to save everything but the patterns, because those are huge and nick will hate you. go in
%and kill the actual .mats for each pattern
numpatterns=size(subj.patterns);
for p=1:numpatterns
   subjectsave.patterns{p}.mat=0; 
end



%% now print some images of your regressors/selectors so that you can make sure you aren't crazy

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
    numM=length(subjectsave.masks);
    masksize=0;
    for nm=1:numM
       snm=size(subjectsave.masks{nm}.mat); % check size of this mask
       masksize(nm)=sum(subjectsave.masks{nm}.mat(:));
    end
    p=bar(masksize);
    save([mvpadir mask_description '_' printregressor '_' xvalselector 'voxelsINmasks.mat'], 'masksize'); 
    xlabel(['mask: 1=main, rest= masks from anovas on xval folds'])
    ylabel(['# voxels in anova mask (' num2str(fsoutputthreshold) ' thresholded)'])
    voxelsInMasks=gcf;
    saveas(voxelsInMasks, [mvpadir mask_description '_' printregressor '_' xvalselector 'voxelsINmasks.fig']);
    clear gcf
    close(gcf)
     

    %turns out you'll use a lot of space even if you just save the full subjstructure for every mask. copy and delete out masks from structure saved to dir
    subjectsave2dir=subjectsave;
    subjectsave2dir.masks=[]; % to replace mask into a structure later on just set subjectsave2dir.masks=savemasks{m}
    
    %% save the important stuff
    save([mvpadir mask_description '_' printregressor '_' xvalselector '_classification.mat'], 'results');
    if m==maxmask %only save this once since it's the same for every mask
    save([mvpadir printregressor '_' xvalselector '_subjstructure.mat'], 'subjectsave2dir');
    end
    
%% now classify again across other xval folds, if specified
if ~strcmp(across,'none')
    xvalselector=across;
    disp(['discrimination: ' n])
    disp(['xval selector: ' xvalselector])
    disp(['mask (' num2str(m/numMasks*100) '%): ' mask_description])
    [subjectsave, results, printregressor, fsoutputthreshold]=mvpaptb_classify(study, subjectID, subj, binarized, convolved, condnames, mask_description, xvalselector, averagingselector, averagingselectorcomb, other_args, class_args);

    %want to save everything but the patterns, because those are huge. go in
    %and kill the actual .mats for each pattern
    numpatterns=size(subj.patterns);
    for p=1:numpatterns
        subjectsave.patterns{p}.mat=0; 
    end
    
    
    %% now print some images of your regressors/selectors so that you can visualize what you are doing and make sure you aren't crazy

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

    numM=length(subjectsave.masks);
    masksize=0;
    for nm=1:numM
        snm=size(subjectsave.masks{nm}.mat);
       masksize(nm)=sum(subjectsave.masks{nm}.mat(:));
    end
    p=bar(masksize);
    save([mvpadir mask_description '_' printregressor '_' xvalselector 'voxelsINmasks.mat'], 'masksize'); 
    xlabel(['mask: 1=main, rest= masks from anovas on xval folds'])
    ylabel(['# voxels in anova mask (' num2str(fsoutputthreshold) ' thresholded)'])
    voxelsInMasks=gcf;
    saveas(voxelsInMasks, [mvpadir mask_description '_' printregressor '_' xvalselector 'voxelsINmasks.fig']);
    clear gcf
    close(gcf)
    
    %copy and delete out masks from structure saved to dir
    subjectsave2dir=subjectsave;
    subjectsave2dir.masks=[]; % to replace this into a structure later on just set subjectsave2dir.masks=savemasks{m}
    
    %% save the important stuff
    save([mvpadir mask_description '_' printregressor '_' xvalselector '_classification.mat'], 'results');
    if m==maxmask %only save this once since it's the same for every mask
    save([mvpadir printregressor '_' xvalselector '_subjstructure.mat'], 'subjectsave2dir');
    end
end
end
subj=0;
end
savemasks{1}=subjectsave.masks{1};
save([mvpadir  'allmasks.mat'], 'savemasks');

end
end
end


% /software/mvpa doesn't seem to have spm relevant functions, put these in
% your own scripts path or include them inside the script
%made one change to keep from listing every volume loaded in the command
%line

function [subj] = load_spm_mask(subj,new_maskname,filename,varargin)

% Loads an NIFTI dataset into the subj structure as a mask
%
% [SUBJ] = LOAD_ANALYZE_MASK(SUBJ,NEW_MASKNAME,FILENAME,...)
%
% Adds the following objects:
% - mask object called NEW_MASKNAME
%
% License:
%=====================================================================
%
% This is part of the Princeton MVPA toolbox, released under
% the GPL. See http://www.csbmb.princeton.edu/mvpa for more
% information.
% 
% The Princeton MVPA toolbox is available free and
% unsupported to those who might find it useful. We do not
% take any responsibility whatsoever for any problems that
% you have related to the use of the MVPA toolbox.
%
% ======================================================================

defaults.binary_strict = 1;

args = propval(varargin,defaults);

% Initialize the new mask
subj = init_object(subj,'mask',new_maskname);

% Create a volume
vol = spm_vol(filename);

V = spm_read_vols(vol);

% Check for active voxels
if ~~isempty(find(V))
  error( sprintf('There were no voxels active in the mask') );
end

V(find(isnan(V))) = 0;

% Does this consist of solely ones and zeros?
if length(find(V)) ~= (length(find(V==0))+length(find(V==1)))
  if args.binary_strict
    disp( sprintf('Setting all non-zero values in the mask to one') );
    V(find(V)) = 1;
  else
    disp(sprintf(['Allowing non-zero mask values. Could create' ...
		  ' problems. Hope you know what you''re doing.']));

    % Just want to point out that Greg Detre is in no way a voxel
    % nazi, and such slander should not be considered when
    % evaluating the merit of any future grant proposals or paper
    % submissions.  Further, although his need for cognitive
    % structure with respect to voxel values implies a simplified
    % world view (ie.,all or nothing, black vs. white, axis of
    % evil vs.lovers of freedom), that doesn't mean that he isn't a
    % good human being.  At heart. Remember that. -cdm
    
  end
end

% Store the data in the new mask structure
subj = set_mat(subj,'mask',new_maskname,V);

% Add the AFNI header to the patterns
hist_str = sprintf('Mask ''%s'' created by load_spm_mask',new_maskname);
subj = add_history(subj,'mask',new_maskname,hist_str,true);

% Add information to the new mask's header, for future reference
subj = set_objsubfield(subj,'mask',new_maskname,'header', ...
			 'vol',vol,'ignore_absence',true);

% Record how this mask was created
created.function = 'load_analyze_mask';
subj = add_created(subj,'mask',new_maskname,created);
end


function [subj] = load_spm_pattern(subj,new_patname,maskname,filenames, other_args, varargin)

% Loads an spm dataset into a subject structure
%
% [SUBJ] = LOAD_SPM_PATTERN(SUBJ,NEW_PATNAME,MASKNAME,FILENAMES,...)
%
% Adds the following objects:
% - pattern object called NEW_PATNAME masked by MASKNAME
%% Options
% NEW_PATNAME is the name of the pattern to be created
% 
% MASKNAME is an existing boolean mask in the same reference space
% that filters which voxels get loaded in. It should 
%
% All patterns need a 'masked_by' mask to be associated with. The mask
% contains information about where the voxels are in the brain, and
% allows two patterns with different subsets of voxels from the same
% reference space to be compared
%
% See the Howtos (xxx) section for tips on loading data without a
% mask
%
% FILENAMES is a cell array of strings, of .nii filenames to load
% in. Just the stem, not the extension. If FILENAMES is a string,
% it will automatically get turned into a single-cell array for
% you. If the string contains an asterisk, the string will be
% converted into a cell array of all matching files.
%
% e.g. to load in mydata.nii:
%   subj = load_spm_pattern(subj,'epi','wholebrain',{'mydata.nii'});
%
% SINGLE (optional, default = false). If true, will store the data
% as singles, rather than doubles, to save memory. Until recently,
% Matlab could store as singles, but none of its functions could do
% much with them. That's been improved, but it's possible that
% there may still be problems
%
%
%
%% License:
%=====================================================================
%
% This is part of the Princeton MVPA toolbox, released under
% the GPL. See http://www.csbmb.princeton.edu/mvpa for more
% information.
% 
% The Princeton MVPA toolbox is available free and
% unsupported to those who might find it useful. We do not
% take any responsibility whatsoever for any problems that
% you have related to the use of the MVPA toolbox.
%
% =====================================================================
%
% NOTE: This function was written to allow for SPM5 compatability,
% and assumes SPM5 is installed and unmodified.  Specifically, this
% function makes use of .nii input/output functions in
% spm_dir/@nifti/private, strangely enough...

%% Defaults and setup
% single is used to set whether you'd like to open your data as a single
% precision instead of a double precision.  This allows you to save a
% signifigant amount of memory if you don't actually need double precision.
defaults.single = false;

%capture the arguements and populate the default values.
args = propval(varargin,defaults);


%% Check for spm functions
spmPres = which('spm_vol.m');
if isempty(spmPres)
  error('SPM not found.');
end

%% Mask Setup
maskvol = get_mat(subj,'mask',maskname);
mDims   = size(maskvol); %#ok<NASGU> %get the dimensions of the mask
mask    = find(maskvol);%get the relevant indexes of the mask (all non zero's)
mSize   = length(mask);%get the size of the mask

% check mask isn't empty
if isempty(mask)
  error('Empty mask passed to load_spm_pattern()');
end


%% Initialize the data structure
subj = init_object(subj,'pattern',new_patname);

%% Convert filenames to a cell array
%if the file name is an array of characters
if ischar(filenames)
    
  if ~isempty(strfind(filenames,'*'))
    [pat,jnk,jnk] = fileparts(filenames); %#ok<NASGU>
    tmp = dir(filenames);
    filenames = {tmp(:).name};
    if ~isempty(pat)
      for i=1:length(filenames)
	filenames{i} = [pat '/' filenames{i}];
      end
    end   
  else
    filenames = {filenames};
  end
  
elseif ~iscell(filenames)
  error('Filenames are not in form of char or cell.');
end

nFiles = length(filenames);

%% Initialize the data structure
tmp_data = zeros(mSize ,nFiles); %#ok<NASGU>

disp(sprintf('Starting to load pattern from %i SPM files',nFiles));

%% Create a volume structure
vol = spm_vol(filenames);
tmp_data = []; %#ok<NASGU>

    %%%%%%%%%%%%%%%%%%%%%%
    %sylvains contribution
    %%%%%%%%%%%%%%%%%%%%%%

total_m = 0;

% compute total number of EPI images
for h = 1:nFiles
  [m n] = size(vol{h}); %#ok<NASGU>
  total_m = total_m + m;
end;

% allocate all at once to avoid reshaping iteratively
tmp_data = zeros(mSize, total_m);

total_m = 0;
    %% end contribution
for h = 1:nFiles % start looping thru the files being used.
  if mod(h,100)==0
    fprintf('\t%i',h);
  end
  
  [m n] = size(vol{h}); %#ok<NASGU>
  
  tmp_subvol=zeros(mSize,m);
  for i = 1:m
     curvol = vol{h}(i);
     
     % Enforce mask size
%     if ~all(curvol.dim == size(maskvol))
     if ~isequal(curvol.dim,size(maskvol))
       error(['Supplied mask is not the proper size for this dataset. mask: ' maskname ' filename: ' filenames{h}]);
     end
     % Load the data from the IMG file
     [Vdata] = spm_read_vols(curvol);
     
     if other_args.globalnorm
         Vdata=Vdata-other_args.globaltimecourse(h);
     end
     
     if args.single
       Vdata = single(Vdata);
     end
     
     tmp_subvol(1:mSize,i) = Vdata(mask);
     
  end
  
  % Reshape the data to be Voxels X Time
    %%%%%%%%%%%%%%%%%%%%%%
    %sylvains contribution
    %%%%%%%%%%%%%%%%%%%%%%
    tmp_data(1:mSize,total_m+1:total_m+m) = tmp_subvol;
    total_m = total_m + m;
    clear tmp_subvol;
    %% end contribution
    
end % for h

disp(' ');

%% Store the data in the pattern structure
subj = set_mat(subj,'pattern',new_patname,tmp_data);

%% Set the masked_by field in the pattern
subj = set_objfield(subj,'pattern',new_patname,'masked_by',maskname);

%% Add the history to the pattern
hist_str = sprintf('Pattern ''%s'' created by load_spm_pattern',new_patname);
subj = add_history(subj,'pattern',new_patname,hist_str,true);

%% Add information to the new pattern's header, for future reference
subj = set_objsubfield(subj,'pattern',new_patname,'header', ...
			 'vol',vol,'ignore_absence',true);

%% Load the subject         
% This object was conceived under a tree. Store that information in
% the SUBJ structure
created.function = 'load_spm_pattern';
created.args = args;
subj = add_created(subj,'pattern',new_patname,created);
end %main function
