function saxelab_checkreg_amy(study, subject, prefix, runs, imnum)
% saxelab_checkreg(study, subject, prefix[, runs, imnum])
%   brings up the check registration screen for a subject. 
%   study = STRING, the study of interest
%   subject = STRING, the subject of interest
%   prefix = STRING, the filename prefix for each image (i.e., 'swrf')
%   runs = [optional], ARRAY of runs of interest (if not included or empty,
%          will return images from all functional runs)
%   imnum = [optional], INTEGER of the volume of interest (if not included
%          or empty, will return the first image)
%
% examples:
%   saxelab_checkreg('MOT','SAX_MOT_08b','swrf')
%       - returns the first volume of each run, that have been smoothed,
%       normalized, and realigned. 
%   saxelab_checkreg('MOT','SAX_MOT_08b','swrf',[],5)
%       - returns the fifth volume of each run, that have been smoothed,
%       normalized and realigned.
%   saxelab_checkreg('MOT','SAX_MOT_08b','swrf',[10 12 14])
%       - returns the first volume of bold directories 10, 12 and 14
%--------------------------------------------------------------------------
% Check inputs
    if ~nargin
        help saxelab_checkreg
        return
    end
    if ~exist('study','var')||~ischar(study)
        error('Please define a study as a string of characters');
    end
    if ~exist('subject','var')||~ischar(subject)
        error('Please define a subject as a string of characters');
    end
    if ~exist('prefix','var')||~ischar(prefix)
        warning('Please define a file prefix as a string of chatacters');
        prefix = 'swrf';
    end
    if ~exist('imnum','var')||isempty(imnum)
        imnum = 1;
    end
    if (exist('runs','var')&&~isempty(runs))&(~strcmpi(class(runs),'double')||~all(round(runs)==runs))
        error('Run specification must be integers')
    end
% Try to find study
    subj = adir(fullfile('/mindhive/saxelab*/',study,subject));
    if ~iscell(subj)
        error('Could not find either study or subject! Please check subject spelling!');
    end
    subj = subj{1};
% Find the images
    if ~exist('runs','var')||isempty(runs)
        images = adir(fullfile(subj,'bold','*',[prefix '*-' sprintf('%06d',imnum) '-*.img']));
        if isnumeric(images)
            error('No images found -- are you sure you are using the correct prefix?');
        end
    else
        images = {};
        for i = runs
            tmp = adir(fullfile(subj,'bold',sprintf('%03d',i),[prefix '*-' sprintf('%06d',imnum) '-*.img']));
            if iscell(tmp)
                images = [images tmp(1)];
            end
        end
    end
% Find anatomical image
    if ~any('w' == prefix)
        % use the unnormalized anatomical...
        anat = adir(fullfile(subj,'3danat','s*.img'));
    else
        anat = adir(fullfile(subj,'3danat','ws*.img'));
    end
% Finally, perform checkreg
    spm_check_registration(spm_vol(char([anat(1) images])));
end