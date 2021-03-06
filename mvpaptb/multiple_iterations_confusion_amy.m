function [confmat guesses desireds] = multiple_iterations_confusion_amy(results,varargin)

% [CONFMAT GUESSES DESIREDS] = MULTIPLE_ITERATIONS_CONFUSION(RESULTS,...)
%
% Loops over the iterations in results, concatenating the
% results.iterations(i).perfmet.desireds and guesses, and then
% feeds in the concatenated versions into confusion to create the
% confusion matrix
%
% SAVE_FILE (optional, default = 'confmat.txt'). By default,
% this will save the CONFMAT matrix out to a .txt file
% (called SAVE_FILE). If you don't want it to save, then
% just set this to ''.
%
% PERFMET_NAME (optional, default = ''). If you only have
% one performance metric, then you don't have to worry
% about this at all, since it will just use whichever one
% you have. If you have multiple performance
% metrics, this allows you to choose which one to
% use. Feed in the function name of the one you want to
% use, e.g. 'perfmet_xcorr'. If you have multiple
% performance metrics, you must specify which one to use,
% or the function will fail fatally.

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

defaults.perfmet_name = '';
args = propval(varargin,defaults);

nPerfmets = length(results.iterations(1).perfmet);
if nPerfmets>1
  pm_no = get_perfmet_called(results,args.perfmet_name);
end

guesses = [];
desireds = [];

for i=1:length(results.iterations)
  if nPerfmets==1
    guesses = [guesses results.iterations(i).perfmet.guesses];
    desireds = [desireds results.iterations(i).perfmet.desireds];
  else
    guesses = [guesses results.iterations(i).perfmet{pm_no}.guesses];
    desireds = [desireds results.iterations(i).perfmet{pm_no}.desireds];
   end % dealing with 1 vs many perfmets 
end % r

confmat = confusion_amy(guesses,desireds, 0);


