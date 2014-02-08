function [E,params]=design_efficiency_amy(varargin)
% DESIGN_EFFICIENCY Efficiency computation for fMRI block designs
%AES stole from mindhive software on 4/18/13. no changes.
%
% E=design_efficiency(argname1,argvalue1,argname2,argvalue2,...)
% possible argument names are:
%   'conditions'    : Condition indexes for each block (vector of M numbers representing indexes ranging from 1 to N)
%   'blocklengths'  : Block lengths for each condition (vector of N numbers representing length in seconds of each condition block)
%   'TR'			: Repetition time (scalar representing acquisition period in seconds)
%	'contrast'	    : Contrast vector or matrix (vector of KxN numbers representing contrast weights)
%	'hparam'		: High-pass filter value (in seconds). It defaults to 120
%	'nscans'		: Total number of scans. It defaults to sum(blocklenghts(conditions))/TR-1
%
% Example:
% E=design_efficiency('conditions',repmat([1,2],[1,10]),'blocklengths',[16,16],'contrast',[1,-1]);
%

% alfnie@gmail.com

fields={'conditions',[],...    
        'blocklengths',[],...
        'tr',[],...
        'contrast',[],...
		'hparam',120,...
		'nparam',120,...
		'nscans',[]};
params=[]; for n1=1:2:nargin, params=setfield(params,lower(varargin{n1}),varargin{n1+1}); end
for n1=1:2:length(fields), if ~isfield(params,fields{n1}) | isempty(getfield(params,fields{n1})), if isstr(fields{n1+1}), params=setfield(params,fields{n1},eval(fields{n1+1})); else, params=setfield(params,fields{n1},fields{n1+1}); end; end; end

params.conditions=double(params.conditions(:)'-min(params.conditions)+1);
params.blocklengths=params.blocklengths(:)';

% Creates design matrix
times=[0,cumsum(params.blocklengths(params.conditions))];
Ltotal=times(end);
params.tr0=params.tr/ceil(params.tr*10);
if isempty(params.nscans), params.nscans=ceil(Ltotal/params.tr)-1; end
t=(0:params.nscans)*params.tr0;
X=zeros(length(t),length(params.blocklengths));
for n1=1:length(params.conditions),
	X(times(n1)/params.tr0+1:times(n1+1)/params.tr0,params.conditions(n1))=1;
end

% Convolves design matrix with hrf
h=spm_hrf(params.tr0);
X=convn(X,h,'all');
params.X=X;
% Convolves design matrix with high-pass filter & inverse noise
n=pow2(ceil(log2(size(X,1))));
X=fft(X,n);
f=[0:n/2,n/2-1:-1:1]'/params.tr0/n;
noiseinv=(f>1/params.hparam)./(1+1./max(eps,f*params.nparam));
X=repmat(noiseinv,[1,size(X,2)]).*X;
X=real(ifft(X,n));
k=1/mean(1./noiseinv(noiseinv>0).^2); 
params.X2=X;
params.t=(0:size(X,1)-1)*params.tr0;

% Resamples design matrix and computes efficiency
X=X(min(size(X,1),1+(0:params.nscans-1)*params.tr/params.tr0),:);
E=zeros(size(params.contrast,1),1);
for n1=1:size(params.contrast,1), E(n1)=k./trace(params.contrast(n1,:)*pinv(X'*X)*params.contrast(n1,:)'); end
end