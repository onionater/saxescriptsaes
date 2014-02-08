function freetheRFX(study, analysis_name, searchkey, varargin)
%edited from saxelab_RFX... specify a set of images (e.g. SL images of accuracy-chance) and perform a simple
%RFX t-test comparing to 0.

% Updated random-effects analysis script. This script will extend the
% functionality of saxelab_RFX_bch_spm8 by adding two sample t-tests, such
% that two groups can be compared. Also added is the ability to construct
% across-subject regressors (i.e., age) as a covariate of interest. 
% 
% The script requires several bits of information, which can be provided
% optionally when the function is called, otherwise the information will be
% acquired via a series of prompts. The full specification is:
%	saxelab_RFX(type,{'study'},'task',imagefile,{'subjects'})
%		where: 
%			:: type : an INTEGER indicating what type of RFX analysis that 
%			will be performed. 
%				1 : one-sample T-test (for one group)
%				2 : two-sample T-test (for comparing two groups)
%				3 : one-sample T-test with covariate
%				4 : two-sample T-test with covariate *** this has yet to be
%						implemented!
%			:: study : a STRING indicating which study the RFX analysis
%			will be performed on. Alternatively, study may be a CELL ARRAY
%			that contains multiple studies. 
%			:: task : a STRING indicating the name of the task folder that
%			contains the images of interest. Alternatively, task may be a
%			CELL ARRAY that contains multiple tasks.
%			:: subjects : a CELL ARRAY of STRINGS specifying the subjects
%			on which the analysis is to be performed. 
%			:: imagefile : a STRING or an INTEGER specifying the images to 
%			be used for the analysis. If a STRING, it must specify the name 
%			of the contrast, in whole or in part. If an INTEGER, the script
%			will select that contrast file from each and every subject. 
%
%
%% ===================== Parse Input Arguments ============================
P=spm_select(inf,'img','Choose subject con images for RFX','',pwd,'.*',1);
numSubj=size(P,1)
cd /mindhive/saxelab2/EIB/
%type = listdlg('ListSize',[300 100],'SelectionMode','single','PromptString','Select analysis type','ListString',{'Single-Group RFX Analysis','Two-Group RFX Analysis','Single-Group RFX Analysis with Covariate','Two-Group RFX Analysis with Covariate'});
type=1;
templatefile = spm_vol(P(1,:));
%% ===================== Move Images ============================
% First, make the RFX analysis folder. This is in the first study folder
% selected by the user. 
fprintf('Creating directory for analysis...\n');
expDir = ['/mindhive/saxelab2/' study '/'];
cd(expDir);
cd RandomEffects
mkdir(analysis_name)
cd(analysis_name)
% going to try to get this to work without these
%if any(analysis_name{:} == '>')
%    analysis_name = {regexprep(analysis_name{:},'>',' greater than ')};
%elseif any(analysis_name{:} == '<')
%    analysis_name = {regexprep(analysis_name{:},'<',' less than ')};
%end 
% % 
% % imstocopy = numel([subjects{:}]);
% % fprintf('Copying Images\n');
% % fprintf('Copying image . . . ');
% % im_count = 0;
% % % iterate over subjects -- this may become a problem if a given subject has
% % % BOTH tasks present. Nonetheless, I don't expect this will be a terribly
% % % large problem. 
% % thelen = 0;
% % for grp=1:length(subjects)
% % 	for subj=1:length(subjects{grp})
% %         try 
% %             cd(subjects{grp}{subj}(2:end));
% %         catch
% %             cd(subjects{grp}{subj}(1:end));
% %         end
% %         tsub = regexp(subjects{grp}{subj},'/','split');
% %         tsub = tsub{5};
% % 		load('SPM.mat');
% % 		im_count = im_count + 1;
% %         fprintf(repmat('\b',1,thelen));
% % 		strin = sprintf('%.0f/%.0f',im_count,imstocopy);
% % 		fprintf(strin);
% % 		switch con_type
% % 			case 1
% % 				% they are using a numeral
% %                 try
% %                     con_file = SPM.xCon(contrast).Vcon.fname;
% %                     con_descript = SPM.xCon(contrast).Vcon.descrip;
% %                 catch
% %                     fprintf('Couldn''t find the xCon field in the SPM file for subject #%d...looking for the actual image...',subj);
% %                     try 
% %                         man_con_file = dir(sprintf('con_%04d.img',contrast));
% %                         con_file = man_con_file(1).name;
% %                         fprintf('found it!\n\n\n\n');
% %                     catch
% %                         fprintf('\nCouldn''t find an appropriate file...terminating.\n');
% %                         return;
% %                     end
% %                 end
% % 			case 2
% % 				% they are using a string
% % 				for con = 1:length(SPM.xCon)
% % 					if strfind(upper(SPM.xCon(con).Vcon.descrip),upper(contrast))
% % 						con_file = SPM.xCon(con).Vcon.fname;
% %                         con_descript = SPM.xCon(con).Vcon.descrip;
% % 						break;
% % 					end
% % 				end
% %         end
% %         num_sessions = length(SPM.nscan); 
% % 		clear SPM;
% % 		%copy_cmd_img = sprintf('!cp %s%s %s/%s_grp_%s_sub.img',subjects{grp}{subj}(2:end),con_file,expDir,num2str(grp),num2str(subj));
% % 		%copy_cmd_hdr = sprintf('!cp %s%s %s/%s_grp_%s_sub.hdr',subjects{grp}{subj}(2:end),regexprep(con_file,'.img','.hdr'),expDir,num2str(grp),num2str(subj));
% % 		% just copy it via imcalc
% % 		f = sprintf('i1/%d',num_sessions);
% % 		Vi = spm_vol(con_file);
% % 		Q = sprintf('%s/%s_grp_%s_con_%s.img',expDir,num2str(grp),tsub,con_descript);
% % 		Vo = struct('fname',Q,'dim',Vi(1).dim(1:3), 'dt', [spm_type('float32'), 1],'mat',Vi(1).mat,'descrip','spm - algebra','mask',1);
% % 		spm_imcalc(Vi,Vo,f);
% % 		P{im_count} = sprintf('%s/%s_grp_%s_con_%s.img',expDir,num2str(grp),tsub,con_descript);
% % 		% eval(copy_cmd_img);
% % 		% eval(copy_cmd_hdr);
% % 		thelen = length(strin);
% % 	end
% % end
% % fprintf('\nImages copied successfully!\n');
% % end
% % cd(expDir);
% % save('RFX_state');

%% ===================== Begin RFX Analysis ============================
for thissubj=1:numSubj
SPM.xY.P{thissubj,1}=P(thissubj,:)
end
SPM.xY.VY		= spm_vol(char(P));
iGMsca			= 9;								% no grand mean scaling
M_X				= 0;								% no masking
iGXcalc			= 1;								% omit global calculation
sGXcalc			= 'omit';
sGMsca			= '<no grand Mean scaling>';
sGloNorm		= '<no global normalisation>';

spm_defaults;

I = ones(numSubj,4);
for i=1:numSubj,I(i,1)=i;end 
D = spm_spm_ui('DesDefs_Stats');
H = [];B = [];C = [];G = [];
Hnames = {};Bnames = {}; Cnames = {}; Gnames = {};
% Thus far it appears that the substantial differences between types of
% RFX analysis is that they have different D structures and regressor
% matricies; everything else stays much the same. 

		D = D(1);
		H = ones(numSubj,1);
		Hnames = {'mean'};

if size(B,2)~=1
	B = reshape(B,[],1);
end
X = [H C B G];
tmp    = cumsum([size(H,2), size(C,2), size(B,2), size(G,2)]);
SPM.xX     = struct(	'X',		X,...
	'iH',		[1:size(H,2)],...
	'iC',		[1:size(C,2)] + tmp(1),...
	'iB',		[1:size(B,2)] + tmp(2),...
	'iG',		[1:size(G,2)] + tmp(3),...
	'name',		{[Hnames; Cnames; Bnames; Gnames]},...
	'I',		I,...
	'sF',		{D.sF});
tmp = {	sprintf('%d condition, +%d covariate, +%d block, +%d nuisance',...
	size(H,2),size(C,2),size(B,2),size(G,2));...
	sprintf('%d total, having %d degrees of freedom',...
	size(X,2),rank(X));...
	sprintf('leaving %d degrees of freedom from %d images',...
	size(X,1)-rank(X),size(X,1))				};
SPM.xsDes = struct(	'Design',			{D.DesName},...
	'Global_calculation',		{sGXcalc},...
	'Grand_mean_scaling',		{sGMsca},...
	'Global_normalisation',		{sGloNorm},...
	'Parameters',			{tmp}			);

% scan number
SPM.nscan	= size(SPM.xX.X,1);


% save progress in cwd
save SPM SPM

% And now give it a shot:
fprintf('Estimating Design for %d subjects\n',numSubj);
SPM = spm_spm(SPM);

% Finally, let's produce the contrasts.
SPM.xCon=[];
SPM.xCon.name= 'DiffFromChance';
SPM.xCon.STAT='T';
SPM.xCon.c=1;
SPM.xCon.X0.ukX0=0;
SPM.xCon.iX0='c';
%SPM.xCon.X1o.ukX1o=0.25;
SPM.xCon.eidf=1.000;
SPM.xCon.Vcon=templatefile;
SPM.xCon.Vcon.fname=[pwd '/beta_0001.img'];
SPM.xCon.Vcon.descrip= 'SPM contrast - 1: DiffFromChance';
SPM.xCon.Vspm=[];
% SPM.xCon.Vspm=templatefile;
% SPM.xCon.Vspm.fname='spmT_0001.img';
% SPM.xCon.Vcon.descrip= 'SPM{T_[15.0]} - contrast 1: DiffFromChance';

switch type
	case 2
		con_vals = [1 -1 0];
		con_name = 'Group 1 versus Group 2';
		SPM.xCon(end+1) = spm_FcUtil('Set', con_name, 'T', 'c', con_vals',SPM.xX.xKXs);
		con_vals = [-1 1 0];
		con_name = 'Group 2 versus Group 1';
		SPM.xCon(end+1) = spm_FcUtil('Set', con_name, 'T', 'c', con_vals',SPM.xX.xKXs);
	case 3
		con_vals = [1 0];
		con_name = sprintf('%s versus rest',covariate_name{:});
		SPM.xCon(end+1) = spm_FcUtil('Set', con_name, 'T', 'c', con_vals',SPM.xX.xKXs);
	case 4
		con_vals = [1 -1 0 0];
		con_name = 'Group 1 versus Group 2';
		SPM.xCon(end+1) = spm_FcUtil('Set', con_name, 'T', 'c', con_vals',SPM.xX.xKXs);
		con_vals = [-1 1 0 0];
		con_name = 'Group 2 versus Group 1';
		SPM.xCon(end+1) = spm_FcUtil('Set', con_name, 'T', 'c', con_vals',SPM.xX.xKXs);
		con_vals = [0 0 1 0];
		con_name = sprintf('%s versus rest',covariate_name{:});
		SPM.xCon(end+1) = spm_FcUtil('Set', con_name, 'T', 'c', con_vals',SPM.xX.xKXs);
end

%spm_contrasts(SPM);


end