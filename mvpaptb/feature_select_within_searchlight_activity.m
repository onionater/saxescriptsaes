function feature_select_within_searchlight_activity(subject, fs_args) 

% extract the "rest" selector
for select=1:length(subject.selectors)
    if strcmp(subject.selectors{select}.name, 'rests')
        restreg=subject.selectors{select}.mat;
    end
end
restreg(2,:)=(restreg==0); %first row is stimulus regressor, second row is rest regressor

%check if these selectors or regressors have been made already
searchedsel = exist_object(subject,'selector','allselector');
searchedreg = exist_object(subject,'regressors','restvsstim');

%make this a regressor for rest vs. stimulus
if ~searchedreg
subject = init_object(subject,'regressors','restvsstim');
subject = set_mat(subject,'regressors','restvsstim',restreg);
end

%make selector of same length
if ~searchedsel
allsel=(restreg(1,:)*0)+1;
subject = init_object(subject,'selector','allselector');
subject = set_mat(subject,'selector','allselector',allsel);
end

%update your regnames and selgroups
selname='allselector';
regsname='restvsstim';

 % Name the new statmap pattern and thresholded mask that will be created
  cur_maskname = args.new_maskstem;
  cur_map_patname = args.new_map_patname;

  % if a pattern with the same name already exists, it
  % will trigger an error later in init_object, but we
  % want to catch it here to save running the entire
  % statmap first
  if exist_object(subject,'pattern',cur_map_patname)
    error('A pattern called %s already exists',cur_map_patname);
  end
  
  if ~isempty(args.statmap_arg) && ~isstruct(args.statmap_arg)
    warning('Statmap_arg is supposed to be a struct');
  end
  
  % Add the current iteration number to the extra_arg, just in case
  % it's useful
  args.statmap_arg(1).cur_iteration = 1;

  % Create a handle for the statmap function handle and then run it
  % to generate the statmaps
  subject = statmap_anova(subject,cur_data_patname,regsname,cur_selname,cur_map_patname,args.statmap_arg);

if fs_args.fixed
  statvect=get_mat(subjrcy, 'pattern', cur_map_patname);
  [sortedValues,sortIndex] = sort(statvect,'ascend');
  if fs_args.fixednum<length(sortIndex)
      lowerbound=statvect(sortIndex(fs_args.fixednum));
  else
      lowerbound=statvect(sortIndex(end));
  end
  cur_maskname=[data_patin '_top' num2str(fs_args.fixednum) '_1'];
  args.new_maskstem=[data_patin '_top' num2str(fs_args.fixednum)];
  subject = create_thresh_mask(subject,cur_map_patname,cur_maskname,lowerbound);
%   subj = init_object(subj,'mask',cur_maskname);
%   subj = set_mat(subj,'mask',cur_maskname,statvect)
  subject = set_objfield(subject,'mask',cur_maskname,'group_name',[args.new_maskstem '_base']); %this group name will be what you set xvalmask to for crossvalidation
end


end