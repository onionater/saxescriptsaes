function quickrunGLM(study, task, subjectlist, varargin)
%e.g. quickrunGLM('EIB', 'tomloc', makeIDs('EIB', [1:3, 16:20]), {'add_art',1})
runs=listbolds(task, subjectlist);

for s=1:length(subjectlist)
    subject=subjectlist{s}
    bold=runs{s};
    if size(varargin,1)>0
    inputparam=varargin{1}
    saxelab_model_2012(study,subject,task,bold, inputparam);
    else
    saxelab_model_2012(study,subject,task,bold);
    end
end