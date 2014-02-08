function gridSubmitAES(cmd,name,varargin)
% e.g. gridSubmitAES('run_searchlight(makeIDs(''EIB'', [2:5, 7:9]))', 'searchlightEIB_01', 'shfile', '/mindhive/saxelab/scripts/aesscripts/gridSearchlightSubmitAES')
%modified from BD wrapper
%cmd = whole command with strings in double quoates
%name= arbitrary name for this submission 
%note if you are calling a sh file you have already written (with a hardcoded command), this name and command won't get used. it will rely on the name in your
%sh file... so you can do something like this... gridSubmitAES('thesedontmatter', 'atall', 'shfile', '/mindhive/saxelab/scripts/aesscripts/gridSearchlightSubmitAES') 

%see http://mindhive.mit.edu/node/1345
%for more general info on the grid: http://web.njit.edu/topics/HPC/basement/sge/SGE.html

% Function to submit a matlab job to the torque grid engine.

if nargin<2
    disp('ERROR: Too few arguments.');
    return;
end;

scripts_dir = '/mindhive/saxelab/scripts/aesscripts';
use_sh = 0;

tmpfile = [scripts_dir '/tmp' int2str(ceil(rand*100000)) '.sh'];
writetempfile=1;

node = 'saxelab';
nodes = {'mindhive','bigmem','recon','saxelab'};

for i = 1:length(varargin)
    
    if strcmpi(varargin{i},'node') && ismember(varargin{i+1},nodes)
        node = varargin{i+1};
    end;
    
    if strcmpi(varargin{i},'shfile') 
        writetempfile=0;
        tmpfile = [varargin{i+1} '.sh'];
    end;
    
    if (strcmpi(varargin{i},'sh')||strcmpi(varargin{i},'usesh')||strcmpi(varargin{i},'use_sh'))...
            && ismember(varargin{i+1},[0 1])
        use_sh = varargin{i+1};
    end;
    
end;

if writetempfile

if use_sh
    template = [scripts_dir '/gridTemplateAESSh.sh'];
else
    template = [scripts_dir '/gridTemplateAES.sh'];
end;

% Load template file.
fid = fopen(template);
grid_template = [];
while 1
    line = fgetl(fid);
    if ~isnumeric(line)
        grid_template = [grid_template '\n' line];
    else break;
    end;
end;
fclose(fid);

% Replace variables in template file.
grid_template = strrep(grid_template,'{NAME}',name);
grid_template = strrep(grid_template,'{NODE}',node);
grid_template = strrep(grid_template,'{CMD}',cmd);

% Write script to run on grid.
fid = fopen(tmpfile,'w');
fprintf(fid,grid_template);
fclose(fid);

end

% Submit job to grid engine, then delete script.
system(['qsub ' tmpfile]);
pause(1);
if writetempfile
system(['rm -rf ' tmpfile]);
end


end