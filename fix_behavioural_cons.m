function fix_behavioural_cons(study, task, subjects)
%% e.g. fix_behavioural_cons('EIB', 'EIB_main', makeIDs('EIB',[1:13, 15:20]))
%% originally modeled durations as just movie period, but want to change it to include response window (5.75 sec/2.875 TR)

rootdir=['/mindhive/saxelab2/' study '/behavioural/']; 
cd(rootdir);
s=size(subjects);
numSubj=s(2);
for subj=1:numSubj
    disp(subjects{subj})
    runs=adir([subjects{subj} '.' task '*']);
    for i=1:length(runs)
        disp(i)
% con_info(1).name = 'sh>su';
% con_info(1).vals = [0 0 0 0 0 -1 0 1];
% con_info(2).name = 'su>sh';
% con_info(2).vals = [ 0 0 0 0 0 1 0 -1];
% con_info(3).name = 'nh>nu';
% con_info(3).vals = [0 0 0 0 -1 0 1 0];
% con_info(4).name = 'nu>nh';
% con_info(4).vals = [0 0 0 0 1 0 -1 0];
% con_info(5).name = 'fh>fu';
% con_info(5).vals = [0 -1 0 1 0 0 0 0];
% con_info(6).name = 'fu>fh';
% con_info(6).vals = [0 1 0 -1 0 0 0 0];
% con_info(7).name = 'mh>mu';
% con_info(7).vals = [-1 0 1 0 0 0 0 0];
% con_info(8).name = 'mu>mh';
% con_info(8).vals = [1 0 -1 0 0 0 0 0];
% con_info(9).name = 'context_h>context_u';
% con_info(9).vals = [0 0 0 0 -1 -1 1 1];
% con_info(10).name = 'context_u>context_h';
% con_info(10).vals = [0 0 0 0 1 1 -1 -1];
% con_info(11).name = 'face_h>face_u';
% con_info(11).vals = [-1 -1 1 1 0 0 0 0];
% con_info(12).name = 'face_u>face_h';
% con_info(12).vals = [1 1 -1 -1 0 0 0 0];
% con_info(13).name = 'all_h>all_u';
% con_info(13).vals = [-1 -1 1 1 -1 -1 1 1];
% con_info(14).name = 'all_u>all_h';
% con_info(14).vals = [1 1 -1 -1 1 1 -1 -1];
% con_info(15).name = 'face>context';
% con_info(15).vals = [1 1 1 1 -1 -1 -1 -1];
% con_info(16).name = 'context>face';
% con_info(16).vals = [-1 -1 -1 -1 1 1 1 1];
% con_info(17).name = 'social>nonsocial';
% con_info(17).vals = [0 0 0 0 -1 1 -1 -1];
% con_info(18).name = 'nonsocial>social';
% con_info(18).vals = [0 0 0 0 1 -1 1 -1];
% con_info(19).name = 'male>female';
% con_info(19).vals = [1 -1 1 -1 0 0 0 0];
% con_info(20).name = 'female>male';
% con_info(20).vals = [-1 1 -1 1 0 0 0 0];

con_info(1).name = 'All>Rest';
con_info(1).vals = [1 1 1 1 1 1];
con_info(2).name = 'Bio>ObjMot';
con_info(2).vals = [1 -1 0 0 0 0];
con_info(3).name = 'Emo>Bio';
con_info(3).vals = [-2 0 1 1 0 0];
con_info(4).name = 'Happy>Bio';
con_info(4).vals = [-1 0 1 0 0 0];
con_info(5).name = 'Sad>Bio';
con_info(5).vals = [-1 0 0 1 0 0];
con_info(6).name = 'Happy>Sad';
con_info(6).vals = [0 0 1 -1 0 0];
con_info(7).name = 'Sad>Happy';
con_info(7).vals = [0 0 -1 1 0 0];
con_info(8).name = 'Faces>Objects';
con_info(8).vals = [0 0 0 0 1 -1];
con_info(9).name = 'Objects>Faces';
con_info(9).vals = [0 0 0 0 -1 1];
con_info(10).name = 'BSH > ObjMot'; %% all biomotion stimuli > sfm
con_info(10).vals= [1 -3 1 1 0 0];

    save([subjects{subj} '.' task '.' num2str(i) '.mat'],'-append','con_info');
    
end
end
end
