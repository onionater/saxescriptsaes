function assess_quality_AES(study, subjectlist, tasklist)
%% e.g. per_run_motion=assess_quality_AES('EIB', makeIDs('EIB',[1:13, 15:20]), {'tomloc', 'EmoBioLoc', 'EIB_main'})
%%march 2013
%% wrapper for various quality scripts by AES
%% run this one script for the study and get full set of relevant quality measures
rootdir=['/mindhive/saxelab2/' study];
addpath(rootdir)
addpath('/mindhive/saxelab/scripts/aesscripts');

motion = build_motion_report(study,subjectlist);
save('motion_report_summary.mat', 'motion');
p=gcf;
saveas(p, [rootdir '/group_quality/motion_report.fig']);
hold off
clear gcf
close(1)


for t=1:length(tasklist)
task=tasklist{t};
report_big_motion(study,task,subjectlist);
files=quality_AES(study, task, subjectlist);
clear gcf
end

cd([rootdir '/group_quality'])
end