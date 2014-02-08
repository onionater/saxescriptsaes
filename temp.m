function temp
condnames={'belief','photo', 'fff'}
numRuns=3;
task_avgtrans_corr=[.8 .2 .9; .5 .7 .9; .1 .1 .5]
subplot(3,1,1);bar(task_avgtrans_corr(:,:));ylim([0 1]);ylabel('r value');title('task-trans corr  ');set(gca,'XTick',[])
legend(condnames, 'location', 'NorthEastOutside');
set(gca,'XTick',1:numRuns)
xlabel('run #')
p=gcf;
saveas(p, [pwd '/tomloc' '_task-motion_plots_']);
hold off
clear gcf
end