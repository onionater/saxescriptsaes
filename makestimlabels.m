function [group_itemnames, group_keys, group_itemnumbers, subjectlist] =makestimlabels(subjectlist, task)
%aes 8/9/13

cd('/mindhive/saxelab2/EIB/behavioural/')
%% update for particular study format
conditions={'mu','fu','mh','fh','nu','su','nh','sh'} % will be labeled as 1:8
%%

numSubj=length(subjectlist);

for s=1:numSubj
    subject=subjectlist{s};
    behavefilenames=[subject '.' task '.*.mat'];
    behavefiles=dir(behavefilenames);
    numRuns=length(behavefiles);
    subjitems=[];
    subjkeys=[];
    for r=1:numRuns
        behave=behavefiles(r).name;
        bdata=load(behave);
        items=bdata.item_orders;
        keys=bdata.key;
        realitems=find(~ismember(items, 'null'));
        realkeys=find(keys~=9999);
        items=items(realitems);
        keys=keys(realkeys);
        subjitems=[subjitems; items];
        subjkeys=[subjkeys; keys];
        clearvars bdata
    end
    group_itemnames(:,s)=subjitems;
    group_keys(:,s)=subjkeys;
    clearvars subjitems subjkeys
end 

%stim number = condition (1 to 8) * stimnumber (1:24)
maxstimnumber=24
for x=1:size(group_itemnames,1)
    for y=1:size(group_itemnames,2)
        stim=group_itemnames{x,y}; 
        stim=stim(1:end-1);
        condname=stim(1:2);
        condnumber=find(ismember(conditions, condname));
        stimnumber=str2num(stim(end-1:end));
        group_itemnumbers(x,y)=(condnumber-1)*maxstimnumber+stimnumber;
    end
end

save('/mindhive/saxelab2/EIB/mvpaptb/subject_stim_orders.mat', 'group_itemnames', 'group_keys', 'group_itemnumbers', 'subjectlist')