function [group_responses, group_rts, group_misses, condensedresps, forplotting, subjectlabels] =getbehavedata(subjectlist, task)
%aes 8/9/13
cd('/mindhive/saxelab2/EIB/behavioural/')

%% update for particular study format
conditions={'mu','fu','mh','fh','nu','su','nh','sh'} % will be labeled as 1:8
numconds=length(conditions)
possibleresponses=[1:4]
nullresponses=9999
%%

numSubj=length(subjectlist);


for s=1:numSubj
    subject=subjectlist{s};
    subjectlabels{s}=subject
    behavefilenames=[subject '.' task '.*.mat']
    behavefiles=dir(behavefilenames);
    numRuns=length(behavefiles);
    for r=1:numRuns
        behave=behavefiles(r).name;
        bdata=load(behave);
        fields=fieldnames(bdata);
        hasthatfield=ismember(fields, 'sorted_keys');
        items=bdata.item_orders;
        realitems=find(~ismember(items, 'null'))
        items=items(realitems)
        if find(hasthatfield)
        thisrunresponses=bdata.sorted_keys;
        thisrunresponses(thisrunresponses==0)=NaN;
        thisrunRTs=bdata.sorted_RTs;
        thisrunRTs(thisrunRTs==0)=NaN;
        for c=1:numconds
        thiscondresp=thisrunresponses(:,c);
        thiscondrts=thisrunRTs(:,c);
        responses(r,c)= mean(thiscondresp(~isnan(thiscondresp)));
        rts(r,c)= mean(thiscondrts(~isnan(thiscondrts)));
        misses(r,c)=sum(isnan(thiscondresp));
        end
        end
        clearvars bdata
    end
    for c=1:numconds
    condresp=responses(:,c);
    condrts=rts(:,c);
    group_responses(s,c)=mean(condresp(~isnan(condresp)));
    group_rts(s,c)=mean(condrts(~isnan(condrts)));
    group_misses(s,c)=sum(misses(:,c));
    end
    clearvars responses misses rts
end 

condensedresps(:,1)=mean(group_responses(:,1:2),2);
condensedresps(:,2)=mean(group_responses(:,3:4),2);
condensedresps(:,3)=mean(group_responses(:,5:6),2);
condensedresps(:,4)=mean(group_responses(:,7:8),2);
condensedSEM=std(condensedresps)/sqrt(size(condensedresps,1));
forplotting.labels={'faces-neg', 'faces-pos', 'context-neg', 'context-pos'};
forplotting.means=mean(condensedresps);
forplotting.SEM=condensedSEM;

save('/mindhive/saxelab2/EIB/behaviouralresponses.mat', 'group_responses', 'group_rts', 'group_misses', 'condensedresps', 'forplotting', 'subjectlabels')

end