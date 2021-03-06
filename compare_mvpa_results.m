function compare_stability(key)
%% created by AES 4/7/13
% compares stability of conditions across repeat presentations for each
% result folder
%% key = keyword that filters MVPA results directories (e.g. 'global' would get only 
%% global normalized MVPA results (can use key='' to get all results)

%% uses whatever data type is in data structure (this should be Zrawdata)
%% ghetto, but this whole thing assumes only 2 suffixes. too lazy to make generalize beyond this



rootdir='/mindhive/saxelab2/EIB/';
mvpadir=[rootdir 'EIB_mvpa/'];
mvpasummarydir=[mvpadir 'mvpa_summary'];
mkdir(mvpasummarydir)
cd(mvpadir)

%% find all results folders that match filter key
resultsfolders=dir(['MVPA_EIB*' key '*'])
s=size(resultsfolders);
numresults=s(1);

%% parameters:
ROItypes={'ind', 'group'}; % currently MVPA summary stats (within results directories) are separated by group vs. individual ROIs
suffixes={'vsr', 'vso'}; %whatever discrimination types you want, script assumes that data structure is organized by these suffixes
prioritynamepieces={'fake_pos_','fake_neg_'};

assessments={'within_avg', 'between_avg', 'diff_avg'};
priorities={'wP_avg', 'bP_avg', 'dP_avg'};
assessments_ste={'within_ste', 'between_ste', 'diff_ste'};
priorities_ste={'wP_ste', 'bP_ste', 'dP_ste'};
yrange={[-.5 2.5; -.3 .8], [-.5 2.5; -.8 .3], [-.08 .2; -.5 1]};

assessmentsROI={'withinROI_avg', 'betweenROI_avg', 'diffROI_avg'};
prioritiesROI={'wP_ROI_avg', 'bP_ROI_avg', 'dP_ROI_avg'};
assessmentsROI_ste={'withinROI_ste', 'betweenROI_ste', 'diffROI_ste'}; 
prioritiesROI_ste={'wP_ROI_ste', 'bP_ROI_ste', 'dP_ROI_ste'};


numAssessments=length(assessments);

% for each type of ROI (group and ind)
for rt=1:length(ROItypes)
    roi=['*' ROItypes{rt} '*']

%% go through each result folder
for result=1:numresults
    resultname=resultsfolders(result).name;
    resultlegend{result}=resultname;
    cd(resultname);

file=dir(roi);
file=file.name;
filename= [mvpadir resultname '/' file]
load(filename)
roilist=fieldnames(data); % creates cell array of fieldnames, each of which is an roi
numROIs=length(roilist);

% just to figure out dimensions
sampleROI=roilist{1};
roidata=(eval(['data.' sampleROI]));
disclist=fieldnames(roidata);
numdisc=length(disclist);

within=zeros(numdisc,numROIs*length(suffixes));
between=within;


for roinum=1:numROIs
sampleROI=roilist{roinum};
roidata=(eval(['data.' sampleROI]));
disclist=fieldnames(roidata);
numdisc=length(disclist);

for disc=1:numdisc
    discrimination=disclist{disc};
    w_vector=(eval(['data.' sampleROI '.' discrimination '.within']));
    b_vector=(eval(['data.' sampleROI '.' discrimination '.across']));
    d_vector=w_vector-b_vector;
    w_mean=mean(w_vector); b_mean=mean(b_vector); d_mean=mean(d_vector);
    w_ste=std(w_vector)/sqrt(length(w_vector)); b_ste=std(b_vector)/sqrt(length(b_vector)); d_ste=std(d_vector)/sqrt(length(d_vector));
    
    suffix=discrimination(end-2:end);
    suffixindex =find(strcmp(suffixes, suffix));
        priorityname=[prioritynamepieces{1}, suffix, prioritynamepieces{2}, suffix];

        within_mean(disc-((numdisc/length(suffixes))*(suffixindex-1)), suffixindex)=w_mean;
        withinste(disc-((numdisc/length(suffixes))*(suffixindex-1)), suffixindex)=w_ste;
        between_mean(disc-((numdisc/length(suffixes))*(suffixindex-1)), suffixindex)=b_mean;
        betweenste(disc-((numdisc/length(suffixes))*(suffixindex-1)), suffixindex)=b_ste;
        diff_mean(disc-((numdisc/length(suffixes))*(suffixindex-1)), suffixindex)=d_mean;
        diffste(disc-((numdisc/length(suffixes))*(suffixindex-1)), suffixindex)=d_ste;
        
        withinROI_mean(disc-((numdisc/length(suffixes))*(suffixindex-1)),suffixindex+length(suffixes)*roinum-length(suffixes))=w_mean;
        withinROIste(disc-((numdisc/length(suffixes))*(suffixindex-1)),suffixindex+length(suffixes)*roinum-length(suffixes))=w_ste;
        betweenROI_mean(disc-((numdisc/length(suffixes))*(suffixindex-1)),suffixindex+length(suffixes)*roinum-length(suffixes))=b_mean;
        betweenROIste(disc-((numdisc/length(suffixes))*(suffixindex-1)),suffixindex+length(suffixes)*roinum-length(suffixes))=b_ste;
        diffROI_mean(disc-((numdisc/length(suffixes))*(suffixindex-1)),suffixindex+length(suffixes)*roinum-length(suffixes))=d_mean;
        diffROIste(disc-((numdisc/length(suffixes))*(suffixindex-1)),suffixindex+length(suffixes)*roinum-length(suffixes))=d_ste;

        if strcmp(discrimination,priorityname); % if this discrimination is the prioritized one
                withinPriority_mean(1,suffixindex)=w_mean;
                betweenPriority_mean(1,suffixindex)=b_mean;
                diffPriority_mean(1,suffixindex)=d_mean;
                
                withinROI_priority_mean(1,suffixindex+length(suffixes)*roinum-length(suffixes))=w_mean;
                betweenROI_priority_mean(1,suffixindex+length(suffixes)*roinum-length(suffixes))=b_mean;
                diffROI_priority_mean(1,suffixindex+length(suffixes)*roinum-length(suffixes))=d_mean;
                
                withinPriority_ste(1,suffixindex)=w_ste;
                betweenPriority_ste(1,suffixindex)=b_ste;
                diffPriority_ste(1,suffixindex)=d_ste;
                
                withinROI_priority_ste(1,suffixindex+length(suffixes)*roinum-length(suffixes))=w_ste;
                betweenROI_priority_ste(1,suffixindex+length(suffixes)*roinum-length(suffixes))=b_ste;
                diffROI_priority_ste(1,suffixindex+length(suffixes)*roinum-length(suffixes))=d_ste;
        end
 
    end
end
        
        
        
     
%% compute mean of within, across and diff score across discriminations


within_avg{1,1}='resultname'; within_avg{1,2}=suffixes;
between_avg{1,1}='resultname'; between_avg{1,2}=suffixes;
diff_avg{1,1}='resultname'; diff_avg{1,2}=suffixes;

withinROI_avg{1,1}='resultname'; withinROI_avg{1,2}=roilist';
betweenROI_avg{1,1}='resultname'; betweenROI_avg{1,2}=roilist';
diffROI_avg{1,1}='resultname'; diffROI_avg{1,2}=roilist';

wP_avg{1,1}='resultname'; wP_avg{1,2}=suffixes;
bP_avg{1,1}='resultname'; bP_avg{1,2}=suffixes;
dP_avg{1,1}='resultname'; dP_avg{1,2}=suffixes;

wP_ROI_avg{1,1}='resultname'; wP_ROI_avg{1,2}=roilist';
bP_ROI_avg{1,1}='resultname'; bP_ROI_avg{1,2}=roilist';
dP_ROI_avg{1,1}='resultname'; dP_ROI_avg{1,2}=roilist';

within_avg{result+1,1}=resultname;
between_avg{result+1,1}=resultname;
diff_avg{result+1,1}=resultname;

   wP_avg{result+1,1}=resultname;
   bP_avg{result+1,1}=resultname;
   dP_avg{result+1,1}=resultname;
   
withinROI_avg{result+1,1}=resultname;
betweenROI_avg{result+1,1}=resultname;
diffROI_avg{result+1,1}=resultname;

   wP_ROI_avg{result+1,1}=resultname;
   bP_ROI_avg{result+1,1}=resultname;
   dP_ROI_avg{result+1,1}=resultname;

within_avg{result+1,2:length(suffixes)}=mean(within_mean);
between_avg{result+1,2:length(suffixes)}=mean(between_mean);
diff_avg{result+1,2:length(suffixes)}=mean(diff_mean);

   wP_avg{result+1,2:length(suffixes)}=mean(withinPriority_mean,1);
   bP_avg{result+1,2:length(suffixes)}=mean(betweenPriority_mean,1);
   dP_avg{result+1,2:length(suffixes)}=mean(diffPriority_mean,1);
   

withinROI_avg{result+1,2:length(suffixes)}=mean(withinROI_mean);
betweenROI_avg{result+1,2:length(suffixes)}=mean(betweenROI_mean);
diffROI_avg{result+1,2:length(suffixes)}=mean(diffROI_mean);

   wP_ROI_avg{result+1,2:length(suffixes)}=mean(withinROI_priority_mean,1);
   bP_ROI_avg{result+1,2:length(suffixes)}=mean(betweenROI_priority_mean,1);
   dP_ROI_avg{result+1,2:length(suffixes)}=mean(diffROI_priority_mean,1);
   
   %% also average together standard error of mean from different discriminations
   
within_ste{1,1}='resultname'; within_ste{1,2}=suffixes;
between_ste{1,1}='resultname'; between_ste{1,2}=suffixes;
diff_ste{1,1}='resultname'; diff_ste{1,2}=suffixes;

withinROI_ste{1,1}='resultname'; withinROI_ste{1,2}=roilist';
betweenROI_ste{1,1}='resultname'; betweenROI_ste{1,2}=roilist';
diffROI_ste{1,1}='resultname'; diffROI_ste{1,2}=roilist';

wP_ste{1,1}='resultname'; wP_ste{1,2}=suffixes;
bP_ste{1,1}='resultname'; bP_ste{1,2}=suffixes;
dP_ste{1,1}='resultname'; dP_ste{1,2}=suffixes;

wP_ROI_ste{1,1}='resultname'; wP_ROI_ste{1,2}=roilist';
bP_ROI_ste{1,1}='resultname'; bP_ROI_ste{1,2}=roilist';
dP_ROI_ste{1,1}='resultname'; dP_ROI_ste{1,2}=roilist';

within_ste{result+1,1}=resultname;
between_ste{result+1,1}=resultname;
diff_ste{result+1,1}=resultname;

   wP_ste{result+1,1}=resultname;
   bP_ste{result+1,1}=resultname;
   dP_ste{result+1,1}=resultname;
   
withinROI_ste{result+1,1}=resultname;
betweenROI_ste{result+1,1}=resultname;
diffROI_ste{result+1,1}=resultname;

   wP_ROI_ste{result+1,1}=resultname;
   bP_ROI_ste{result+1,1}=resultname;
   dP_ROI_ste{result+1,1}=resultname;
   
within_ste{result+1,2:length(suffixes)}=mean(withinste);
between_ste{result+1,2:length(suffixes)}=mean(betweenste);
diff_ste{result+1,2:length(suffixes)}=mean(diffste);

   wP_ste{result+1,2:length(suffixes)}=mean(withinPriority_ste,1);
   bP_ste{result+1,2:length(suffixes)}=mean(betweenPriority_ste,1);
   dP_ste{result+1,2:length(suffixes)}=mean(diffPriority_ste,1);
   
   
withinROI_ste{result+1,2:length(suffixes)}=mean(withinROIste);
betweenROI_ste{result+1,2:length(suffixes)}=mean(betweenROIste);
diffROI_ste{result+1,2:length(suffixes)}=mean(diffROIste);

   wP_ROI_ste{result+1,2:length(suffixes)}=mean(withinROI_priority_ste,1);
   bP_ROI_ste{result+1,2:length(suffixes)}=mean(betweenROI_priority_ste,1);
   dP_ROI_ste{result+1,2:length(suffixes)}=mean(diffROI_priority_ste,1);
 
   
 cd ..
end


   savefile=['MVPA_comparisons_' ROItypes{rt} '_' key];
   cd(mvpasummarydir)
  save(savefile, 'within_avg', 'between_avg', 'diff_avg', 'wP_avg', 'bP_avg', 'dP_avg', 'withinROI_avg', 'betweenROI_avg', 'diffROI_avg', 'wP_ROI_avg', 'bP_ROI_avg', 'dP_ROI_avg', 'within_ste', 'between_ste', 'diff_ste', 'wP_ste', 'bP_ste', 'dP_ste', 'withinROI_ste', 'betweenROI_ste', 'diffROI_ste', 'wP_ROI_ste', 'bP_ROI_ste', 'dP_ROI_ste')



for a=1:numAssessments
    
%summary plot
 width=24;
 height=12;
hFig = figure(1);
set(hFig,'units','inches')
set(hFig, 'Position', [10 10 width height])
    
    assessment=assessments{a};
    pA=priorities{a};
   evenvector=[];
   oddvector=[];
   evenvector_ste=[];
   oddvector_ste=[];
    vector=eval([assessments{a} '(2:end,2)']);
    vector_ste=eval([assessments_ste{a} '(2:end,2)']);
    evenvectorP=[];
   oddvectorP=[];
   evenvector_steP=[];
   oddvector_steP=[];
    vectorP=eval([priorities{a} '(2:end,2)']);
    vectorP_ste=eval([priorities_ste{a} '(2:end,2)']);
    
    for v=1:length(vector)
        evenvector=[evenvector; vector{v}(2)];
        oddvector=[oddvector; vector{v}(1)];
        evenvector_ste=[evenvector_ste; vector_ste{v}(2)];
        oddvector_ste=[oddvector_ste; vector_ste{v}(1)];
        evenvectorP=[evenvectorP; vectorP{v}(2)];
        oddvectorP=[oddvectorP; vectorP{v}(1)];
        evenvector_steP=[evenvector_steP; vectorP_ste{v}(2)];
        oddvector_steP=[oddvector_steP; vectorP_ste{v}(1)];
    end

 subplot(2,2,1);barwitherr(oddvector_ste, oddvector);ylim(yrange{a}(1,:));ylabel('raw Z');title(assessment);set(gca,'XTick',[])
 subplot(2,2,2);barwitherr(evenvector_ste, evenvector);ylim(yrange{a}(2,:));ylabel('raw Z');title(assessment);set(gca,'XTick',[])
 subplot(2,2,3);barwitherr(oddvector_steP, oddvectorP);ylim(yrange{a}(1,:));ylabel('raw Z');title(pA);set(gca,'XTick',[])
 subplot(2,2,4);barwitherr(evenvector_steP, evenvectorP);ylim(yrange{a}(2,:));ylabel('raw Z');title(pA);set(gca,'XTick',[])

set(gca,'XTick',1:numresults)
xlabel(['results folders ' ROItypes{rt}])
p=gcf;
saveas(p, ['summary_stats_' ROItypes{rt} '_' key '_' assessment]);

  prnstr = ['print -dpsc2 -painters -append ',['summary_stats_' ROItypes{rt} '_' key '.ps']];
               eval(prnstr);

hold off
clear gcf
close (1)

end

hFig = figure(1);
set(hFig,'units','inches')
text(0,0,resultlegend)
p=gcf;
saveas(p, ['summary_stats_' ROItypes{rt} '_' key '_legend' ]);

prnstr = ['print -dpsc2 -painters -append ',['summary_stats_' ROItypes{rt} '_' key '.ps']];
               eval(prnstr);
hold off
clear gcf
close (1)





assessmentsROI={'withinROI_avg', 'betweenROI_avg', 'diffROI_avg'};
prioritiesROI={'wP_ROI_avg', 'bP_ROI_avg', 'dP_ROI_avg'};
assessmentsROI_ste={'withinROI_ste', 'betweenROI_ste', 'diffROI_ste'}; 
prioritiesROI_ste={'wP_ROI_ste', 'bP_ROI_ste', 'dP_ROI_ste'};

for a=1:numAssessments
    
    %roi plot
 width=24;
 height=12;
hFig = figure(1);
set(hFig,'units','inches')
set(hFig, 'Position', [10 10 width height])
    
    assessment=assessmentsROI{a};
    pA=prioritiesROI{a};

for currresult=1:numresults

    resultvector=eval([assessmentsROI{a} '{' num2str(currresult+1) ',2}']);
    resultvector_ste=eval([assessmentsROI_ste{a} '{' num2str(currresult+1) ',2}']);

    %resultvector=cell2mat(wP_ROI_avg(currresult+1,2));
    indices=1:numROIs*2;
    odds=find(mod(indices,2));
    evens=find(~mod(indices,2));
    evenvector=resultvector(evens)';
    oddvector=resultvector(odds)';
    evenvectorste=resultvector_ste(evens)';
    oddvectorste=resultvector_ste(odds)';
    
   
subplot(numresults,2,(currresult*2-1));barwitherr(oddvectorste', oddvector');ylim(yrange{a}(1,:));set(gca,'XTick',[])
subplot(numresults,2,(currresult*2));barwitherr(evenvectorste', evenvector');ylim(yrange{a}(2,:));set(gca,'XTick',[])

mtit(assessment);

% add to print for the priority values (won't be able to fit these so make
% second fig??

end

p=gcf;
saveas(p, ['ROI_stats_' ROItypes{rt} '_' key '_' assessment ]);

  prnstr = ['print -dpsc2 -painters -append ',['ROI_stats_' ROItypes{rt} '_' key '.ps']];
               eval(prnstr);

hold off
clear gcf
close (1)

end

hFig = figure(1);
set(hFig,'units','inches')
text(0,0,roilist)
p=gcf;
saveas(p, ['ROI_stats_' ROItypes{rt} '_' key '_legend' ]);

prnstr = ['print -dpsc2 -painters -append ',['ROI_stats_' ROItypes{rt} '_' key '.ps']];
               eval(prnstr);
hold off
clear gcf
close (1)


for a=1:numAssessments
    
        %roi priority plot
 width=24;
 height=12;
hFig = figure(1);
set(hFig,'units','inches')
set(hFig, 'Position', [10 10 width height])
    
    assessment=assessmentsROI{a};
    pA=prioritiesROI{a};

for currresult=1:numresults

    resultvectorP=eval([prioritiesROI{a} '{' num2str(currresult+1) ',2}']);
    resultvectorP_ste=eval([prioritiesROI_ste{a} '{' num2str(currresult+1) ',2}']);

    %resultvector=cell2mat(wP_ROI_avg(currresult+1,2));
    indices=1:numROIs*2;
    odds=find(mod(indices,2));
    evens=find(~mod(indices,2));
    evenvectorP=resultvectorP(evens)';
    oddvectorP=resultvectorP(odds)';
    evenvectorsteP=resultvectorP_ste(evens)';
    oddvectorsteP=resultvectorP_ste(odds)';
    
   
subplot(numresults,2,(currresult*2-1));barwitherr(oddvectorsteP', oddvectorP');ylim(yrange{a}(1,:));set(gca,'XTick',[])
subplot(numresults,2,(currresult*2));barwitherr(evenvectorsteP', evenvectorP');ylim(yrange{a}(2,:));set(gca,'XTick',[])

mtit(assessment);
% add to print for the priority values (won't be able to fit these so make
% second fig??

end

p=gcf;
saveas(p, ['ROI_stats_' ROItypes{rt} '_' key '_' assessment 'priority']);

  prnstr = ['print -dpsc2 -painters -append ',['ROI_stats_' ROItypes{rt} '_' key 'priority.ps']];
               eval(prnstr);

hold off
clear gcf
close (1)

end

hFig = figure(1);
set(hFig,'units','inches')
text(0,0,roilist)
p=gcf;
saveas(p, ['ROI_stats_' ROItypes{rt} '_' key '_legend' ]);

prnstr = ['print -dpsc2 -painters -append ',['ROI_stats_' ROItypes{rt} '_' key '.ps']];
               eval(prnstr);
hold off
clear gcf
close (1)

cd ..

   
end
end


