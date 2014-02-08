function seeit(mvpa_stats, discrimination)
rois=fieldnames(mvpa_stats)
numrois=length(rois)

for r=1:numrois
    answer=eval(['mvpa_stats.' rois{r} '.' discrimination]);
    disp([rois{r} '--' discrimination ': ' answer(end-8:end)]);
end
end