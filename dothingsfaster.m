function dothingsfaster(subjects)
%e.g. dothingsfaster(makeIDs('EIB', [1:13, 15:20]), '*psc*')
%% fill this part in with what you want to do
%indDirectory='mvpa_ptb/bpbp0_swrf_binary_wart_featureselect_averaged/'
%'results/EmoBioLoc_with_art_reg_results_normed'

%%

rootdir=['/mindhive/saxelab2/EIB/'];


numsubj=length(subjects)
for n=1:numsubj
   subjectID=subjects{n}
    %subjdir=[rootdir indDirectory]
    subjdir=[rootdir subjectID '/mvpa_ptb/FINALsearchlight_crossruns_newROIS_EIB_main_1to8_libsvm_swrf_bin_tpoint_hrfshift4_fsactivity_ranktop80_avgd_zsc_hpfilt_detr_noglnorm_costfix/'];
    %subjdir=[rootdir subjectID '/'];
    cd(subjdir);
    listdirs=dir([subjdir, subjectID '*.*'])
%          for x=1:length(listdirs)
%          listdirs(x).name 
%          delete(listdirs(x).name )
%          end
    %rmdir('null_permutationcrossruns_newROIS_EIB_main_1to8_libsvm_swrf_bin_wart_hrfshift4_fsactivity_ranktop80_avgd_zsc_hpfilt_detr_noglnorm_costfix', 's')
    %rmdir('autoROI', 's')
    %delete(key)
    %rmdir(key, 's')
%     for x=1:length(listdirs)
%     listdirs(x).name 
%     cd(listdirs(x).name)
%     delete('*selectors_noavg.fig')
%     delete('*selectors_avg.fig')
%     delete('*regressors_noavg.fig')
%     %delete('*regressors_avg.fig')
%     delete('*voxelsINmasks.fig')
%     cd ..
%    end
    %rmdir([subjdir, 'crossruns_newROIS_EIB_main_1to8_libsvm_swrf_bin_wart_hrfshift4_fsactivity_ranktop80_avgd_zsc_hpfilt_detr_noglnorm_costfix/'], 's')
    %movefile([subjdir, 'newROIS_EIB_main_1to8_libsvm_swrf_bin_wart_hrfshift4_fsactivity_ranktop80_avgd_zsc_hpfilt_detr_noglnorm_costfix/'], [subjdir, 'old_newROIS_EIB_main_1to8_libsvm_swrf_bin_wart_hrfshift4_fsactivity_ranktop80_avgd_zsc_hpfilt_detr_noglnorm_costfix/'])

   % end
    %end
    %%mkdir()

%rename folder (by moving)
%     oldname=[pwd '/' key]
%     if exist(oldname)
%     newname=[pwd '/' key '_noglobal']
%     movefile(oldname, newname);
%     end

    end
end
   


