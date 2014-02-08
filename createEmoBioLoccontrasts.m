function createEmoBioLoccontrasts(study, subjects, resultfolder)
%e.g. createEIBcontrasts('EIB', {'SAX_EIB_01','SAX_EIB_02','SAX_EIB_03','SAX_EIB_04','SAX_EIB_05','SAX_EIB_06','SAX_EIB_07','SAX_EIB_08','SAX_EIB_09', 'SAX_EIB_10', 'SAX_EIB_11', 'SAX_EIB_12', 'SAX_EIB_13'}, 'EIB_main_results_normed')


%NEW CREATE EIB CONTRASTS (VSR (vs. rest) = raw betas, VSO (vs. opposite) = contrast of two conditions to be discriminated

saxelab_create_contrast_amy('ODD_pos_all_vsr',{'Sn(1) mh', 'Sn(1) fh', 'Sn(1) sh', 'Sn(1) nh', 'Sn(3) mh', 'Sn(3) fh', 'Sn(3) sh', 'Sn(3) nh','Sn(5) mh', 'Sn(5) fh', 'Sn(5) sh', 'Sn(5) nh','Sn(7) mh', 'Sn(7) fh', 'Sn(7) sh', 'Sn(7) nh'},[1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1], 'EIB', subjects, resultfolder, 'nostop');

saxelab_create_contrast_amy('EVEN_pos_all_vsr',{'Sn(2) mh', 'Sn(2) fh', 'Sn(2) sh', 'Sn(2) nh', 'Sn(4) mh', 'Sn(4) fh', 'Sn(4) sh', 'Sn(4) nh','Sn(6) mh', 'Sn(6) fh', 'Sn(6) sh', 'Sn(6) nh','Sn(8) mh', 'Sn(8) fh', 'Sn(8) sh', 'Sn(8) nh'},[1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1], 'EIB', subjects, resultfolder, 'nostop');

saxelab_create_contrast_amy('ODD_neg_all_vsr',{'Sn(1) mu', 'Sn(1) fu', 'Sn(1) su', 'Sn(1) nu', 'Sn(3) mu', 'Sn(3) fu', 'Sn(3) su', 'Sn(3) nu','Sn(5) mu', 'Sn(5) fu', 'Sn(5) su', 'Sn(5) nu','Sn(7) mu', 'Sn(7) fu', 'Sn(7) su', 'Sn(7) nu'},[1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1], 'EIB', subjects, resultfolder, 'nostop');

saxelab_create_contrast_amy('EVEN_neg_all_vsr',{'Sn(2) mu', 'Sn(2) fu', 'Sn(2) su', 'Sn(2) nu', 'Sn(4) mu', 'Sn(4) fu', 'Sn(4) su', 'Sn(4) nu','Sn(6) mu', 'Sn(6) fu', 'Sn(6) su', 'Sn(6) nu','Sn(8) mu', 'Sn(8) fu', 'Sn(8) su', 'Sn(8) nu'},[1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1], 'EIB', subjects, resultfolder, 'nostop');



saxelab_create_contrast_amy('ODD_pos_c_vsr',{'Sn(1) sh', 'Sn(1) nh', 'Sn(3) sh', 'Sn(3) nh', 'Sn(5) sh', 'Sn(5) nh','Sn(7) sh', 'Sn(7) nh'},[1 1 1 1 1 1 1 1 ], 'EIB', subjects, resultfolder, 'nostop');

saxelab_create_contrast_amy('EVEN_pos_c_vsr',{'Sn(2) sh', 'Sn(2) nh', 'Sn(4) sh', 'Sn(4) nh', 'Sn(6) sh', 'Sn(6) nh','Sn(8) sh', 'Sn(8) nh'},[1 1 1 1 1 1 1 1 ], 'EIB', subjects, resultfolder, 'nostop');

saxelab_create_contrast_amy('ODD_neg_c_vsr',{'Sn(1) su', 'Sn(1) nu', 'Sn(3) su', 'Sn(3) nu', 'Sn(5) su', 'Sn(5) nu','Sn(7) su', 'Sn(7) nu'},[1 1 1 1 1 1 1 1 ], 'EIB', subjects, resultfolder, 'nostop');

saxelab_create_contrast_amy('EVEN_neg_c_vsr',{'Sn(2) su', 'Sn(2) nu', 'Sn(4) su', 'Sn(4) nu', 'Sn(6) su', 'Sn(6) nu','Sn(8) su', 'Sn(8) nu'},[1 1 1 1 1 1 1 1 ], 'EIB', subjects, resultfolder, 'nostop');



saxelab_create_contrast_amy('ODD_pos_p_vsr',{'Sn(1) mh', 'Sn(1) fh', 'Sn(3) mh', 'Sn(3) fh', 'Sn(5) mh', 'Sn(5) fh','Sn(7) mh', 'Sn(7) fh'},[1 1 1 1 1 1 1 1 ], 'EIB', subjects, resultfolder, 'nostop');

saxelab_create_contrast_amy('EVEN_pos_p_vsr',{'Sn(2) mh', 'Sn(2) fh', 'Sn(4) mh', 'Sn(4) fh', 'Sn(6) mh', 'Sn(6) fh','Sn(8) mh', 'Sn(8) fh'},[1 1 1 1 1 1 1 1 ], 'EIB', subjects, resultfolder, 'nostop');

saxelab_create_contrast_amy('ODD_neg_p_vsr',{'Sn(1) mu', 'Sn(1) fu', 'Sn(3) mu', 'Sn(3) fu', 'Sn(5) mu', 'Sn(5) fu','Sn(7) mu', 'Sn(7) fu'},[1 1 1 1 1 1 1 1 ], 'EIB', subjects, resultfolder, 'nostop');

saxelab_create_contrast_amy('EVEN_neg_p_vsr',{'Sn(2) mu', 'Sn(2) fu', 'Sn(4) mu', 'Sn(4) fu', 'Sn(6) mu', 'Sn(6) fu','Sn(8) mu', 'Sn(8) fu'},[1 1 1 1 1 1 1 1 ], 'EIB', subjects, resultfolder, 'nostop');



saxelab_create_contrast_amy('ODD_possocial_vsr',{'Sn(1) sh', 'Sn(3) sh', 'Sn(5) sh','Sn(7) sh'},[1 1 1 1 ], 'EIB', subjects, resultfolder, 'nostop');

saxelab_create_contrast_amy('EVEN_possocial_vsr',{'Sn(2) sh', 'Sn(4) sh', 'Sn(6) sh', 'Sn(8) sh'},[1 1 1 1 ], 'EIB', subjects, resultfolder, 'nostop');

saxelab_create_contrast_amy('ODD_negsocial_vsr',{'Sn(1) su', 'Sn(3) su', 'Sn(5) su', 'Sn(7) su'},[1 1 1 1 ], 'EIB', subjects, resultfolder, 'nostop');

saxelab_create_contrast_amy('EVEN_negsocial_vsr',{'Sn(2) su', 'Sn(4) su', 'Sn(6) su', 'Sn(8) su'},[1 1 1 1 ], 'EIB', subjects, resultfolder, 'nostop');


saxelab_create_contrast_amy('ODD_posnonsoc_vsr',{'Sn(1) nh', 'Sn(3) nh', 'Sn(5) nh','Sn(7) nh'},[1 1 1 1 ], 'EIB', subjects, resultfolder, 'nostop');

saxelab_create_contrast_amy('EVEN_posnonsoc_vsr',{'Sn(2) nh', 'Sn(4) nh', 'Sn(6) nh', 'Sn(8) nh'},[1 1 1 1 ], 'EIB', subjects, resultfolder, 'nostop');

saxelab_create_contrast_amy('ODD_negnonsoc_vsr',{'Sn(1) nu', 'Sn(3) nu', 'Sn(5) nu', 'Sn(7) nu'},[1 1 1 1 ], 'EIB', subjects, resultfolder, 'nostop');

saxelab_create_contrast_amy('EVEN_negnonsoc_vsr',{'Sn(2) nu', 'Sn(4) nu', 'Sn(6) nu', 'Sn(8) nu'},[1 1 1 1 ], 'EIB', subjects, resultfolder, 'nostop');


saxelab_create_contrast_amy('ODD_posmale_vsr',{'Sn(1) mh', 'Sn(3) mh', 'Sn(5) mh','Sn(7) mh'},[1 1 1 1 ], 'EIB', subjects, resultfolder, 'nostop');

saxelab_create_contrast_amy('EVEN_posmale_vsr',{'Sn(2) mh', 'Sn(4) mh', 'Sn(6) mh', 'Sn(8) mh'},[1 1 1 1 ], 'EIB', subjects, resultfolder, 'nostop');

saxelab_create_contrast_amy('ODD_negmale_vsr',{'Sn(1) mu', 'Sn(3) mu', 'Sn(5) mu', 'Sn(7) mu'},[1 1 1 1 ], 'EIB', subjects, resultfolder, 'nostop');

saxelab_create_contrast_amy('EVEN_negmale_vsr',{'Sn(2) mu', 'Sn(4) mu', 'Sn(6) mu', 'Sn(8) mu'},[1 1 1 1 ], 'EIB', subjects, resultfolder, 'nostop');


saxelab_create_contrast_amy('ODD_posfema_vsr',{'Sn(1) fh', 'Sn(3) fh', 'Sn(5) fh','Sn(7) fh'},[1 1 1 1 ], 'EIB', subjects, resultfolder, 'nostop');

saxelab_create_contrast_amy('EVEN_posfema_vsr',{'Sn(2) fh', 'Sn(4) fh', 'Sn(6) fh', 'Sn(8) fh'},[1 1 1 1 ], 'EIB', subjects, resultfolder, 'nostop');

saxelab_create_contrast_amy('ODD_negfema_vsr',{'Sn(1) fu', 'Sn(3) fu', 'Sn(5) fu', 'Sn(7) fu'},[1 1 1 1 ], 'EIB', subjects, resultfolder, 'nostop');

saxelab_create_contrast_amy('EVEN_negfema_vsr',{'Sn(2) fu', 'Sn(4) fu', 'Sn(6) fu', 'Sn(8) fu'},[1 1 1 1 ], 'EIB', subjects, resultfolder, 'nostop');


%%These aren't actually divided by even and odd. even=persons,
%%odd=contexts.
%pos_p=EVEN
saxelab_create_contrast_amy('EVEN_fake_pos_vsr',{'Sn(1) mh', 'Sn(1) fh', 'Sn(3) mh', 'Sn(3) fh', 'Sn(5) mh', 'Sn(5) fh','Sn(7) mh', 'Sn(7) fh', 'Sn(2) mh', 'Sn(2) fh', 'Sn(4) mh', 'Sn(4) fh', 'Sn(6) mh', 'Sn(6) fh','Sn(8) mh', 'Sn(8) fh'},[1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 ], 'EIB', subjects, resultfolder, 'nostop');

%neg_p=EVEN
saxelab_create_contrast_amy('EVEN_fake_neg_vsr',{'Sn(1) mu', 'Sn(1) fu', 'Sn(3) mu', 'Sn(3) fu', 'Sn(5) mu', 'Sn(5) fu','Sn(7) mu', 'Sn(7) fu', 'Sn(2) mu', 'Sn(2) fu', 'Sn(4) mu', 'Sn(4) fu', 'Sn(6) mu', 'Sn(6) fu','Sn(8) mu', 'Sn(8) fu'},[1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 ], 'EIB', subjects, resultfolder, 'nostop');

%pos_c=ODD
saxelab_create_contrast_amy('ODD_fake_pos_vsr',{'Sn(1) sh', 'Sn(1) nh', 'Sn(3) sh', 'Sn(3) nh', 'Sn(5) sh', 'Sn(5) nh','Sn(7) sh', 'Sn(7) nh', 'Sn(2) sh', 'Sn(2) nh', 'Sn(4) sh', 'Sn(4) nh', 'Sn(6) sh', 'Sn(6) nh','Sn(8) sh', 'Sn(8) nh'},[1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 ], 'EIB', subjects, resultfolder, 'nostop');

%neg_c=ODD
saxelab_create_contrast_amy('ODD_fake_neg_vsr',{'Sn(1) su', 'Sn(1) nu', 'Sn(3) su', 'Sn(3) nu', 'Sn(5) su', 'Sn(5) nu','Sn(7) su', 'Sn(7) nu', 'Sn(2) su', 'Sn(2) nu', 'Sn(4) su', 'Sn(4) nu', 'Sn(6) su', 'Sn(6) nu','Sn(8) su', 'Sn(8) nu'},[1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 ], 'EIB', subjects, resultfolder, 'nostop');

%% similarly, odd=social, even=nonsocial
saxelab_create_contrast_amy('ODD_fake_poscontext_vsr',{'Sn(1) sh', 'Sn(3) sh', 'Sn(5) sh','Sn(7) sh', 'Sn(2) sh', 'Sn(4) sh', 'Sn(6) sh','Sn(8) sh'},[1 1 1 1 1 1 1 1 ], 'EIB', subjects, resultfolder, 'nostop');

saxelab_create_contrast_amy('EVEN_fake_poscontext_vsr',{'Sn(2) nh', 'Sn(4) nh', 'Sn(6) nh', 'Sn(8) nh', 'Sn(1) nh', 'Sn(3) nh', 'Sn(5) nh', 'Sn(7) nh'},[1 1 1 1 1 1 1 1 ], 'EIB', subjects, resultfolder, 'nostop');

saxelab_create_contrast_amy('ODD_fake_negcontext_vsr',{'Sn(1) su', 'Sn(3) su', 'Sn(5) su', 'Sn(7) su', 'Sn(2) su', 'Sn(4) su', 'Sn(6) su', 'Sn(8) su'},[1 1 1 1 1 1 1 1 ], 'EIB', subjects, resultfolder, 'nostop');

saxelab_create_contrast_amy('EVEN_fake_negcontext_vsr',{'Sn(2) nu', 'Sn(4) nu', 'Sn(6) nu', 'Sn(8) nu', 'Sn(1) nu', 'Sn(3) nu', 'Sn(5) nu', 'Sn(7) nu'},[1 1 1 1 1 1 1 1 ], 'EIB', subjects, resultfolder, 'nostop');

%%%
%% similarly, odd=male, even=female
saxelab_create_contrast_amy('ODD_fake_posperson_vsr',{'Sn(1) mh', 'Sn(3) mh', 'Sn(5) mh','Sn(7) mh', 'Sn(2) mh', 'Sn(4) mh', 'Sn(6) mh','Sn(8) mh'},[1 1 1 1 1 1 1 1 ], 'EIB', subjects, resultfolder, 'nostop');

saxelab_create_contrast_amy('EVEN_fake_posperson_vsr',{'Sn(2) fh', 'Sn(4) fh', 'Sn(6) fh', 'Sn(8) fh', 'Sn(1) fh', 'Sn(3) fh', 'Sn(5) fh', 'Sn(7) fh'},[1 1 1 1 1 1 1 1 ], 'EIB', subjects, resultfolder, 'nostop');

saxelab_create_contrast_amy('ODD_fake_negperson_vsr',{'Sn(1) mu', 'Sn(3) mu', 'Sn(5) mu', 'Sn(7) mu', 'Sn(2) mu', 'Sn(4) mu', 'Sn(6) mu', 'Sn(8) mu'},[1 1 1 1 1 1 1 1 ], 'EIB', subjects, resultfolder, 'nostop');

saxelab_create_contrast_amy('EVEN_fake_negperson_vsr',{'Sn(2) fu', 'Sn(4) fu', 'Sn(6) fu', 'Sn(8) fu', 'Sn(1) fu', 'Sn(3) fu', 'Sn(5) fu', 'Sn(7) fu'},[1 1 1 1 1 1 1 1 ], 'EIB', subjects, resultfolder, 'nostop');

%%%

%%%other less controlled contrasts

saxelab_create_contrast_amy('ODD_nonsoc_vsr',{'Sn(1) nh', 'Sn(1) nu', 'Sn(3) nh', 'Sn(3) nu', 'Sn(5) nh', 'Sn(5) nu','Sn(7) nh', 'Sn(7) nu'},[1 1 1 1 1 1 1 1 ], 'EIB', subjects, resultfolder, 'nostop');

saxelab_create_contrast_amy('EVEN_nonsoc_vsr',{'Sn(2) nh', 'Sn(2) nu', 'Sn(4) nh', 'Sn(4) nu', 'Sn(6) nh', 'Sn(6) nu','Sn(8) nh', 'Sn(8) nu'},[1 1 1 1 1 1 1 1 ], 'EIB', subjects, resultfolder, 'nostop');

saxelab_create_contrast_amy('ODD_social_vsr',{'Sn(1) sh', 'Sn(1) su', 'Sn(3) sh', 'Sn(3) su', 'Sn(5) sh', 'Sn(5) su','Sn(7) sh', 'Sn(7) su'},[1 1 1 1 1 1 1 1 ], 'EIB', subjects, resultfolder, 'nostop');

saxelab_create_contrast_amy('EVEN_social_vsr',{'Sn(2) sh', 'Sn(2) su', 'Sn(4) sh', 'Sn(4) su', 'Sn(6) sh', 'Sn(6) su','Sn(8) sh', 'Sn(8) su'},[1 1 1 1 1 1 1 1 ], 'EIB', subjects, resultfolder, 'nostop');


saxelab_create_contrast_amy('ODD_persons_vsr',{'Sn(1) mh', 'Sn(1) mu', 'Sn(3) mh', 'Sn(3) mu', 'Sn(5) mh', 'Sn(5) mu','Sn(7) mh', 'Sn(7) mu', 'Sn(1) fh', 'Sn(1) fu', 'Sn(3) fh', 'Sn(3) fu', 'Sn(5) fh', 'Sn(5) fu','Sn(7) fh', 'Sn(7) fu'},[1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 ], 'EIB', subjects, resultfolder, 'nostop');

saxelab_create_contrast_amy('EVEN_persons_vsr',{'Sn(2) mh', 'Sn(2) mu', 'Sn(4) mh', 'Sn(4) mu', 'Sn(6) mh', 'Sn(6) mu','Sn(8) mh', 'Sn(8) mu', 'Sn(2) fh', 'Sn(2) fu', 'Sn(4) fh', 'Sn(4) fu', 'Sn(6) fh', 'Sn(6) fu','Sn(8) fh', 'Sn(8) fu'}, [1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1], 'EIB', subjects, resultfolder, 'nostop');

saxelab_create_contrast_amy('ODD_cartoon_vsr',{'Sn(1) nh', 'Sn(1) nu', 'Sn(3) nh', 'Sn(3) nu', 'Sn(5) nh', 'Sn(5) nu','Sn(7) nh', 'Sn(7) nu','Sn(1) sh', 'Sn(1) su', 'Sn(3) sh', 'Sn(3) su', 'Sn(5) sh', 'Sn(5) su','Sn(7) sh', 'Sn(7) su'},[1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 ], 'EIB', subjects, resultfolder, 'nostop');

saxelab_create_contrast_amy('EVEN_cartoon_vsr',{'Sn(2) nh', 'Sn(2) nu', 'Sn(4) nh', 'Sn(4) nu', 'Sn(6) nh', 'Sn(6) nu','Sn(8) nh', 'Sn(8) nu','Sn(2) sh', 'Sn(2) su', 'Sn(4) sh', 'Sn(4) su', 'Sn(6) sh', 'Sn(6) su','Sn(8) sh', 'Sn(8) su'},[1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 ], 'EIB', subjects, resultfolder, 'nostop');


saxelab_create_contrast_amy('ODD_gendermale_vsr',{'Sn(1) mh', 'Sn(1) mu', 'Sn(3) mh', 'Sn(3) mu', 'Sn(5) mh', 'Sn(5) mu','Sn(7) mh', 'Sn(7) mu'},[1 1 1 1 1 1 1 1], 'EIB', subjects, resultfolder, 'nostop');

saxelab_create_contrast_amy('EVEN_gendermale_vsr',{'Sn(2) mh', 'Sn(2) mu', 'Sn(4) mh', 'Sn(4) mu', 'Sn(6) mh', 'Sn(6) mu','Sn(8) mh', 'Sn(8) mu'}, [1 1 1 1 1 1 1 1], 'EIB', subjects, resultfolder, 'nostop');

saxelab_create_contrast_amy('ODD_genderfema_vsr',{'Sn(1) fh', 'Sn(1) fu', 'Sn(3) fh', 'Sn(3) fu', 'Sn(5) fh', 'Sn(5) fu','Sn(7) fh', 'Sn(7) fu'},[1 1 1 1 1 1 1 1], 'EIB', subjects, resultfolder, 'nostop');

saxelab_create_contrast_amy('EVEN_genderfema_vsr',{'Sn(2) fh', 'Sn(2) fu', 'Sn(4) fh', 'Sn(4) fu', 'Sn(6) fh', 'Sn(6) fu','Sn(8) fh', 'Sn(8) fu'}, [1 1 1 1 1 1 1 1], 'EIB', subjects, resultfolder, 'nostop');



%%%%%%%%%%%


saxelab_create_contrast_amy('ODD_pos_all_vso',{'Sn(1) mh', 'Sn(1) fh', 'Sn(1) sh', 'Sn(1) nh', 'Sn(3) mh', 'Sn(3) fh', 'Sn(3) sh', 'Sn(3) nh','Sn(5) mh', 'Sn(5) fh', 'Sn(5) sh', 'Sn(5) nh','Sn(7) mh', 'Sn(7) fh', 'Sn(7) sh', 'Sn(7) nh', 'Sn(1) mu', 'Sn(1) fu', 'Sn(1) su', 'Sn(1) nu', 'Sn(3) mu', 'Sn(3) fu', 'Sn(3) su', 'Sn(3) nu','Sn(5) mu', 'Sn(5) fu', 'Sn(5) su', 'Sn(5) nu','Sn(7) mu', 'Sn(7) fu', 'Sn(7) su', 'Sn(7) nu'},[1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1], 'EIB', subjects, resultfolder, 'nostop');

saxelab_create_contrast_amy('EVEN_pos_all_vso',{'Sn(2) mh', 'Sn(2) fh', 'Sn(2) sh', 'Sn(2) nh', 'Sn(4) mh', 'Sn(4) fh', 'Sn(4) sh', 'Sn(4) nh','Sn(6) mh', 'Sn(6) fh', 'Sn(6) sh', 'Sn(6) nh','Sn(8) mh', 'Sn(8) fh', 'Sn(8) sh', 'Sn(8) nh', 'Sn(2) mu', 'Sn(2) fu', 'Sn(2) su', 'Sn(2) nu', 'Sn(4) mu', 'Sn(4) fu', 'Sn(4) su', 'Sn(4) nu','Sn(6) mu', 'Sn(6) fu', 'Sn(6) su', 'Sn(6) nu','Sn(8) mu', 'Sn(8) fu', 'Sn(8) su', 'Sn(8) nu'},[1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1], 'EIB', subjects, resultfolder, 'nostop');

saxelab_create_contrast_amy('ODD_neg_all_vso',{'Sn(1) mu', 'Sn(1) fu', 'Sn(1) su', 'Sn(1) nu', 'Sn(3) mu', 'Sn(3) fu', 'Sn(3) su', 'Sn(3) nu','Sn(5) mu', 'Sn(5) fu', 'Sn(5) su', 'Sn(5) nu','Sn(7) mu', 'Sn(7) fu', 'Sn(7) su', 'Sn(7) nu', 'Sn(1) mh', 'Sn(1) fh', 'Sn(1) sh', 'Sn(1) nh', 'Sn(3) mh', 'Sn(3) fh', 'Sn(3) sh', 'Sn(3) nh','Sn(5) mh', 'Sn(5) fh', 'Sn(5) sh', 'Sn(5) nh','Sn(7) mh', 'Sn(7) fh', 'Sn(7) sh', 'Sn(7) nh'},[1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1], 'EIB', subjects, resultfolder, 'nostop');

saxelab_create_contrast_amy('EVEN_neg_all_vso',{'Sn(2) mu', 'Sn(2) fu', 'Sn(2) su', 'Sn(2) nu', 'Sn(4) mu', 'Sn(4) fu', 'Sn(4) su', 'Sn(4) nu','Sn(6) mu', 'Sn(6) fu', 'Sn(6) su', 'Sn(6) nu','Sn(8) mu', 'Sn(8) fu', 'Sn(8) su', 'Sn(8) nu', 'Sn(2) mh', 'Sn(2) fh', 'Sn(2) sh', 'Sn(2) nh', 'Sn(4) mh', 'Sn(4) fh', 'Sn(4) sh', 'Sn(4) nh','Sn(6) mh', 'Sn(6) fh', 'Sn(6) sh', 'Sn(6) nh','Sn(8) mh', 'Sn(8) fh', 'Sn(8) sh', 'Sn(8) nh'},[1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1], 'EIB', subjects, resultfolder, 'nostop');



saxelab_create_contrast_amy('ODD_pos_c_vso',{'Sn(1) sh', 'Sn(1) nh', 'Sn(3) sh', 'Sn(3) nh', 'Sn(5) sh', 'Sn(5) nh','Sn(7) sh', 'Sn(7) nh', 'Sn(1) su', 'Sn(1) nu', 'Sn(3) su', 'Sn(3) nu', 'Sn(5) su', 'Sn(5) nu','Sn(7) su', 'Sn(7) nu'},[1 1 1 1 1 1 1 1 -1 -1 -1 -1 -1 -1 -1 -1], 'EIB', subjects, resultfolder, 'nostop');

saxelab_create_contrast_amy('EVEN_pos_c_vso',{'Sn(2) sh', 'Sn(2) nh', 'Sn(4) sh', 'Sn(4) nh', 'Sn(6) sh', 'Sn(6) nh','Sn(8) sh', 'Sn(8) nh', 'Sn(2) su', 'Sn(2) nu', 'Sn(4) su', 'Sn(4) nu', 'Sn(6) su', 'Sn(6) nu','Sn(8) su', 'Sn(8) nu'},[1 1 1 1 1 1 1 1 -1 -1 -1 -1 -1 -1 -1 -1], 'EIB', subjects, resultfolder, 'nostop');

saxelab_create_contrast_amy('ODD_neg_c_vso',{'Sn(1) su', 'Sn(1) nu', 'Sn(3) su', 'Sn(3) nu', 'Sn(5) su', 'Sn(5) nu','Sn(7) su', 'Sn(7) nu', 'Sn(1) sh', 'Sn(1) nh', 'Sn(3) sh', 'Sn(3) nh', 'Sn(5) sh', 'Sn(5) nh','Sn(7) sh', 'Sn(7) nh'},[1 1 1 1 1 1 1 1 -1 -1 -1 -1 -1 -1 -1 -1], 'EIB', subjects, resultfolder, 'nostop');

saxelab_create_contrast_amy('EVEN_neg_c_vso',{'Sn(2) su', 'Sn(2) nu', 'Sn(4) su', 'Sn(4) nu', 'Sn(6) su', 'Sn(6) nu','Sn(8) su', 'Sn(8) nu', 'Sn(2) sh', 'Sn(2) nh', 'Sn(4) sh', 'Sn(4) nh', 'Sn(6) sh', 'Sn(6) nh','Sn(8) sh', 'Sn(8) nh'},[1 1 1 1 1 1 1 1 -1 -1 -1 -1 -1 -1 -1 -1], 'EIB', subjects, resultfolder, 'nostop');



saxelab_create_contrast_amy('ODD_pos_p_vso',{'Sn(1) mh', 'Sn(1) fh', 'Sn(3) mh', 'Sn(3) fh', 'Sn(5) mh', 'Sn(5) fh','Sn(7) mh', 'Sn(7) fh', 'Sn(1) mu', 'Sn(1) fu', 'Sn(3) mu', 'Sn(3) fu', 'Sn(5) mu', 'Sn(5) fu','Sn(7) mu', 'Sn(7) fu'},[1 1 1 1 1 1 1 1 -1 -1 -1 -1 -1 -1 -1 -1], 'EIB', subjects, resultfolder, 'nostop');

saxelab_create_contrast_amy('EVEN_pos_p_vso',{'Sn(2) mh', 'Sn(2) fh', 'Sn(4) mh', 'Sn(4) fh', 'Sn(6) mh', 'Sn(6) fh','Sn(8) mh', 'Sn(8) fh', 'Sn(2) mu', 'Sn(2) fu', 'Sn(4) mu', 'Sn(4) fu', 'Sn(6) mu', 'Sn(6) fu','Sn(8) mu', 'Sn(8) fu'},[1 1 1 1 1 1 1 1 -1 -1 -1 -1 -1 -1 -1 -1], 'EIB', subjects, resultfolder, 'nostop');

saxelab_create_contrast_amy('ODD_neg_p_vso',{'Sn(1) mu', 'Sn(1) fu', 'Sn(3) mu', 'Sn(3) fu', 'Sn(5) mu', 'Sn(5) fu','Sn(7) mu', 'Sn(7) fu', 'Sn(1) mh', 'Sn(1) fh', 'Sn(3) mh', 'Sn(3) fh', 'Sn(5) mh', 'Sn(5) fh','Sn(7) mh', 'Sn(7) fh'},[1 1 1 1 1 1 1 1 -1 -1 -1 -1 -1 -1 -1 -1], 'EIB', subjects, resultfolder, 'nostop');

saxelab_create_contrast_amy('EVEN_neg_p_vso',{'Sn(2) mu', 'Sn(2) fu', 'Sn(4) mu', 'Sn(4) fu', 'Sn(6) mu', 'Sn(6) fu','Sn(8) mu', 'Sn(8) fu', 'Sn(2) mh', 'Sn(2) fh', 'Sn(4) mh', 'Sn(4) fh', 'Sn(6) mh', 'Sn(6) fh','Sn(8) mh', 'Sn(8) fh'},[1 1 1 1 1 1 1 1 -1 -1 -1 -1 -1 -1 -1 -1], 'EIB', subjects, resultfolder, 'nostop');



saxelab_create_contrast_amy('ODD_possocial_vso',{'Sn(1) sh', 'Sn(3) sh', 'Sn(5) sh','Sn(7) sh', 'Sn(1) su', 'Sn(3) su', 'Sn(5) su', 'Sn(7) su'},[1 1 1 1 -1 -1 -1 -1], 'EIB', subjects, resultfolder, 'nostop');

saxelab_create_contrast_amy('EVEN_possocial_vso',{'Sn(2) sh', 'Sn(4) sh', 'Sn(6) sh', 'Sn(8) sh', 'Sn(2) su', 'Sn(4) su', 'Sn(6) su', 'Sn(8) su'},[1 1 1 1 -1 -1 -1 -1], 'EIB', subjects, resultfolder, 'nostop');

saxelab_create_contrast_amy('ODD_negsocial_vso',{'Sn(1) su', 'Sn(3) su', 'Sn(5) su', 'Sn(7) su', 'Sn(1) sh', 'Sn(3) sh', 'Sn(5) sh','Sn(7) sh'},[1 1 1 1 -1 -1 -1 -1], 'EIB', subjects, resultfolder, 'nostop');

saxelab_create_contrast_amy('EVEN_negsocial_vso',{'Sn(2) su', 'Sn(4) su', 'Sn(6) su', 'Sn(8) su', 'Sn(2) sh', 'Sn(4) sh', 'Sn(6) sh', 'Sn(8) sh'},[1 1 1 1 -1 -1 -1 -1], 'EIB', subjects, resultfolder, 'nostop');


saxelab_create_contrast_amy('ODD_posnonsoc_vso',{'Sn(1) nh', 'Sn(3) nh', 'Sn(5) nh','Sn(7) nh', 'Sn(1) nu', 'Sn(3) nu', 'Sn(5) nu', 'Sn(7) nu'},[1 1 1 1 -1 -1 -1 -1], 'EIB', subjects, resultfolder, 'nostop');

saxelab_create_contrast_amy('EVEN_posnonsoc_vso',{'Sn(2) nh', 'Sn(4) nh', 'Sn(6) nh', 'Sn(8) nh', 'Sn(2) nu', 'Sn(4) nu', 'Sn(6) nu', 'Sn(8) nu'},[1 1 1 1 -1 -1 -1 -1], 'EIB', subjects, resultfolder, 'nostop');

saxelab_create_contrast_amy('ODD_negnonsoc_vso',{'Sn(1) nu', 'Sn(3) nu', 'Sn(5) nu', 'Sn(7) nu', 'Sn(1) nh', 'Sn(3) nh', 'Sn(5) nh','Sn(7) nh'},[1 1 1 1 -1 -1 -1 -1], 'EIB', subjects, resultfolder, 'nostop');

saxelab_create_contrast_amy('EVEN_negnonsoc_vso',{'Sn(2) nu', 'Sn(4) nu', 'Sn(6) nu', 'Sn(8) nu', 'Sn(2) nh', 'Sn(4) nh', 'Sn(6) nh', 'Sn(8) nh'},[1 1 1 1 -1 -1 -1 -1], 'EIB', subjects, resultfolder, 'nostop');


saxelab_create_contrast_amy('ODD_posmale_vso',{'Sn(1) mh', 'Sn(3) mh', 'Sn(5) mh', 'Sn(7) mh', 'Sn(1) mu', 'Sn(3) mu', 'Sn(5) mu', 'Sn(7) mu'},[1 1 1 1 -1 -1 -1 -1], 'EIB', subjects, resultfolder, 'nostop');

saxelab_create_contrast_amy('EVEN_posmale_vso',{'Sn(2) mh', 'Sn(4) mh', 'Sn(6) mh', 'Sn(8) mh', 'Sn(2) mu', 'Sn(4) mu', 'Sn(6) mu', 'Sn(8) mu'},[1 1 1 1 -1 -1 -1 -1], 'EIB', subjects, resultfolder, 'nostop');

saxelab_create_contrast_amy('ODD_negmale_vso',{'Sn(1) mu', 'Sn(3) mu', 'Sn(5) mu', 'Sn(7) mu', 'Sn(1) mh', 'Sn(3) mh', 'Sn(5) mh','Sn(7) mh'},[1 1 1 1 -1 -1 -1 -1], 'EIB', subjects, resultfolder, 'nostop');

saxelab_create_contrast_amy('EVEN_negmale_vso',{'Sn(2) mu', 'Sn(4) mu', 'Sn(6) mu', 'Sn(8) mu', 'Sn(2) mh', 'Sn(4) mh', 'Sn(6) mh', 'Sn(8) mh'},[1 1 1 1 -1 -1 -1 -1], 'EIB', subjects, resultfolder, 'nostop');


saxelab_create_contrast_amy('ODD_posfema_vso',{'Sn(1) fh', 'Sn(3) fh', 'Sn(5) fh','Sn(7) fh', 'Sn(1) fu', 'Sn(3) fu', 'Sn(5) fu', 'Sn(7) fu'},[1 1 1 1 -1 -1 -1 -1], 'EIB', subjects, resultfolder, 'nostop');

saxelab_create_contrast_amy('EVEN_posfema_vso',{'Sn(2) fh', 'Sn(4) fh', 'Sn(6) fh', 'Sn(8) fh', 'Sn(2) fu', 'Sn(4) fu', 'Sn(6) fu', 'Sn(8) fu'},[1 1 1 1 -1 -1 -1 -1], 'EIB', subjects, resultfolder, 'nostop');

saxelab_create_contrast_amy('ODD_negfema_vso',{'Sn(1) fu', 'Sn(3) fu', 'Sn(5) fu', 'Sn(7) fu', 'Sn(1) fh', 'Sn(3) fh', 'Sn(5) fh','Sn(7) fh'},[1 1 1 1 -1 -1 -1 -1], 'EIB', subjects, resultfolder, 'nostop');

saxelab_create_contrast_amy('EVEN_negfema_vso',{'Sn(2) fu', 'Sn(4) fu', 'Sn(6) fu', 'Sn(8) fu', 'Sn(2) fh', 'Sn(4) fh', 'Sn(6) fh', 'Sn(8) fh'},[1 1 1 1 -1 -1 -1 -1], 'EIB', subjects, resultfolder, 'nostop');


%%These aren't actually divided by even and odd. even=persons,
%%odd=contexts.
%pos_p=EVEN
saxelab_create_contrast_amy('EVEN_fake_pos_vso',{'Sn(1) mh', 'Sn(1) fh', 'Sn(3) mh', 'Sn(3) fh', 'Sn(5) mh', 'Sn(5) fh','Sn(7) mh', 'Sn(7) fh', 'Sn(2) mh', 'Sn(2) fh', 'Sn(4) mh', 'Sn(4) fh', 'Sn(6) mh', 'Sn(6) fh','Sn(8) mh', 'Sn(8) fh', 'Sn(1) mu', 'Sn(1) fu', 'Sn(3) mu', 'Sn(3) fu', 'Sn(5) mu', 'Sn(5) fu','Sn(7) mu', 'Sn(7) fu', 'Sn(2) mu', 'Sn(2) fu', 'Sn(4) mu', 'Sn(4) fu', 'Sn(6) mu', 'Sn(6) fu','Sn(8) mu', 'Sn(8) fu'},[1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1], 'EIB', subjects, resultfolder, 'nostop');

%neg_p=EVEN
saxelab_create_contrast_amy('EVEN_fake_neg_vso',{'Sn(1) mu', 'Sn(1) fu', 'Sn(3) mu', 'Sn(3) fu', 'Sn(5) mu', 'Sn(5) fu','Sn(7) mu', 'Sn(7) fu', 'Sn(2) mu', 'Sn(2) fu', 'Sn(4) mu', 'Sn(4) fu', 'Sn(6) mu', 'Sn(6) fu','Sn(8) mu', 'Sn(8) fu', 'Sn(1) mh', 'Sn(1) fh', 'Sn(3) mh', 'Sn(3) fh', 'Sn(5) mh', 'Sn(5) fh','Sn(7) mh', 'Sn(7) fh', 'Sn(2) mh', 'Sn(2) fh', 'Sn(4) mh', 'Sn(4) fh', 'Sn(6) mh', 'Sn(6) fh','Sn(8) mh', 'Sn(8) fh'},[1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1], 'EIB', subjects, resultfolder, 'nostop');

%pos_c=ODD
saxelab_create_contrast_amy('ODD_fake_pos_vso',{'Sn(1) sh', 'Sn(1) nh', 'Sn(3) sh', 'Sn(3) nh', 'Sn(5) sh', 'Sn(5) nh','Sn(7) sh', 'Sn(7) nh', 'Sn(2) sh', 'Sn(2) nh', 'Sn(4) sh', 'Sn(4) nh', 'Sn(6) sh', 'Sn(6) nh','Sn(8) sh', 'Sn(8) nh', 'Sn(1) su', 'Sn(1) nu', 'Sn(3) su', 'Sn(3) nu', 'Sn(5) su', 'Sn(5) nu','Sn(7) su', 'Sn(7) nu', 'Sn(2) su', 'Sn(2) nu', 'Sn(4) su', 'Sn(4) nu', 'Sn(6) su', 'Sn(6) nu','Sn(8) su', 'Sn(8) nu'},[1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1], 'EIB', subjects, resultfolder, 'nostop');

%neg_c=ODD
saxelab_create_contrast_amy('ODD_fake_neg_vso',{'Sn(1) su', 'Sn(1) nu', 'Sn(3) su', 'Sn(3) nu', 'Sn(5) su', 'Sn(5) nu','Sn(7) su', 'Sn(7) nu', 'Sn(2) su', 'Sn(2) nu', 'Sn(4) su', 'Sn(4) nu', 'Sn(6) su', 'Sn(6) nu','Sn(8) su', 'Sn(8) nu', 'Sn(1) sh', 'Sn(1) nh', 'Sn(3) sh', 'Sn(3) nh', 'Sn(5) sh', 'Sn(5) nh','Sn(7) sh', 'Sn(7) nh', 'Sn(2) sh', 'Sn(2) nh', 'Sn(4) sh', 'Sn(4) nh', 'Sn(6) sh', 'Sn(6) nh','Sn(8) sh', 'Sn(8) nh'},[1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1], 'EIB', subjects, resultfolder, 'nostop');

%% similarly, odd=social, even=nonsocial
saxelab_create_contrast_amy('ODD_fake_poscontext_vso',{'Sn(1) sh', 'Sn(3) sh', 'Sn(5) sh','Sn(7) sh', 'Sn(2) sh', 'Sn(4) sh', 'Sn(6) sh','Sn(8) sh', 'Sn(1) su', 'Sn(3) su', 'Sn(5) su', 'Sn(7) su', 'Sn(2) su', 'Sn(4) su', 'Sn(6) su', 'Sn(8) su'},[1 1 1 1 1 1 1 1 -1 -1 -1 -1 -1 -1 -1 -1], 'EIB', subjects, resultfolder, 'nostop');

saxelab_create_contrast_amy('EVEN_fake_poscontext_vso',{'Sn(2) nh', 'Sn(4) nh', 'Sn(6) nh', 'Sn(8) nh', 'Sn(1) nh', 'Sn(3) nh', 'Sn(5) nh', 'Sn(7) nh', 'Sn(2) nu', 'Sn(4) nu', 'Sn(6) nu', 'Sn(8) nu', 'Sn(1) nu', 'Sn(3) nu', 'Sn(5) nu', 'Sn(7) nu'},[1 1 1 1 1 1 1 1 -1 -1 -1 -1 -1 -1 -1 -1], 'EIB', subjects, resultfolder, 'nostop');

saxelab_create_contrast_amy('ODD_fake_negcontext_vso',{'Sn(1) su', 'Sn(3) su', 'Sn(5) su', 'Sn(7) su', 'Sn(2) su', 'Sn(4) su', 'Sn(6) su', 'Sn(8) su', 'Sn(1) sh', 'Sn(3) sh', 'Sn(5) sh','Sn(7) sh', 'Sn(2) sh', 'Sn(4) sh', 'Sn(6) sh','Sn(8) sh'},[1 1 1 1 1 1 1 1 -1 -1 -1 -1 -1 -1 -1 -1], 'EIB', subjects, resultfolder, 'nostop');

saxelab_create_contrast_amy('EVEN_fake_negcontext_vso',{'Sn(2) nu', 'Sn(4) nu', 'Sn(6) nu', 'Sn(8) nu', 'Sn(1) nu', 'Sn(3) nu', 'Sn(5) nu', 'Sn(7) nu', 'Sn(2) nh', 'Sn(4) nh', 'Sn(6) nh', 'Sn(8) nh', 'Sn(1) nh', 'Sn(3) nh', 'Sn(5) nh', 'Sn(7) nh'},[1 1 1 1 1 1 1 1 -1 -1 -1 -1 -1 -1 -1 -1], 'EIB', subjects, resultfolder, 'nostop');

%%%
%% similarly, odd=male, even=female
saxelab_create_contrast_amy('ODD_fake_posperson_vso',{'Sn(1) mh', 'Sn(3) mh', 'Sn(5) mh','Sn(7) mh', 'Sn(2) mh', 'Sn(4) mh', 'Sn(6) mh','Sn(8) mh', 'Sn(1) mu', 'Sn(3) mu', 'Sn(5) mu', 'Sn(7) mu', 'Sn(2) mu', 'Sn(4) mu', 'Sn(6) mu', 'Sn(8) mu'},[1 1 1 1 1 1 1 1 -1 -1 -1 -1 -1 -1 -1 -1], 'EIB', subjects, resultfolder, 'nostop');

saxelab_create_contrast_amy('EVEN_fake_posperson_vso',{'Sn(2) fh', 'Sn(4) fh', 'Sn(6) fh', 'Sn(8) fh', 'Sn(1) fh', 'Sn(3) fh', 'Sn(5) fh', 'Sn(7) fh', 'Sn(2) fu', 'Sn(4) fu', 'Sn(6) fu', 'Sn(8) fu', 'Sn(1) fu', 'Sn(3) fu', 'Sn(5) fu', 'Sn(7) fu'},[1 1 1 1 1 1 1 1 -1 -1 -1 -1 -1 -1 -1 -1], 'EIB', subjects, resultfolder, 'nostop');

saxelab_create_contrast_amy('ODD_fake_negperson_vso',{'Sn(1) mu', 'Sn(3) mu', 'Sn(5) mu', 'Sn(7) mu', 'Sn(2) mu', 'Sn(4) mu', 'Sn(6) mu', 'Sn(8) mu', 'Sn(1) mh', 'Sn(3) mh', 'Sn(5) mh','Sn(7) mh', 'Sn(2) mh', 'Sn(4) mh', 'Sn(6) mh','Sn(8) mh'},[1 1 1 1 1 1 1 1 -1 -1 -1 -1 -1 -1 -1 -1], 'EIB', subjects, resultfolder, 'nostop');

saxelab_create_contrast_amy('EVEN_fake_negperson_vso',{'Sn(2) fu', 'Sn(4) fu', 'Sn(6) fu', 'Sn(8) fu', 'Sn(1) fu', 'Sn(3) fu', 'Sn(5) fu', 'Sn(7) fu', 'Sn(2) fh', 'Sn(4) fh', 'Sn(6) fh', 'Sn(8) fh', 'Sn(1) fh', 'Sn(3) fh', 'Sn(5) fh', 'Sn(7) fh'},[1 1 1 1 1 1 1 1 -1 -1 -1 -1 -1 -1 -1 -1], 'EIB', subjects, resultfolder, 'nostop');

%%%

%%%other less controlled contrasts

saxelab_create_contrast_amy('ODD_nonsoc_vso',{'Sn(1) nh', 'Sn(1) nu', 'Sn(3) nh', 'Sn(3) nu', 'Sn(5) nh', 'Sn(5) nu','Sn(7) nh', 'Sn(7) nu', 'Sn(1) sh', 'Sn(1) su', 'Sn(3) sh', 'Sn(3) su', 'Sn(5) sh', 'Sn(5) su','Sn(7) sh', 'Sn(7) su'},[1 1 1 1 1 1 1 1 -1 -1 -1 -1 -1 -1 -1 -1], 'EIB', subjects, resultfolder, 'nostop');

saxelab_create_contrast_amy('EVEN_nonsoc_vso',{'Sn(2) nh', 'Sn(2) nu', 'Sn(4) nh', 'Sn(4) nu', 'Sn(6) nh', 'Sn(6) nu','Sn(8) nh', 'Sn(8) nu', 'Sn(2) sh', 'Sn(2) su', 'Sn(4) sh', 'Sn(4) su', 'Sn(6) sh', 'Sn(6) su','Sn(8) sh', 'Sn(8) su'},[1 1 1 1 1 1 1 1 -1 -1 -1 -1 -1 -1 -1 -1], 'EIB', subjects, resultfolder, 'nostop');

saxelab_create_contrast_amy('ODD_social_vso',{'Sn(1) sh', 'Sn(1) su', 'Sn(3) sh', 'Sn(3) su', 'Sn(5) sh', 'Sn(5) su','Sn(7) sh', 'Sn(7) su', 'Sn(1) nh', 'Sn(1) nu', 'Sn(3) nh', 'Sn(3) nu', 'Sn(5) nh', 'Sn(5) nu','Sn(7) nh', 'Sn(7) nu'},[1 1 1 1 1 1 1 1 -1 -1 -1 -1 -1 -1 -1 -1], 'EIB', subjects, resultfolder, 'nostop');

saxelab_create_contrast_amy('EVEN_social_vso',{'Sn(2) sh', 'Sn(2) su', 'Sn(4) sh', 'Sn(4) su', 'Sn(6) sh', 'Sn(6) su','Sn(8) sh', 'Sn(8) su', 'Sn(2) nh', 'Sn(2) nu', 'Sn(4) nh', 'Sn(4) nu', 'Sn(6) nh', 'Sn(6) nu','Sn(8) nh', 'Sn(8) nu'},[1 1 1 1 1 1 1 1 -1 -1 -1 -1 -1 -1 -1 -1], 'EIB', subjects, resultfolder, 'nostop');


saxelab_create_contrast_amy('ODD_persons_vso',{'Sn(1) mh', 'Sn(1) mu', 'Sn(3) mh', 'Sn(3) mu', 'Sn(5) mh', 'Sn(5) mu','Sn(7) mh', 'Sn(7) mu', 'Sn(1) fh', 'Sn(1) fu', 'Sn(3) fh', 'Sn(3) fu', 'Sn(5) fh', 'Sn(5) fu','Sn(7) fh', 'Sn(7) fu', 'Sn(1) nh', 'Sn(1) nu', 'Sn(3) nh', 'Sn(3) nu', 'Sn(5) nh', 'Sn(5) nu','Sn(7) nh', 'Sn(7) nu','Sn(1) sh', 'Sn(1) su', 'Sn(3) sh', 'Sn(3) su', 'Sn(5) sh', 'Sn(5) su','Sn(7) sh', 'Sn(7) su'},[1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1], 'EIB', subjects, resultfolder, 'nostop');

saxelab_create_contrast_amy('EVEN_persons_vso',{'Sn(2) mh', 'Sn(2) mu', 'Sn(4) mh', 'Sn(4) mu', 'Sn(6) mh', 'Sn(6) mu','Sn(8) mh', 'Sn(8) mu', 'Sn(2) fh', 'Sn(2) fu', 'Sn(4) fh', 'Sn(4) fu', 'Sn(6) fh', 'Sn(6) fu','Sn(8) fh', 'Sn(8) fu', 'Sn(2) nh', 'Sn(2) nu', 'Sn(4) nh', 'Sn(4) nu', 'Sn(6) nh', 'Sn(6) nu','Sn(8) nh', 'Sn(8) nu','Sn(2) sh', 'Sn(2) su', 'Sn(4) sh', 'Sn(4) su', 'Sn(6) sh', 'Sn(6) su','Sn(8) sh', 'Sn(8) su'}, [1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1], 'EIB', subjects, resultfolder, 'nostop');

saxelab_create_contrast_amy('ODD_cartoon_vso',{'Sn(1) nh', 'Sn(1) nu', 'Sn(3) nh', 'Sn(3) nu', 'Sn(5) nh', 'Sn(5) nu','Sn(7) nh', 'Sn(7) nu','Sn(1) sh', 'Sn(1) su', 'Sn(3) sh', 'Sn(3) su', 'Sn(5) sh', 'Sn(5) su','Sn(7) sh', 'Sn(7) su', 'Sn(1) mh', 'Sn(1) mu', 'Sn(3) mh', 'Sn(3) mu', 'Sn(5) mh', 'Sn(5) mu','Sn(7) mh', 'Sn(7) mu', 'Sn(1) fh', 'Sn(1) fu', 'Sn(3) fh', 'Sn(3) fu', 'Sn(5) fh', 'Sn(5) fu','Sn(7) fh', 'Sn(7) fu'},[1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1], 'EIB', subjects, resultfolder, 'nostop');

saxelab_create_contrast_amy('EVEN_cartoon_vso',{'Sn(2) nh', 'Sn(2) nu', 'Sn(4) nh', 'Sn(4) nu', 'Sn(6) nh', 'Sn(6) nu','Sn(8) nh', 'Sn(8) nu','Sn(2) sh', 'Sn(2) su', 'Sn(4) sh', 'Sn(4) su', 'Sn(6) sh', 'Sn(6) su','Sn(8) sh', 'Sn(8) su', 'Sn(2) mh', 'Sn(2) mu', 'Sn(4) mh', 'Sn(4) mu', 'Sn(6) mh', 'Sn(6) mu','Sn(8) mh', 'Sn(8) mu', 'Sn(2) fh', 'Sn(2) fu', 'Sn(4) fh', 'Sn(4) fu', 'Sn(6) fh', 'Sn(6) fu','Sn(8) fh', 'Sn(8) fu',},[1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1], 'EIB', subjects, resultfolder, 'nostop');


saxelab_create_contrast_amy('ODD_gendermale_vso',{'Sn(1) mh', 'Sn(1) mu', 'Sn(3) mh', 'Sn(3) mu', 'Sn(5) mh', 'Sn(5) mu','Sn(7) mh', 'Sn(7) mu', 'Sn(1) fh', 'Sn(1) fu', 'Sn(3) fh', 'Sn(3) fu', 'Sn(5) fh', 'Sn(5) fu','Sn(7) fh', 'Sn(7) fu'},[1 1 1 1 1 1 1 1 -1 -1 -1 -1 -1 -1 -1 -1], 'EIB', subjects, resultfolder, 'nostop');

saxelab_create_contrast_amy('EVEN_gendermale_vso',{'Sn(2) mh', 'Sn(2) mu', 'Sn(4) mh', 'Sn(4) mu', 'Sn(6) mh', 'Sn(6) mu','Sn(8) mh', 'Sn(8) mu', 'Sn(2) fh', 'Sn(2) fu', 'Sn(4) fh', 'Sn(4) fu', 'Sn(6) fh', 'Sn(6) fu','Sn(8) fh', 'Sn(8) fu'}, [1 1 1 1 1 1 1 1 -1 -1 -1 -1 -1 -1 -1 -1], 'EIB', subjects, resultfolder, 'nostop');

saxelab_create_contrast_amy('ODD_genderfema_vso',{'Sn(1) fh', 'Sn(1) fu', 'Sn(3) fh', 'Sn(3) fu', 'Sn(5) fh', 'Sn(5) fu','Sn(7) fh', 'Sn(7) fu', 'Sn(1) mh', 'Sn(1) mu', 'Sn(3) mh', 'Sn(3) mu', 'Sn(5) mh', 'Sn(5) mu','Sn(7) mh'},[1 1 1 1 1 1 1 1 -1 -1 -1 -1 -1 -1 -1 -1], 'EIB', subjects, resultfolder, 'nostop');

saxelab_create_contrast_amy('EVEN_genderfema_vso',{'Sn(2) fh', 'Sn(2) fu', 'Sn(4) fh', 'Sn(4) fu', 'Sn(6) fh', 'Sn(6) fu','Sn(8) fh', 'Sn(8) fu', 'Sn(2) mh', 'Sn(2) mu', 'Sn(4) mh', 'Sn(4) mu', 'Sn(6) mh', 'Sn(6) mu','Sn(8) mh', 'Sn(8) mu'}, [1 1 1 1 1 1 1 1 -1 -1 -1 -1 -1 -1 -1 -1], 'EIB', subjects, resultfolder, 'nostop');


end
