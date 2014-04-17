%AES edited out some DIS specifics

addpath /software/spm8
addpath /software/spm_ss

study='EIB';
rootdir=['/mindhive/saxelab2/' study '/'];
mvpadir='EIB_mvpa';
resultsdir='results/EIB_main_with_art_reg_results_normed';



%% defines data sources
%   contrast files to consider for each subject are read from subject-specific SPM.mat files
experiments=struct(...
    'name','EIB',...
    'pwd1',rootdir,...   %folder with participants 
    'pwd2',resultsdir,...   %inside each participant, path to .spm   
   'data',{{'SAX_EIB_01','SAX_EIB_02','SAX_EIB_03','SAX_EIB_04','SAX_EIB_05','SAX_EIB_06','SAX_EIB_07','SAX_EIB_08','SAX_EIB_09','SAX_EIB_10','SAX_EIB_11','SAX_EIB_12','SAX_EIB_13', 'SAX_EIB_15', 'SAX_EIB_16', 'SAX_EIB_17', 'SAX_EIB_18', 'SAX_EIB_19'}});
 
partition_names={'ODD_','EVEN_'};                              % partition names
condition_names={
    %'pos_all','neg_all'
    %'pos_p','neg_p'
    %'pos_c','neg_c'
    %'fake_pos','fake_neg'
    %'possocial', 'negsocial'
    %'posnonsoc', 'negnonsoc'
    %'posmale', 'negmale'
    %'posfema', 'negfema'
    %'fake_poscontext', 'fake_negcontext'
    %'fake_posperson', 'fake_negperson'
    %'social', 'nonsoc'
    %'cartoon', 'persons'
    'gendermale', 'genderfema'
    };
                            
filepathoutput=[rootdir '/' mvpadir '/SL_mvpa_' resultsdir '_' condition_names{1} '-' condition_names{2}];  % absolute path for output files

%% options
smoothROI=1;             % make ROI probabalistic so that voxels closer to the center are weighted more strongly                
roisize=14;             % FWHM (mm) of searchlight analysis sphere
center=0;               % between-conditions centering (condition-specific patterns of activation are centered to a mean of zero in each voxel across all conditions) NOTE: this introduces between-condition differences in within-condition estimates, use with care 
removetemporalfiles=1;
perform_firstlevel=1;


%% locates appropriate subject-level files
opwd=pwd;
nconditions=numel(condition_names);
npartitions=numel(partition_names);

if perform_firstlevel
    fprintf('\nLocating appropriate subject-level files');
    spm_data=[];
    datafilename={};
    if ~isempty(experiments)
        for nsubject=1:numel(experiments.data),
            fprintf('.');
            current_spm=fullfile(experiments.pwd1,experiments.data{nsubject},experiments.pwd2,'SPM.mat'); %select spm.mat for task
            [spm_data,SPM]=spm_ss_importspm(spm_data,current_spm);
            Cnames={SPM.xCon(:).name}; %%make list of all con names
            ic=[];ok=1;
            for n1=1:npartitions,
                for n2=1:nconditions,
                    temp=strmatch([partition_names{n1},condition_names{n2}],Cnames,'exact');if numel(temp)~=1,ok=0;break;else ic(n1,n2)=temp;end; %make sure contrast exists, and if it does 
                    datafilename{nsubject,n1,n2}=fullfile(fileparts(current_spm),['con_',num2str(ic(n1,n2),'%04d'),'.img']); % subjects x partitions x conditions
                end
            end
            if ~ok, error(['contrast name ',[partition_names{n1},condition_names{n2}],' not found at ',current_spm]); end
        end
    end
    
    %% Computes within-condition and between-condition spatial correlations of effect sizes. Bivariate correlations are computed in both cases across different data partitions (e.g. ODD vs. EVEN)
    nsubs=size(datafilename,1);
    [nill,nill]=mkdir(filepathoutput);
    fprintf('\nPerforming first-level analyses');
    for nsub=1:nsubs,
        fprintf('\nProcessing subject #%d',nsub);
        
        for npart=1:npartitions,
            if center
                B=0; Bn=0;
                for ncondition=1:nconditions,
                    filename=datafilename{nsub,npart,ncondition};
                    a=spm_vol(filename);
                    b=spm_read_vols(a);
                    B=B+b;
                    Bn=Bn+1;
                end
                B=B/Bn;%% average response across all conditions?
            else B=0; end
            for ncondition=1:nconditions,
                filename=datafilename{nsub,npart,ncondition};
                a=spm_vol(filename);
                b=spm_read_vols(a);
                filename=fullfile(filepathoutput,sprintf('__SL_x_Subject_%d_Condition_%d_Partition_%d.nii',nsub,ncondition,npart));
                c=struct('fname',filename,'mat',a.mat,'dim',a.dim,'dt',[spm_type('float32'),spm_platform('bigend')]);
                spm_write_vol(c,b-B); %%subtract mean across conditions from individual condition (centering). if center=0, B=0, and you are left with condition beta
            end
            
            for ncondition=1:nconditions,
                fprintf('.');
                % load and smooth each condition x partition
                filename=fullfile(filepathoutput,sprintf('__SL_x_Subject_%d_Condition_%d_Partition_%d.nii',nsub,ncondition,npart));
                a=spm_vol(filename);
                filename=fullfile(filepathoutput,sprintf('__SL_sx_Subject_%d_Condition_%d_Partition_%d.nii',nsub,ncondition,npart)); %s for smooth
                spm_smooth(a,filename,roisize);
                
                % load, square, and smooth each condition x partition
                b=spm_read_vols(a);
                filename=fullfile(filepathoutput,sprintf('__SL_x2_Subject_%d_Condition_%d_Partition_%d.nii',nsub,ncondition,npart)); %x2 for squared
                c=struct('fname',filename,'mat',a.mat,'dim',a.dim,'dt',[spm_type('float32'),spm_platform('bigend')]);
                c=spm_write_vol(c,b.^2);
                filename=fullfile(filepathoutput,sprintf('__SL_sx2_Subject_%d_Condition_%d_Partition_%d.nii',nsub,ncondition,npart)); %s for smooth
                spm_smooth(c,filename,roisize);
            end
        end
        for npart1=1:npartitions,
            for npart2=npart1+1:npartitions,
                for ncondition1=1:nconditions,
                    for ncondition2=1:nconditions,
                        fprintf('.');
                        filename=fullfile(filepathoutput,sprintf('__SL_x_Subject_%d_Condition_%d_Partition_%d.nii',nsub,ncondition1,npart1));
                        a1=spm_vol(filename);
                        filename=fullfile(filepathoutput,sprintf('__SL_x_Subject_%d_Condition_%d_Partition_%d.nii',nsub,ncondition2,npart2));
                        a2=spm_vol(filename);
                        x=spm_read_vols(a1);
                        y=spm_read_vols(a2);
                        filename=fullfile(filepathoutput,sprintf('__SL_xy_Subject_%d_Condition_%d_Partition_%d_Condition_%d_Partition_%d.nii',nsub,ncondition1,npart1,ncondition2,npart2));
                        c=struct('fname',filename,'mat',a1.mat,'dim',a1.dim,'dt',[spm_type('float32'),spm_platform('bigend')]);
                        c=spm_write_vol(c,x.*y);
                        filename=fullfile(filepathoutput,sprintf('__SL_sxy_Subject_%d_Condition_%d_Partition_%d_Condition_%d_Partition_%d.nii',nsub,ncondition1,npart1,ncondition2,npart2));
                        spm_smooth(c,filename,roisize);
                        
                        filename=fullfile(filepathoutput,sprintf('__SL_sxy_Subject_%d_Condition_%d_Partition_%d_Condition_%d_Partition_%d.nii',nsub,ncondition1,npart1,ncondition2,npart2));
                        sxy=spm_read_vols(spm_vol(filename));
                        filename=fullfile(filepathoutput,sprintf('__SL_sx_Subject_%d_Condition_%d_Partition_%d.nii',nsub,ncondition1,npart1));
                        sx=spm_read_vols(spm_vol(filename));
                        filename=fullfile(filepathoutput,sprintf('__SL_sx_Subject_%d_Condition_%d_Partition_%d.nii',nsub,ncondition2,npart2));
                        sy=spm_read_vols(spm_vol(filename));
                        
                        filename=fullfile(filepathoutput,sprintf('__SL_sx2_Subject_%d_Condition_%d_Partition_%d.nii',nsub,ncondition1,npart1));
                        sx2=spm_read_vols(spm_vol(filename));
                        filename=fullfile(filepathoutput,sprintf('__SL_sx2_Subject_%d_Condition_%d_Partition_%d.nii',nsub,ncondition2,npart2));
                        sy2=spm_read_vols(spm_vol(filename));
                        
                        r=(sxy - sx.*sy)./max(eps,(sqrt(sx2-sx.^2).*sqrt(sy2-sy.^2)));
                        r(isnan(x)|isnan(y))=nan; %if either x or y is NaN for a voxel, make r=NaN
                        
                        filename=fullfile(filepathoutput,sprintf('__SL_r_Subject_%d_Condition_%d_Partition_%d_Condition_%d_Partition_%d.nii',nsub,ncondition1,npart1,ncondition2,npart2));
                        c=struct('fname',filename,'mat',a1.mat,'dim',a1.dim,'dt',[spm_type('float32'),spm_platform('bigend')]);
                        c=spm_write_vol(c,r);
                        filename=fullfile(filepathoutput,sprintf('__SL_z_Subject_%d_Condition_%d_Partition_%d_Condition_%d_Partition_%d.nii',nsub,ncondition1,npart1,ncondition2,npart2));
                        c=struct('fname',filename,'mat',a1.mat,'dim',a1.dim,'dt',[spm_type('float32'),spm_platform('bigend')]);
                        c=spm_write_vol(c,atanh(r));
                    end
                end
            end
        end
        
        for ncondition1=1:nconditions,
            for ncondition2=1:nconditions,
                fprintf('.');
                B=0;Bn=0;
                for npart1=1:npartitions,
                    for npart2=npart1+1:npartitions,
                        filename=fullfile(filepathoutput,sprintf('__SL_z_Subject_%d_Condition_%d_Partition_%d_Condition_%d_Partition_%d.nii',nsub,ncondition1,npart1,ncondition2,npart2));
                        a=spm_vol(filename);
                        b=spm_read_vols(a);
                        B=B+b;
                        Bn=Bn+1;
                        if ncondition1~=ncondition2,
                            filename=fullfile(filepathoutput,sprintf('__SL_z_Subject_%d_Condition_%d_Partition_%d_Condition_%d_Partition_%d.nii',nsub,ncondition2,npart1,ncondition1,npart2));
                            a=spm_vol(filename);
                            b=spm_read_vols(a);
                            B=B+b;
                            Bn=Bn+1;
                        end
                    end
                end
                B=B/Bn;
                if ncondition1==ncondition2, filename=fullfile(filepathoutput,sprintf('SL_Within_%s_Subject_%d.nii',condition_names{ncondition1},nsub));
                else filename=fullfile(filepathoutput,sprintf('SL_Between_%s_%s_Subject_%d.nii',condition_names{ncondition1},condition_names{ncondition2},nsub)); end
                c=struct('fname',filename,'mat',a1.mat,'dim',a1.dim,'dt',[spm_type('float32'),spm_platform('bigend')]);
                c=spm_write_vol(c,B);
            end
        end
        
        for ncondition1=1:nconditions,
            for ncondition2=[1:ncondition1-1,ncondition1+1:nconditions],
                filename=fullfile(filepathoutput,sprintf('SL_Within_%s_Subject_%d.nii',condition_names{ncondition1},nsub));
                a=spm_vol(filename);
                B=spm_read_vols(a);
                filename=fullfile(filepathoutput,sprintf('SL_Between_%s_%s_Subject_%d.nii',condition_names{ncondition1},condition_names{ncondition2},nsub));
                a=spm_vol(filename);
                B=B-spm_read_vols(a);
                filename=fullfile(filepathoutput,sprintf('SL_Within_%s_vs_Between_%s%s_Subject_%d.nii',condition_names{ncondition1},condition_names{ncondition1},condition_names{ncondition2},nsub));
                c=struct('fname',filename,'mat',a.mat,'dim',a.dim,'dt',[spm_type('float32'),spm_platform('bigend')]);
                c=spm_write_vol(c,B);
            end
        end
        
        % removing temporal files 
        if removetemporalfiles
            filesremove=spm_select('List',filepathoutput,'^__SL_.*\.nii$');
            for n1=1:size(filesremove,1), spm_unlink(fullfile(filepathoutput,deblank(filesremove(n1,:)))); end
        end
        
%mkdir(fullfile(filepathoutput,'data'));
        cd(fullfile(filepathoutput));  
      
         
        a =   sprintf('SL_Within_%s_Subject_%d.nii',condition_names{ncondition1},nsub);
        b =   sprintf('SL_Within_%s_Subject_%d.nii',condition_names{ncondition2},nsub);
        c = sprintf('SL_Between_%s_%s_Subject_%d.nii',condition_names{ncondition1},condition_names{ncondition2},nsub)
    	d = sprintf('SL_Between_%s_%s_Subject_%d.nii',condition_names{ncondition2},condition_names{ncondition1},nsub)
       

        
        hdr = (spm_vol(a));
        
         a = spm_read_vols(spm_vol(a));
         b = spm_read_vols(spm_vol(b));
         c = spm_read_vols(spm_vol(c));
         d = spm_read_vols(spm_vol(d));

        e = ((a+b)./2)-((c+d)./2);

        

        name = sprintf(['1SL_Within_Between_' condition_names{1} 'vs' condition_names{2} '_%s.img'], num2str(nsub));

        hdr.fname = name;
        spm_write_vol(hdr,e);

    end 
end