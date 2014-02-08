function roi_combine2img_aes()

% prompts the user to select a set of ROI files, and a template image. It
% then combines the ROIs into a single image, where each ROI has a
% different number (i.e., 1, 2, 3, 4, 5...)

% weirdly, it appears that spm_write_vol expects coordinates in
% millimeters (whereas spm_sample_vol, which extracts data from volumes,
% expects it voxelwise)

multifactor		= 1;    
ROI				= spm_select(1,'img','Choose a Template image','',pwd,'.*',1);
roi_fulloc		= spm_select(inf,'mat','Choose an ROI .mat file(s)','',pwd,'.mat',1);
outp			= spm_select(1,'dir','Select an output directory.','',pwd,'.*',1);
name			= input(sprintf('Please type a name for the resulting image.\t'),'s');
v				= spm_vol(ROI);
[Y, XYZ]		= spm_read_vols(v);
Y_roi   		= zeros(size(Y));
Y_size			= [size(Y,1) size(Y,2) size(Y,3)];
Y_vector		= reshape(Y_roi,1,[]);
for i=1:size(roi_fulloc,1)
	[ROIp ROIn] = fileparts(roi_fulloc(i,:));
	temp		= load(fullfile(ROIp,[ROIn '.mat']));
    try
        ROI_t	= temp.ROI.XYZmm';
    catch
        try
            ROI_t = temp.roi_xyz;
        catch
            ROI_t = temp.xY.XYZmm';
        end
    end
	for j = 1:size(ROI_t,1)
		Y_vector(XYZ(1,:) == ROI_t(j,1) & XYZ(2,:) == ROI_t(j,2) & XYZ(3,:) == ROI_t(j,3)) = 1; %i*multifactor;
    end
    fprintf('%s denoted by voxels having value %.0f\n',ROIn,i*multifactor);
end
Y_roi			= reshape(Y_vector,Y_size(1),Y_size(2),Y_size(3));
curD			= pwd;
cd(outp);
v.fname			= fullfile(outp,[name '.img']);
spm_write_vol(v,Y_roi);
fprintf('Wrote %s.img\n',name);
cd(curD);
end