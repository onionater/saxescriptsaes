function img2roi_proper(roidir, key)
% similar to img2roi but gets things set up in appropriate format for the
% new autoroi scripts
lookingfor=[roidir, key, '.img']
rois2fix=dir(lookingfor)

for r=1:length(rois2fix)
roisel=rois2fix(r);    
roiimg=roisel.name;
roistring=roiimg(1:end-4)
roimat=[roistring, '.mat'];
%copyfile([roidir, roimat],[roidir, roistring, '_old.mat'] )
V=spm_vol([roidir, roiimg]);
[Y,XYZ] = spm_read_vols(V);
roi_xyz         = XYZ(:,Y>0);
roi_xyz         = roi_xyz';
xY.def='parcel';
xY.xyz='';
xY.M=V.mat;
xY.spec=9;
if exist('roi_notes')
xY.str=roi_notes;
else
xY.str='created by AES';
end
xY.XYZmm=roi_xyz';

save([roidir, roimat], 'roi_xyz','xY')


end
end