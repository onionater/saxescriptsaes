
delete_images=0;

imgfiles=dir('q*.img')
numfiles=length(imgfiles);

for i=1:numfiles
    
    imgname=imgfiles(i).name
    matname=[imgname(2:end-3), 'mat']
    temp=dir(matname)
    if isempty(temp)
        matname=[imgname(1:end-3), 'mat']
    end
    mat=load(matname)
    fields=fieldnames(mat)
    if length(fields)>1
    roinotes=mat.roinotes;
    else 
        roinotes='none'
    end

    img             = spm_vol(imgname);
    
    [Y,XYZ]         = spm_read_vols(img);
    
    roi_xyz         = XYZ(:,Y>0);
    
    roi_xyz         = roi_xyz';
    save(['q' imgname(2:end-3), 'mat'],'roi_xyz', 'roinotes')

    if delete_images
        delete([namepart '.img']);
        delete([namepart '.hdr']);
    end
end