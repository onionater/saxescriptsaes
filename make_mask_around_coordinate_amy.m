function make_mask_around_coordinate_amy(coordinate, radius, name)


-make .mat file containing coordinates you want to include in mask (by computing radius around coordinate)


then call mat2img to make an image

save in format 'ROI_nameofregion_whereitcamefrom_xyz.img' to '/mindhive/saxelab/roi_library/functional/EIBrois'
