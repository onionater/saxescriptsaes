function make_taskruns_EIB()
%% created by AES on 3/26/13 
%%make .mat file specifying bold directories for each subject
%% mat file can then be called by listbolds and other scripts to identify task-relevant directories for each subject

rootdir='/mindhive/saxelab2/EIB';

s(1).ID='SAX_EIB_01';
s(1).EIB_main=[11,13,17,19,21,23,25,27];
s(1).tomloc=[29, 31];
s(1).EmoBioLoc=[33];

s(2).ID='SAX_EIB_02';
s(2).EIB_main=[11,13,15,17,19,21,23,25];
s(2).tomloc=[27];
s(2).EmoBioLoc=[];

s(3).ID='SAX_EIB_03';
s(3).EIB_main=[11, 13, 15, 17, 23, 25, 27, 29];
s(3).tomloc=[31, 33];
s(3).EmoBioLoc=[];

s(4).ID='SAX_EIB_04';
s(4).EIB_main=[11, 13, 15, 17, 23, 25, 27, 29];
s(4).tomloc=[31, 33];
s(4).EmoBioLoc=[19, 21];

s(5).ID='SAX_EIB_05';
s(5).EIB_main=[11, 13, 15, 17, 23, 25, 27, 29];
s(5).tomloc=[31, 33];
s(5).EmoBioLoc=[19, 21];

s(6).ID='SAX_EIB_06';
s(6).EIB_main=[11, 13, 15, 17, 23, 25, 27, 29];
s(6).tomloc=[31, 33];
s(6).EmoBioLoc=[19, 21];

s(7).ID='SAX_EIB_07';
s(7).EIB_main=[11, 13, 15, 17, 23, 25, 27, 29];
s(7).tomloc=[31, 33];
s(7).EmoBioLoc=[19, 21];

s(8).ID='SAX_EIB_08';
s(8).EIB_main=[11, 13, 15, 17, 23, 25, 27, 29];
s(8).tomloc=[31, 33];
s(8).EmoBioLoc=[19, 21];

s(9).ID='SAX_EIB_09';
s(9).EIB_main=[11, 13, 15, 17, 23, 25, 27, 29];
s(9).tomloc=[31, 33];
s(9).EmoBioLoc=[19, 21];

s(10).ID='SAX_EIB_10';
s(10).EIB_main=[11, 13, 15, 17, 23, 25, 27, 29];
s(10).tomloc=[31, 33];
s(10).EmoBioLoc=[19, 21];

s(11).ID='SAX_EIB_11';
s(11).EIB_main=[13, 15, 17, 25, 27, 29 , 31, 33];
s(11).tomloc=[21, 23];
s(11).EmoBioLoc=[];

s(12).ID='SAX_EIB_12';
s(12).EIB_main=[11, 13, 15, 17, 23, 25, 27, 29];
s(12).tomloc=[31, 33];
s(12).EmoBioLoc=[19, 21];

s(13).ID='SAX_EIB_13';
s(13).EIB_main=[11, 13, 15, 17, 23, 25, 27, 29];
s(13).tomloc=[31, 33];
s(13).EmoBioLoc=[19, 21];

s(15).ID='SAX_EIB_15';
s(15).EIB_main=[11, 13, 15, 17, 23, 25, 27, 29];
s(15).tomloc=[31, 33];
s(15).EmoBioLoc=[19, 21];

s(16).ID='SAX_EIB_16';
s(16).EIB_main=[11, 13, 15, 17, 23, 25, 27, 29];
s(16).tomloc=[31, 33];
s(16).EmoBioLoc=[19, 21];

s(17).ID='SAX_EIB_17';
s(17).EIB_main=[11, 13, 15, 17, 23, 25, 27, 29];
s(17).tomloc=[31, 33];
s(17).EmoBioLoc=[19, 21];

s(18).ID='SAX_EIB_18';
s(18).EIB_main=[11, 13, 15, 17, 23, 25, 27, 29];
s(18).tomloc=[31, 33];
s(18).EmoBioLoc=[19, 21];

s(19).ID='SAX_EIB_19';
s(19).EIB_main=[11, 13, 15, 17, 23, 25, 27, 29];
s(19).tomloc=[31];
s(19).EmoBioLoc=[19, 21];

s(20).ID='SAX_EIB_20';
s(20).EIB_main=[11, 13, 15, 17, 23, 25, 27, 29];
s(20).tomloc=[31, 33];
s(20).EmoBioLoc=[19, 21];

cd(rootdir)
save('EIB_subject_taskruns.mat', 's')

end