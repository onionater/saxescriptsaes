function motion = get_motion_amy(study,subject)
% returns motion vectors catenated across all runs for a given subject
% (x,y,z,roll,pitch,yaw)?
% example:
% motion = get_motion('Nars2','SAX_nars_32');

%currently isn't that helpful. figure out if there is something useful to do with this


cd /mindhive/saxelab2/
cd(study);
cd(subject);
v = adir('bold/*/art_*_and_movement_*.mat');
motion = [];
for i=1:length(v)
    load(v{i});
    motion = [motion; R(:,end-5:end)];
end
a = [1 3 5 2 4 6];
for i=1:6
    subplot(3,2,a(i));plot(cumsum(motion(:,i)));
    if i==1
        title('Translation');
    end
    if i==4
        title('Rotation');
    end
    axis tight
end
end
