function computetime=testparfor()

matlabpool

t1=now;

for x=1:1000
    y=x+10;
    pause(1)
end


t2=now;

computetime(1)=t2-t1


t1=now;

parfor x=1:1000
    z=x+10;
    pause(1)
end


t2=now;

computetime(2)=t2-t1

diff=computetime(1)-computetime(2)
ratio=computetime(1)/computetime(2)