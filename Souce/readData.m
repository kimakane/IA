clc
close all 
clear

 addpath(genpath('./'));
 
fd_feature = fopen('t10k-images.idx3-ubyte','r');
fd_target = fopen('t10k-labels.idx1-ubyte','r');

test_feature_info = fread(fd_feature,4,'uint32','b');
test_target_info  = fread(fd_target,2,'uint32','b');

test_feature     = fread(fd_feature,[28*28 60000],'unsigned char','b');
test_target      = fread(fd_target,60000,'unsigned char','b');

fclose(fd_feature);
fclose(fd_target);

save('testSet','test_feature_info','test_target_info','test_feature','test_target')