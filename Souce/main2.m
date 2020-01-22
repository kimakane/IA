clc
close all
clear

disp    = @(x)imagesc(reshape(x,28,28)');
sigmoid = @(x) 1./(1+exp(-x));
softmax = @(x) exp(x)./repmat(sum(exp(x)),size(x,1),1);
% softmax = @(x) exp(x)./sum(exp(x));
model   = @(x,theta) softmax(theta'*x);

load('trainingSet.mat');

[n,m] = size(training_feature);
k = 11;
alpha = 0.00001;

%% training
theta = randn(n,k-1)*0.001;
for ii = 1:15
    for i = 1:m
        x = training_feature(:,i);
        y = zeros(k-1,1);
         y(training_target(i)+1) = 1;
        h_x =  model(x,theta);
        gradi = alpha*grad(x,h_x,y);
        theta = theta -gradi ;
        if (mod(i,2000)==0)
        end
    end
    
end

load('testSet.mat');
model(test_feature(:,1),theta)
