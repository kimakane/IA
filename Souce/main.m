clc
close all
clear

sigmoid = @(x) 1./(1+exp(-x));
softmax = @(x) exp(x)./sum(exp(x));
load('trainingSet.mat');

[n,m] = size(training_feature);
weight =randn(n,1);


target = (training_target==2);
alpha = 3;
for k = 1:1000
    input = training_feature;
    h = sigmoid(input'*weight);
    weight = weight + alpha*sum(input*(target-h),2);
    disp(k)
end

load('testSet.mat');
target = (training_target==2);
resultat = sigmoid(training_feature'*weight)>0.5;
err = sum(resultat==target)/60000;

