clc
close all
clear


%% definition fonction
disp    = @(x)imagesc(reshape(x,28,28)');
disp2   = @(set,ind) disp(set(:,ind));
sigmoid = @(x) 1./(1+exp(-x));
softmax = @(x) exp(x)./repmat(sum(exp(x)),size(x,1),1);
% softmax = @(x) exp(x)./sum(exp(x));
model   = @(x,theta) softmax(theta'*x);

%% paramètre
 addpath(genpath('./'));
load('trainingSet.mat');
load('testSet.mat');
load('model1.mat')
[n,m] = size(training_feature);
k = 11;
alpha = 0.00001;

%% training
% theta = randn(n,k-1)*0.001;
% for ii = 1:15
%     for i = 1:m
%         x = training_feature(:,i);
%         y = zeros(k-1,1);
%          y(training_target(i)+1) = 1;
%         h_x =  model(x,theta);
%         gradi = alpha*grad(x,h_x,y);
%         theta = theta -gradi ;
%         if (mod(i,2000)==0)
%         end
%     end
%     
% end
% filename = './model_save/model1';
% save(filename,'theta');
%% test
test_output = model(test_feature,theta); %% matrice avec les proba que image appartient a la classe
[test_proba_max , test_predict_class] = max(test_output.',[],2);
test_predict_err = (test_predict_class~= (test_target+1));
err_img          = test_feature(:,test_predict_err==1);
err_predic_proba = test_proba_max(test_predict_err==1);
err_pourcent     = length(err_img)/length(test_target);  