clc
close all
clear

%% Base d'entrainement
disp = @(x)imagesc(reshape(x,28,28)');
load('testSet.mat');
load('trainingSet');

[~,n] = size(training_feature);
m = 1000 ;%length(test_feature);

% Nombre de voisins
K = 5;

%Calcul des distances
Matrice_Norme = zeros(m,n);
for i=1:m
   z = vecnorm(training_feature - test_feature(:,i),2);
   Matrice_Norme(i,:) = z;
   i
end

[Distances_Min_Triees,indices]= sort(Matrice_Norme,2);
training_target = training_target';
Matrice_Classe = training_target(indices);
valeur= mode(Matrice_Classe(:,1:K),2);

%Pourcentage d'erreur
erreur = sum(valeur~=test_target(1:m))/m;
    



