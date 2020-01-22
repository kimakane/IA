function [ gradient ] = grad(X,h_x,y)
m = length(y);
gradient = 1/m*X*(h_x - y)'; 
end

