function [X_norm, mu, sigma] = featureNormalize(X)
%FEATURENORMALIZE Normalizes the features in X 
%   FEATURENORMALIZE(X) returns a normalized version of X where
%   the mean value of each feature is 0 and the standard deviation
%   is 1. This is often a good preprocessing step to do when
%   working with learning algorithms.

% You need to set these values correctly
X_norm = X;
mu = zeros(1, size(X, 2));
sigma = zeros(1, size(X, 2));

% ====================== YOUR CODE HERE ======================
% Instructions: First, for each feature dimension, compute the mean
%               of the feature and subtract it from the dataset,
%               storing the mean value in mu. Next, compute the 
%               standard deviation of each feature and divide
%               each feature by it's standard deviation, storing
%               the standard deviation in sigma. 
%
%               Note that X is a matrix where each column is a 
%               feature and each row is an example. You need 
%               to perform the normalization separately for 
%               each feature. 
%
% Hint: You might find the 'mean' and 'std' functions useful.
%       
% MB -  HERE X is the transfromed X such that (x_1^i - mu_a)/sig_a,(x_2^i - mu_b)/sig_b, 
% mu is [mu_a, mu_b]
% sig is [sig_a, sig_b

% first would be get the features 
% commented as thi impl does not work for all matrix sizes
%{
area = X(:,1);
room = X(:,2);

format short;

mu = [mean(area), mean(room)]
sig = [std(area), std(room)]

area = (area - mu(1,1))/sig(1,1);
room = (room - mu(1,2))/sig(1,2);

X_norm = [area, room];
%}

mu = sum(X,1)/size(X, 1)
%sig = max(X) - min(X)
sigma = std(X)

X_norm = (X - mu) ./ sigma



% ============================================================

end
