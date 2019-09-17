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



%fprintf("Taille de X : %d, %d \n", size(X));
%fprintf("Taille de mu : %d, %d \n", size(mu));
%fprintf("Taille de sigma : %d, %d \n", size(sigma));
mu(1)=mean(X(:,1));
mu(2)=mean(X(:,2));
mu;
sigma(1)=std(X(:,1));
sigma(2)=std(X(:,2));
sigma;

X_norm=(X.-mu)./sigma;





% ============================================================

endfunction
