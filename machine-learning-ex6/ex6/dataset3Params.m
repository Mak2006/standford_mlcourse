function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

%{
For both C and , we
suggest trying values in multiplicative steps (e.g., 0:01; 0:03; 0:1; 0:3; 1; 3; 10; 30).
Note that you should try all possible pairs of values for C and  (e.g., C = 0:3
and  = 0:1).
%}

errormin = 10000 # assuming a large error
for c = [0.01 0.03 0.1 0.3 1 3 10 30]
  for sigma = [0.01 0.03 0.1 0.3 1 3 10 30]
     modl = svmTrain(X, y, c, @(x1, x2) gaussianKernel(x1, x2, sigma));
     predictions = svmPredict(modl, Xval);
     
     % going over to the mean
     %  Note: You can compute the prediction error using 
     err =  mean(double(predictions ~= yval))
     
     % get the minimum
     if err < errormin    
       errormin = err;
       cmin = c
       sigmin = sigma
     end
  endfor
endfor
  
% use the cmin and sigmin to return the result
C = c(imin);
sigma = sigma(sigma);




% =========================================================================

end
