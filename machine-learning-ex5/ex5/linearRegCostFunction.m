function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%


% Setting J to the cost and 
% grad to the gradient 
%  We start with J, from prev assignment we have 
% j = 1/2m sum (h theta x  - y) squared + reg parameter
%{
>> size(X), size(y), size(theta), size(lambda)
ans = 
   12    1
ans =
   12    1
ans =
   9   1
ans =
   1   1
>>
%}
% the reg parameter is required 
J = (1/( 2*m)) * sum((X*theta - y ).^2) + (lambda/(2*m))* sum((theta(2:end).^2))

% Moving on the grad
% =========================================================================
% as per earlier notes, we can represent bot the = 0 case and >= 1 case using the log functino
%j(?)=?1/m[?mi=1y(i)logh?(x(i))+(1?y(i))log(1?h?(x(i)))]
% great octave editor can show the chars 
h = x*theta
g  = (-1/m) * sum(ylog(h) + (1 - y)* log(1 - h))
%grad = grad(:);

end
