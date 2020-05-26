function [J, grad] = costFunction(theta, X, y)
% This is WEEK 2   
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%  dim(theta) = m,1
%  dim(X) = m, 1
%  dim (Y) = m, 1

% J th = (1/m) ylog(h th) 
% J th = - 1/m  [y log (g(th T . x) ) - (1 - y)* log( 1 - g(th T.x))h
% J th = 1/m [(-1) y log(h) - (1 - y) * log(1 - h)]
%size(X)
%size(theta)
%size(y)
h = sigmoid(X*theta);
#print ("h zise")
%size(h)
#print(h)
J = -y'*log(h) - (1-y')*(log(1 - h));
J = J./m;
% grad = theta - sum (h(x) - y)x
grad= (1/m )*X'*(h-y)







% =============================================================

end
