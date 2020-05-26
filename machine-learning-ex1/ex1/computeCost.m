function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.

% MB - note this funciton merely computes the cost and does not have the iterations 
% MB iterations are there in the gradiencent .m file. 
% MB - dim(theta) = 2,1
% MB - dim(X) = m, 2

%
%%%
cost_v = X*theta - y;  % dim m, 1
sq_cost_v = cost_v.^2;   %sqyared it 
sum_cost = sum(sq_cost_v);
J = (1/(2*m)) * sum_cost;



% =========================================================================

end
