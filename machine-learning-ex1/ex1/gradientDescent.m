function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);
converge = 0.000001;
cost_diff = 20;

for iter = 1:num_iters

    cost = computeCost(X, y, theta);
    sprintf( "iter = %d cost = %f and cost_diff = %f", iter, cost, cost_diff)
    if(iter >5)
      cost_diff = abs(cost - J_history(iter - 1));
      sprintf( " cost_diff updated to = %f",  cost_diff)
    endif
    if(cost_diff < converge)
      sprintf( " CONVERGING AT ITER %d",  iter)
      break;
    endif
    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
    %{ MB -  we require to do 
    
    % th_j = th_j  - a/m * ( 
    
    
    %}
    
    t1 = theta(1, 1) - alpha*(1/m)*sum((X*theta - y)'*X(:, 1));
    t2 = theta(2, 1) - alpha*(1/m)*sum((X*theta - y)'*X(:, 2));
    sprintf( "iter = %d theta [1,2] =[ %d, %d]", iter, t1, t2)
   
    theta(1, 1) = t1;
    theta(2, 1) = t2;
    
    %sprintf( "iter = " + iter + " theta [1,2] =[" + t1 + ", " + t2);
    
    %sprintf( "iter = " + iter + " theta [1,2] =[" + theta(1, 1) + ", " + theta(2, 1));
    
   

    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = cost;

end

end
