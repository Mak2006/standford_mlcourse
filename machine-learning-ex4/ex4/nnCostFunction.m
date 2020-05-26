function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% ----------- Part 1 -------------------
% We first do analysis 
%>> size(Theta1) =  25   401
%>> size(Theta2) =  10   26
%>> size(X)      =  5000    400
%>> size of hidden_layer_size and input_layer_size are 1 1
% We are after cost function like 
%J = sum(-y.*log(h) - (1-y).*(log(1 - h))) +(lambda/(2))*theta1'*theta1;
%J = J./m
% from the previous exercise/
 
alayer1 = X
%add 1 as the first col
alayer1 = [ones(m, 1), alayer1];

zl2 = alayer1 * Theta1';
al2 = sigmoid(zl2) % this is not done yet. 
% add the bias unit
al2 = [ones(size(al2,1), 1), al2];

%on to the thrid layer - same as layer 2 withdifferent indices
zl3 = al2 * Theta2';
al3 = sigmoid(zl3) % this is not done yet. 
% add the bias unit 
% al3 = [ones(size(al3,1), 1), al3]; % this works, there is no need to add a bias unit here. 

%the final layer
h = al3

% the double summation did not work as the y is not processed. 
Y = (1:num_labels)==y;

% so J
%J = sum(sum((-Y.*log(h) - (1-Y).*(log(1 - h)))))
%J = sum(sum((-Y.*log(h))-((1-Y).*log(1-h)))); 
J = (1/m) * sum(sum((-Y.*log(h))-((1-Y).*log(1-h))));

% for now we ignore the reg parameter
% J = J./m


% we now implement the sigmoid. 
% seems already provided into the assigment, so using that. 

% that now works


% end here


% - ----------------- moving to part 2 - ---------------------
%FROM THE previous part 
%
 
alayer1 = X
%add 1 as the first col
alayer1 = [ones(m, 1), alayer1];

zl2 = alayer1 * Theta1';
al2 = sigmoid(zl2) % this is not done yet. 
% add the bias unit
al2 = [ones(size(al2,1), 1), al2];

%on to the thrid layer - same as layer 2 withdifferent indices
zl3 = al2 * Theta2';
al3 = sigmoid(zl3) % this is not done yet. 
% add the bias unit 
% al3 = [ones(size(al3,1), 1), al3]; % this works, there is no need to add a bias unit here. 

%the final layer
%we calculate  this again
%h = al3 % this is not needed as we bp to calculate


% the double summation did not work as the y is not processed. 
Y = (1:num_labels)==y;


% now instead of normal way to calculate J 
% we use back propogation and get something like 
% J = J * regulation term
% we now calculate the delta
delta3 = al3 - Y; 
delta2 = (delta3 * Theta2) .* [ones(size(zl2,1),1) sigmoidGradient(zl2)]; 
%signmodi gradient is not implement. so we move to implement it. 
%sigmoid gradient done

delta2 = delta2(:,2:end); 
  
Theta1_gradient = (1/m) * (delta2' * alayer1); 
Theta2_gradient = (1/m) * (delta3' * al2); 

% - Part 3 - 
% adding the reg term 
reg = (lambda/(2*m)) * (sum(sum(Theta1(:,2:end).^2)) + sum(sum(Theta2(:,2:end).^2))); 

%So now J 
J = J + reg

%So now we calculte grads for reg
Theta1_reg_grad = (lambda/m) * [zeros(size(Theta1, 1), 1) Theta1(:,2:end)]; 
Theta2_reg_grad = (lambda/m) * [zeros(size(Theta2, 1), 1) Theta2(:,2:end)];


  



% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
