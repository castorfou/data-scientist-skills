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
                 hidden_layer_size, (input_layer_size + 1)); % dimcheck hidden_layer_size, input_layer_size+1 : 25, 401

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1)); % dimcheck output_layer_size, hidden_layer_size+1 : 10, 26

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1)); 
Theta2_grad = zeros(size(Theta2));
Delta1_grad = zeros(size(Theta1)); 
Delta2_grad = zeros(size(Theta2));


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

a1 = [ones(m, 1) X]; % dimcheck m, input_layer_size+1: 5000, 401

z2=a1 * Theta1'; % dimcheck  m, hidden_layer_size :(5000x25) = (5000x401) * (401x25) : 5000, 25
a2=sigmoid(z2); % dimcheck  m, hidden_layer_size :(5000x25) = (5000x401) * (401x25) : 5000, 25
A2= [ones(m, 1) a2]; % dimcheck  m, hidden_layer_size+1 : 5000, 26

z3=A2 * Theta2'; % dimcheck m, output_layer_size : (5000x10) = (5000x26) * (26x10) : 5000, 10
a3=sigmoid(z3); % dimcheck m, output_layer_size : (5000x10) = (5000x26) * (26x10) : 5000, 10

%a3 est égal à h_theta

yv=[1:num_labels] == y; % dimcheck m, K(ou output_layer_size) : 5000,10

theta1_without1stcolumn=Theta1(:,2:end);
theta1_without1stcolumn=theta1_without1stcolumn.^2;
theta1_without1stcolumn=theta1_without1stcolumn(:);

theta2_without1stcolumn=Theta2(:,2:end);
theta2_without1stcolumn=theta2_without1stcolumn.^2;
theta2_without1stcolumn=theta2_without1stcolumn(:);

J=sum(sum(-yv .* log(a3) - (1-yv) .* log(1-a3)))/m + lambda*sum(theta1_without1stcolumn)/(2*m) + lambda*sum(theta2_without1stcolumn)/(2*m);
%   5000x10 5000x10       5000x10    5000x10                         25x400                        10x25

theta1_with_null_1stcolumn=Theta1(:,2:end);
theta1_with_null_1stcolumn=[zeros(hidden_layer_size,1) theta1_with_null_1stcolumn];

output_layer_size= size(yv, 2);
theta2_with_null_1stcolumn=Theta2(:,2:end);
theta2_with_null_1stcolumn=[zeros(output_layer_size,1) theta2_with_null_1stcolumn];

for t=1:m % m=5000
  x_t=X(t,:); % 1, 400
  y_t=yv(t,:); % 1, 10
  a_1 = [1 x_t]; % 1, 401
  z_2 = a_1 * Theta1'; % 1, 25 [1, 401 * 401, 25]
  a_2 = sigmoid(z_2); % 1, 25
  A_2 = [1 a_2]; % 1, 26
  z_3 = A_2 * Theta2'; % 1, 10 [1, 26 * 26, 10]
  a_3 = sigmoid(z_3); % 1, 10
  
  delta3 = a_3 - y_t; % 1, 10
  delta2 = delta3*Theta2.*sigmoidGradient([0 z_2]); % 1,26 [1,10 * 10, 26  1,26]
  delta2=delta2(2:end); % 1, 25
  
  Delta1_grad=Delta1_grad+delta2'*a_1;% 25,401 [25,1 1,401]
  Delta2_grad=Delta2_grad+delta3'*A_2;% 10,26 [10,1 1,26]
  
%  Theta1_grad = Theta1_grad+(delta2'*a_1 +lambda*theta1_with_null_1stcolumn)/m;  % 25,401 [25,1 1,401]
%  Theta2_grad = Theta2_grad+(delta3'*A_2 +lambda*theta2_with_null_1stcolumn)/m;  % 10,26 [10,1 1,26]
  Theta1_grad=Delta1_grad/m+lambda*theta1_with_null_1stcolumn/m;
  Theta2_grad=Delta2_grad/m+lambda*theta2_with_null_1stcolumn/m;
  
  
endfor







% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
