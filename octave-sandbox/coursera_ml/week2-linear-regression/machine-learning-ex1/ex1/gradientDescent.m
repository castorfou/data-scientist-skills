function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);


%fprintf("Taille de X : %d, %d \n", size(X));
%fprintf("Taille de y : %d, %d \n", size(y));
%fprintf("Taille de theta : %d, %d \n", size(theta));
%fprintf("Taille de alpha : %d, %d \n", size(alpha));

%thetaXy = X * theta - y;
%fprintf("Taille de thetaX - y : %d, %d \n", size(thetaXy));

%fprintf("Cost %d\n", computeCost(X, y, theta));


for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %

    newtheta0=theta(1)-alpha * sum( (X*theta-y) .* X(:,1)) / m;
    newtheta1=theta(2)-alpha * sum( (X*theta-y) .* X(:,2)) / m;
    theta=[newtheta0;newtheta1];

    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

endfor

endfunction
