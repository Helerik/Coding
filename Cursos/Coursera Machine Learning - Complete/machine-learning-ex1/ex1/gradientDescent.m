function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
n = length(theta);
J_history = zeros(num_iters, 1);

for iter = 1:num_iters,
    h = X*theta;
    del = zeros(1,n);
    theta_tmp = zeros(1,n);
    for i = 1:n,
        for j = 1:m,
            del(i) += (h(j) - y(j))*X(j, i);
        endfor
        theta_tmp(i) = theta(i) - (alpha/m) * del(i);
        theta(i) = theta_tmp(i) - (alpha/m) * del(i);


    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
