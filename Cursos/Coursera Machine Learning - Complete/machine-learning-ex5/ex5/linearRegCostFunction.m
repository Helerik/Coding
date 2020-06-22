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

h = X * theta;
for i = 1:m,
    J += (h(i) - y(i))^2;
endfor
sum = 0;
for i = 2:length(theta),
    sum += (theta(i))^2;
endfor
J = (J + lambda*sum)/(2*m);

for j = 1:length(theta),
    for i = 1:m,
        grad(j) += (h(i) - y(i))*X(i,j);
    endfor
endfor
grad(2:end) += lambda*theta(2:end);
grad /= m;









% =========================================================================

grad = grad(:);

end
