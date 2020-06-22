function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));
h = sigmoid(X*theta);

for i = 1:m,
    J += -y(i)*log(h(i)) - (1-y(i))*log(1-h(i));
endfor
J /= m;

for j = 1:length(theta),
    for i = 1:m,
        grad(j) += (h(i) - y(i))*X(i,j);
    endfor
endfor
grad /= m;

% =============================================================

end
