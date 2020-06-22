function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));
h = sigmoid(X*theta);

theta_tmp = theta();
theta_tmp(1) = 0;

for i = 1:m,
    J += -y(i)*log(h(i)) - (1-y(i))*log(1-h(i));
endfor
J += lambda*sum(theta_tmp.^2)/2;
J /= m;

for j = 1:length(theta),
    for i = 1:m,
        grad(j) += (h(i) - y(i))*X(i,j);
    endfor
endfor
grad += lambda*theta_tmp;
grad /= m;

% =============================================================

end
