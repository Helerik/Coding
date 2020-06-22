clear()

% Multiclass Classification: Objective: classify on more than one tag
% i.e. y = {1,2,3,4, ..., n}
% One vs. All classification
% Idea: find 1 by 1 the diferent classes.
% So we use multiple classifiers h(x)[i].
% choose i that makes max_i h(x)[i] (i.e. argmax)
%
% Overfitting:
%
% The curve "fits too well". The problem is that the model wont be able to 
% generalize enough for new examples.
%
% Ways to adress overfitting:
% - Reduce amount of features (Model selection algorithm (MQMQ?))
% - Regularization:
%   -> Keep features but reduce magnitude
%   -> Works well with many features
%
% Regularization and Cost Function:
%
% min cost_function + "big number" * theta{*},
% where theta{*} are he ones that are making the function overfit.
%
% New cost function J(theta):
% 1/2m * [sum_{i=1:m}((h(i) - y(i))^2) - lambda * sum_{j=1:n}(theta(j)^2)]
%
% Gradient Descent with regularization:
% theta[0] remains the same
% theta[j] will have an added (lambda/n) * theta[j]
% theta[j] = (1-alpha*lambda/m)*theta[j] + the usual stuff.
%
% Normal Equation:
% theta = (X'*X + lambda*eye(n+1)(but 1st line = 0))^-1 * X'y
% if lambda > 0 => no problem regarding invertibility!
%
% Logistic regression regularization:
% J = usual stuff + 1/2m * lambda * sum_{j = 1:n}(theta[j]^2)
% The gradient descent will look just like the one for linear regression
% with regularization.
%
% Advanced Optimization:
% (not complete code, won't run like this!!)

function [jVal, gradient] = costFunc(theta),
    jVal = J(theta);
    gradient = zeros(length(theta), 1)
    for i = 1:length(theta),
        gradient(i) = d/dtheta(J(theta))
    endfor
endfunction











