%Classification Problems:
%Examples: email: spam/not spam?
%          tumor: malign/benign?
%
% y = {0,1} -> 0 is "negative"; 1 is "positive"
%
% Option 1: Use linear regression
%           - Use a treshold: h(x) >= treshold -> y = 1
%             Else: y = 0
%           Not the best idea...
%
% New idea: Logistic Regression (Classification Algorithm)
%
% Hypothesis representation:
%
% h(x) = theta' * X -> h(x) = g(theta' * X),
% where g(z) = 1/(1+e^(-z)), the sigmoid function
%
% Interpretation of hypothesis:
% h(x) = 0.7 -> 70% chance of y = 1; 30% of y = 0.
%
% Decision Boundary:
% We could say, for example that if h(x) > 0.5, y = 1
% Else, y = 0
%
% Note: g(z) >= 0.5 <=> theta' * X >= 0.
% Basically, sets a line that separates a plane where y = 1 and y = 0.
% 
% Non-linear decision boundaries:
%
% Example: h(x) = g(theta3 * x1^2 + theta4 * x2^2
% Forms a circle or radius 1, where the outside is where y = 1.
%
% Defining a new cost function:
%
% J(theta) = 1/m * sum{i = 1:m}(cost(h(x[i], y))
% where cost = 1/2 * (h(x[i]) - y[i])^2
% Problem: cost might be non-convex, which is bad for finding global minimum
%
% Solution: cost = -log(h(x))   if y = 1
%                = -log(1-h(x)) if y = 0
% This way, the penalty for getting a wrong answer (i.e. predicting y = 0, when
% y = 1) will be very high.
%
% Simplified Cost function and Gradient Descent:

clear();

function retval = Cost(h, y),
    retval = -y*log(h) - (1-y)*log(1-h);
endfunction

function retval = CostFun(theta, X, y),
    h = theta' * X
    summ = 0
    for i = 1:length(y),
        summ += Cost(h(i), y(i))
    endfor
    retval = summ/m
endfunction

% We can prove that the gradient descent for the logistic regression is
% VERY similar to the linear regression gradient descent, with the exception
% that h(x) = 1/(1+e^(-theta' * X))














