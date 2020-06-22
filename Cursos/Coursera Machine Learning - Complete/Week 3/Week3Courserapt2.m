clear()

% Advanced Optimization:
% Gradient Descent (currently studying)
% -> Gradient conjugate
% -> BFGS
% -> L-BFGS
% The last three are complex, however, they dont require the manual choice of
% alpha and they are also faster
%
% (Won't go deep in details about these algorithms, but will show how to 
%  use the built in functions of octave to optimize)

function [jVal, gradient] = costFunction(theta),
    jVal = (theta(1) - 5)^2 + (theta(2) - 5)^2;
    gradient = zeros(2,1);
    gradient(1) = 2*(theta(1) - 5);
    gradient(2) = 2*(theta(2) - 5);
endfunction

options = optimset('GradObj', 'on', 'MaxIter', '100'); # in this setting it is
initialTheta = zeros(2,1);                             # assumed we gave the 
                                                       # gradient in costFunc...
[optTheta, functionVal, exitFlag] = ...
fminunc(@costFunction, initialTheta, options)


