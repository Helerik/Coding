function g = sigmoid(z)
%SIGMOID Compute sigmoid function
%   g = SIGMOID(z) computes the sigmoid of z.

% You need to return the following variables correctly 
g = zeros(size(z));

for i = 1:size(g)(1),
    for j = 1:size(g)(2),
        g(i,j) = 1/(1 + e^(-z(i,j)));
    endfor
endfor

% =============================================================

end
