%Exaple with the linear regression:'

Houses = [1, 2104; %house sizes
          1, 1416;
          1, 1534;
          1,  852]
          
theta = [-40; %parameters
         0.25]
         
Mat = Houses*theta %predictions

% note that this type of calculation is faster (!) than using a for loop (in
% ANY programming language. At least for large amounts of data.)

I = eye(2)
J = eye(3)

A = [1,2;
     4,5]

A_1 = inv(A)

A*A_1