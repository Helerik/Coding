a = 3; %output supression ;
b = 2;

a = pi;

disp(a); %disp as in display

disp(sprintf('2 decimals: %0.2f', a));
disp(sprintf('6 decimals: %0.6f', a));

A = [1,2;
     3,4];
disp("")     
disp(A)
disp("")
disp(A)
disp("")

V = [1;2;3];
U = [1,2,3];

disp(V)
disp("")
disp(U)
disp("")

C = ones(1,3)
D = ones(2,3)
E = 2*D

disp("")

W = rand(2,2) %uniform random
UU = randn(2,2) %normal random

w = rand(1,1)

VV = randn(1000,1);

hist(VV)
hist(VV, 50)

I = eye(4)
plot(VV)

K = [2;3;1]
disp(K+1) %equivalent to summing to a only 1's vector


















