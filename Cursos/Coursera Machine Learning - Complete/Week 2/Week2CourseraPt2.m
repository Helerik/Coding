A = [2,3,4;
     5,6,7];
     
disp(size(A))
disp(length(A)) %highest A dimension

V = [2;3;4;5;6];

disp("")
disp(length(V))
disp("")
% load files:

pwd 
ls

% load somearchive.xxx

save hello.txt V;

clear V;

load hello.txt

disp(V)
disp('')

disp(A(1,1))
disp(A(1,2))

disp(A(:,1))
disp(A(1,:))
disp("")
disp(A(:,[1 3]))
disp("")

A(:, 2) = [10; 9]

disp('')

A = [A, [1;2]] %add column

A = [A;[1,2,3,4]] %add line

A(:)

B = 2*A

C = [A B]
C = [C;C;C]

D = [2,3,4;
     5,6,7;];
     
D_ = [D eye(2)]



























