clear()

A = [1,2;
     3,4;
     5,6];
B = [11,12;
     13,14;
     15,16];
C = [1,1 ; 2,2];

A*C

A*transpose(B)

A .* B %elementwise multiplication

A .^ 2 %elementwise power

log(C)

V = -C

abs(V)

V = [1;2;3]

V + ones(length(V), 1)

A*B' %B transpose

a = [1,2,3,4,5,90,45]

max(a)
min(a)

max(A) %maximum of each column
max(A') %maximum of each row

find(a>3) %indexes where a>3

find(2<A<4)

prod(a)

A = magic(3)

max(A,[],1) %maximum per column
max(A,[],2) %maximum per line

max(A(:)) %maximum number in a

A = magic(9)

sum(A, 1)
sum(A, 2)

% diagonal multiplication

A .* eye(9)
sum(sum(A .* eye(9)))
A .* flipud(eye(9))

A = magic(3)
pinv(A)
round(A*pinv(A))















