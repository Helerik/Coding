v = zeros(10,1);

for i = 1:10
    v(i) = 2^i;
endfor;
v

for i = v
    disp(i/2);
endfor

i = 1;
while i <= 5
    v(i) = 100;
    i += 1;
endwhile
v

i = 1;
while true
    v(i) = 999;
    i = i+1;
    if i == 6
        break;
    endif
endwhile
v

function retval =  func (x)
    retval = x^2;
endfunction
val = func(3)

function retval = func2 (x)
    retval = [x^2;x^3];
endfunction
V = func2(3)