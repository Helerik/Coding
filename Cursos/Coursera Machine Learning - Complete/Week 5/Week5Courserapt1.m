% Neural Network Cost Function

clear()

function retval = costFunc(theta, X, y, lambda),
    m = length(y)
    
    h = sigmoid(X * theta)
    sum = 0;
    for i = 1:m,
        for k = 1:K,
            sum += y(k,i) * log(h(k,i)) + (1 - y(k,i)) * loh(1 - h(k,i));
        endfor
    endfor
    sum /= (-1/m);
    retval = sum;
    sum = 0;
    for l = 1:L-1,
        for i = 1:sl,
            for j = 1:sl+1,
                sum += (theta(j, i, l))^2;
            endfor
        endfor
    endfor
    sum /= (-lambda/(2*m));
    retval += sum;
endfunction

% Backpropagation:



