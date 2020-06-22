import time
from math import sqrt

def sieve(n):
    size = n//2
    sieve = [1]*size
    limit = int(n**0.5)
    for i in range(1, limit):
        if sieve[i]:
            val = 2*i+1
            tmp = ((size-1) - i)//val 
            sieve[i+val::val] = [0]*tmp
    return [2] + [i*2+1 for i, v in enumerate(sieve) if v and i>0]

def prime_find(n):
    primes = [2]
    for num in range(3, n):
        is_prime = 1
        for p in primes:
            if p > sqrt(num):
                break
            if not (num%p):
                is_prime = 0
                break
        if is_prime:
            primes.append(num)
    return primes

t = time.time()
sieve(1000000000)
print(time.time() - t)













