class fraction:


    ##reduz a fracao para sua forma mais simplificada
    def _reduce(self, n,d, sign):
        p = abs(n)
        q = abs(d)
        while p % q != 0:
            pp = p
            qq = q
            p = qq
            q = pp % qq
        n = abs(n)//q * sign
        d = abs(d)//q
        return(n, d)

    def getnumerador(self):
        return self._numerador

    def getdenominador(self):
        return self._denominador
    
    def __init__(self, numerador, denominador):
        
        if (not isinstance(numerador, int)):
            raise (TypeError("O numerador deve ser inteiro, ou uma fracao"))
        if (not isinstance(denominador, int)):
            raise (TypeError("O denominador deve ser inteiro, ou uma fracao"))

        if denominador == 0:
            raise ZeroDivisionError("O denominador da fracao nao pode ser zero")

        if numerador == 0:
            self._numerador = 0
            self._denominador = 1
        else:
            if (numerador < 0 and denominador >= 0) or (numerador >=0 and denominador <0):
                sign = -1
            else:
                sign = 1
            (self._numerador, self._denominador) = self._reduce(numerador, denominador, sign)

    def __repr__(self):
        return str(self._numerador)+"/"+str(self._denominador)

    def __eq__(self, direita):
        esquerda = self
        if esquerda.getnumerador() == direita.getnumerador() and esquerda.getdenominador() == direita.getdenominador():
            return True
        else:
            return False

    def __ne__(self, direita):
        esquerda = self
        return not esquerda == direita

    def __lt__(self, direita):
        esquerda = self
        return (esquerda.getnumerador() * direita.getdenominador()) < (esquerda.getdenominador() * direita.getnumerador())

    def __le__(self, direita):
        esquerda = self
        return not direita < esquerda

    def __gt__(self, direita):
        esquerda = self
        return direita < esquerda

    def __ge__(self, direita):
        esquerda = self
        return not direita > esquerda

    def __add__(self, direita):
        esquerda = self
        num = esquerda.getnumerador() * direita.getdenominador() + direita.getnumerador() * esquerda.getdenominador()
        den = direita.getdenominador() * esquerda.getdenominador()
        return fraction(num,den)

    def __sub__(self, direita):
        esquerda = self
        num = esquerda.getnumerador() * direita.getdenominador() - direita.getnumerador() * esquerda.getdenominador()
        den = direita.getdenominador() * esquerda.getdenominador()
        return fraction(num,den)

    def __mul__(self, direita):
        esquerda = self
        num = esquerda.getnumerador() * direita.getnumerador()
        den = esquerda.getdenominador() * direita.getdenominador()
        return fraction(num,den)

    def __truediv__(self, direita):
        esquerda = self
        num = esquerda.getnumerador() *  direita.getdenominador()
        den = esquerda.getdenominador() * direita.getnumerador()
        return fraction(num,den)

    def __abs__(self):
        num = abs(self.getnumerador())
        den = abs(self.getdenominador())
        return fraction(num, den)


a = fraction(fraction(2,3),100000)

print(a)


            
