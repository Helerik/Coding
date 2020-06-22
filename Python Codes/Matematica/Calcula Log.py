## Calculdora de logaritimo 

def main():
    while True:
        base = float(input("Base do logaritimo: "))
        valor = float(input("Valor do log a ser calculado: "))
        precisao = int(input("Numero de casas de precisao: "))
        resultado = 0
        j = 0
        
        while precisao != 0:
            i = 0
            while valor >= base**i:
                i += 1
            i -= 1

            valor = (valor/(base**i))**10
            
            
            resultado = resultado + i*(10**j)
            j -= 1
            precisao -= 1
        print(resultado)
        saida = str(input())
        if isinstance(saida, str):
            pass
        else: return

main()
