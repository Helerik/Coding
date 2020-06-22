/* EP 2 */

/* Funcoes auxiliares */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef char TipoDado;

typedef struct _RegPilha{
    TipoDado dado;
    struct _RegPilha *prox;
} RegPilha;

typedef RegPilha* Pilha;

typedef enum boolean {false,true} bool;

RegPilha *AlocaRegPilha(){
 RegPilha* q;
 q = (RegPilha*)calloc(1, sizeof(RegPilha));
 if(q==NULL) exit(-1);
 return q;
}

Pilha CriaPilha(){
    Pilha p;
    p = AlocaRegPilha();
    p->prox = NULL;
    return p;
}

void LiberaPilha(Pilha p){
    RegPilha *q,*t;
    q = p;
    while(q!=NULL){
        t = q;
        q = q->prox;
        free(t);
    }
}

bool PilhaVazia(Pilha p){
    return (p->prox==NULL);
}

void Empilha(Pilha p, TipoDado x){
    RegPilha *q;
    q = AlocaRegPilha();
    q->dado = x;
    q->prox = p->prox;
    p->prox = q;
}

TipoDado Desempilha(Pilha p){
    RegPilha *q;
    TipoDado x;
    q = p->prox;
    if(q==NULL) exit(-1);
    x = q->dado;
    p->prox = q->prox;
    free(q);
    return x;
}





int ValorExpressao(char prefixa[]){

    int i;
    int v1, v2, num;
    char c;
    Pilha p;
    p = CriaPilha();
    i = strlen(prefixa);
    printf ("i = %d", i);
    while (i > -1){
        c = prefixa[i];
        if (c >= '0' && c <= '9'){
            num  = c - '0';
            Empilha(p, num);
        }
        else if (c == '+'){
            v1 = Desempilha(p);
            v2 = Desempilha(p);
            printf("v1 = %d", v1);
            printf(" ");
            printf("v2 = %d", v2);
            printf(" ");
            num = v1 + v2;
            printf("num = %d", num);
            printf(" ");
            Empilha(p, num);
        }
        else if (c == '*'){
            v1 = Desempilha(p);
            v2 = Desempilha(p);
            printf("v1 = %d", v1);
            printf(" ");
            printf("v2 = %d", v2);
            printf(" ");
            num = v1 * v2;
            printf("num =  %d", num);
            printf(" ");
            Empilha(p, num);
        }
        i--;
    }
    num = Desempilha(p);
    return num;

}








int main(){
    char pre[512];
    int v;
    scanf("%s", pre);
    v = ValorExpressao(pre);
    printf("%d\n",v);
    return 0;
}






















