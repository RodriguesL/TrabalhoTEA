#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include "clock_timer.h"

//Checagem de erros para as funções do CUDA
/*#define CUDA_SAFE_CALL(call) { \
	cudaError_t err = call;     \
	if(err != cudaSuccess) {    \
		fprintf(stderr,"Erro no arquivo '%s', linha %i: %s.\n", \
			__FILE__, __LINE__,cudaGetErrorString(err)); \
		exit(EXIT_FAILURE); } }
*/
#define TAM_BLOCO 32
#define ITER 500
#define UO 0
#define UE 10
#define UN 5
#define US 5

int N1;
int N2;

double h1;
double h2;

double a(double x, double y) {
    return ITER*x*(1 - x)*(0.5 - y);
}

double b(double x, double y) {
    return ITER*y*(1-y)*(x - 0.5);
}
double o (int i, int j){
    return (2 + h1*a(i*h1, j*h2))/4*(1 + ((h1*h1)/(h2*h2)));
}

double e (int i, int j) {
    return (2 - h1*a(i*h1, j*h2))/4*(1 + ((h1*h1))/(h2*h2));
}

double s (int i, int j) {
    return (2 + h2*b(i*h1, j*h2))/4*(1 + ((h2*h2)/(h1*h1)));
}

double n (int i, int j) {
    return (2 - h2*b(i*h1, j*h2))/4*(1 + (h2*h2/h1*h1));
}

void geraMatriz(double *matriz, int N1, int N2) {
    for (int j = 1; j < N1-1 ; j++) {
        matriz[j] = US; // GERANDO CONTORNO PLACA INFERIOR
        matriz[(N1-1)*(N1) + j] = UN; // GERANDO CONTORNO PLACA TOPO
    }

    for (int i = 1; i < N2-1 ; i++) {
        matriz[i*(N1)] = UO; // GERANDO CONTORNO PLACA ESQUERDA
        matriz[i*(N1) + N2-1] = UE; // GERANDO CONTORNO PLACA DIREITA
    }


    //Resto do chute inicial vai ser o proprio lixo de memoria? Ainda pensar sobre...
    for (int i=1;i<N1-1;i++) {
        for(int j=1;j<N2-1;j++) {
            matriz[i*N1 + j] = (UO+UE+US+UN)/4;
        }
    }
}

void gauss_seidel_seq(double* atual, int N1, int N2, double w) {
    for (int k = 0; k < ITER; k++) {
        for (int i = 1; i < (N1+1); i++) {
            for (int j = 1; j < (N2+1); j++) {
                    atual[i*(N1+2) + j] = (1 - w)*atual[i*(N1+2) + j] + w*(o(i,j)*atual[(i-1)*(N1+2) + j] +
                            e(i,j)*atual[(i+1)*(N1+2) + j] + s(i,j)*atual[i*(N1+2) + (j - 1)] + n(i,j)*atual[i*(N1+2) + (j+1)]);
            }
        }
    }
}

void gauss_seidel_local_seq(double *matriz) {

}

void imprimeMatriz(double *a, int N1, int N2) {
    int i, j;
    for (i = 0 ; i < N1 + 2; i++) {
        for (j = 0; j < N2 + 2; j++) {
            printf("%.2lf ", a[i*(N1+2)+j]);
        }
        printf("\n");
    }
}

int main (int argc, char** argv) {
    if (argc < 3) {
        printf("Digite: %s <N1> <N2>\n", argv[0]);
        exit(EXIT_FAILURE);
    }
    N1 = atoi(argv[1]);
    N2 = atoi(argv[2]);
    h1 = 1/(N1 + 2);
    h2 = 1/(N2 + 2);
    double *m;
    m = (double *) calloc((N1+2)*(N2+2), sizeof(double));
    printf("N1 = %d, N2 = %d\n"
            "h1 = %lf, h2 = %lf\n", N1, N2, h1, h2);
    geraMatriz(m, N1, N2);
    imprimeMatriz(m, N1, N2);
    printf("\n\n\n\n");
    gauss_seidel_seq(m, N1, N2, 1);
    imprimeMatriz(m, N1, N2);

    return 0;
}