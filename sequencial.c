#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include "clock_timer.h"

#define ITER 1000
#define UO 0
#define UE 10
#define UN 5
#define US 5
#define PI 3.141592653589793

double h1, h2;
double a(double x, double y);
double b(double x, double y);
double o (int i, int j);
double n (int i, int j);
double e (int i, int j);
double s (int i, int j);

void gauss_seidel_seq(double* atual, int N1, int N2, double w);
void geraMatriz(double *matriz, int N1, int N2);
void imprimeMatriz(double *a, int N1, int N2);
void gauss_seidel_local_seq(double *atual, int N1, int N2);
void testaResultado(double *resultado_local, double *resultado, int N1, int N2);

int main (int argc, char** argv) {
    if (argc < 4) {
        printf("Digite: %s <N1> <N2> <w>\n", argv[0]);
        exit(EXIT_FAILURE);
    }
    int N1 = atoi(argv[1]);
    int N2 = atoi(argv[2]);
    double w = atof(argv[3]);
    h1 = 1/(N1 + 1);
    h2 = 1/(N2 + 1);
    double *matriz, *matriz_local;
    matriz = (double *)malloc(N1*N2*sizeof(double));
    matriz_local = (double *)malloc(N1*N2*sizeof(double));
    geraMatriz(matriz, N1, N2);
    geraMatriz(matriz_local, N1, N2);
    gauss_seidel_seq(matriz, N1, N2, w);
    imprimeMatriz(matriz, N1, N2);
    printf("\n\n\n\n");
    gauss_seidel_local_seq(matriz_local, N1, N2);
    imprimeMatriz(matriz_local, N1, N2);
    testaResultado(matriz_local, matriz, N1, N2);
    return 0;
}

void testaResultado(double *resultado_local, double *resultado, int N1, int N2) {
    for (int i = 0; i < N1*N2; i++) {
        if (abs(resultado_local[i] - resultado[i]) > 1e-5) {
            fprintf(stderr, "Resultado incorreto para o elemento de indice %d!\n", i);
            exit(EXIT_FAILURE);
        }
    }
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
            matriz[i*N1 + j] = (double)(UO+UE+US+UN)/4.0;
        }
    }
}

double n (int i, int j) {
    return (2 - h2*b(i*h1, j*h2))/(4*(1 + ((h2*h2)/(h1*h1))));
}

double s (int i, int j) {
    return (2 + h2*b(i*h1, j*h2))/(4*(1 + ((h2*h2)/(h1*h1))));
}

double e (int i, int j) {
    return (2 - h1*a(i*h1, j*h2))/(4*(1 + ((h1*h1)/(h2*h2))));
}

double o (int i, int j){
    return (2 + h1*a(i*h1, j*h2))/(4*(1 + ((h1*h1)/(h2*h2))));
}

double b(double x, double y) {
    return 500*y*(1 - y)*(x - 0.5);
}


double a(double x, double y) {
    return 500*x*(1 - x)*(0.5 - y);
}

void gauss_seidel_local_seq(double *atual, int N1, int N2) {
    for (int k = 0; k < ITER; k++) {
        for (int i = 1; i < (N1 - 1); i++) {
            for (int j = 1; j < (N2 - 1); j++) {
                double ro = 2*((sqrt(e(i,j)*o(i,j))*cos(h1*PI)) + (sqrt(s(i,j)*n(i,j))*cos(h2*PI)));
                double omega = 2/(1 + sqrt(1 - ro*ro));
                atual[i*N1 + j] = (1 - omega)*atual[i*N1 + j] + omega*(o(i,j)*atual[(i-1)*N1 + j] +
                                                                       e(i,j)*atual[(i+1)*N1 + j] + s(i,j)*atual[i*N1 + (j - 1)] + n(i,j)*atual[i*N1 + (j+1)]);
            }
        }
    }
}

void gauss_seidel_seq(double* atual, int N1, int N2, double w) {
    for (int k = 0; k < ITER; k++) {
        for (int i = 1; i < (N1-1); i++) {
            for (int j = 1; j < (N2-1); j++) {
                atual[i*N1 + j] = (1 - w)*atual[i*N1 + j] + w*(o(i,j)*atual[(i-1)*N1 + j] +
                                                               e(i,j)*atual[(i+1)*N1 + j] + s(i,j)*atual[i*N1 + (j - 1)] + n(i,j)*atual[i*N1 + (j+1)]);
            }
        }
    }
}

void imprimeMatriz(double *a, int N1, int N2) {
    int i, j;
    for (i = 0 ; i < N1; i++) {
        for (j = 0; j < N2; j++) {
            printf("%.2lf ", a[i*N1+j]);
        }
        printf("\n");
    }
}