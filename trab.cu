#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include "clock_timer.h"

//Checagem de erros para as funções do CUDA
#define CUDA_SAFE_CALL(call) { \
    cudaError_t err = call;     \
    if(err != cudaSuccess) {    \
        fprintf(stderr,"Erro no arquivo '%s', linha %i: %s.\n", \
            __FILE__, __LINE__,cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); } }

#define TAM_BLOCO 32
#define ITER 1000
#define UO 0
#define UE 10
#define UN 5
#define US 5
#define PI 3.141592653589793

int N1;
int N2;

__constant__ double h1;
__constant__ double h2;

__device__ double a(double x, double y);

__device__ double a(double x, double y);

__device__ double o(int i, int j);

__device__ double b(double x, double y);

__device__ double e(int i, int j);

__device__ double o(int i, int j);

__device__ double n(int i, int j);

__device__ double e(int i, int j);

__device__ double s(int i, int j);


void geraMatriz(double *matriz, int N1, int N2);

void imprimeMatriz(double *a, int N1, int N2);

void testaResultado(double *resultado_gpu, double *resultado, int N1, int N2);

__global__ void gauss_seidel_gpu(double *atual, int N1, int N2, double w);

__global__ void gauss_seidel_gpu_par(double *atual, int N1, int N2, double w);

__global__ void gauss_seidel_gpu_impar(double *atual, int N1, int N2, double w);

__global__ void gauss_seidel_local_gpu_par(double *atual, int N1, int N2);

__global__ void gauss_seidel_local_gpu_impar(double *atual, int N1, int N2);


__global__ void gauss_seidel_gpu_par_shar(double *atual, int N1, int N2, double w);

__global__ void gauss_seidel_gpu_impar_shar(double *atual, int N1, int N2, double w);


int main(int argc, char **argv) {
    if (argc < 4) {
        printf("Digite: %s <N1> <N2> <w>\n", argv[0]);
        exit(EXIT_FAILURE);
    }
    N1 = atoi(argv[1]);
    N2 = atoi(argv[2]);
    double w = atof(argv[3]);
    double temp_h1 = 1.0 / (N1 - 1);
    double temp_h2 = 1.0 / (N2 - 1);
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(h1, &temp_h1, sizeof(double)));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(h2, &temp_h2, sizeof(double)));
    printf("N1 = %d, N2 = %d\n"
                   "h1 = %lf, h2 = %lf\n", N1, N2, temp_h1, temp_h2);
    dim3 threadsBloco(TAM_BLOCO, TAM_BLOCO);
    dim3 blocosGrade(N1 / threadsBloco.x, N2 / threadsBloco.y);
    double *matriz, *matriz_gpu, *matriz_gpu_volta;
    int matriz_bytes = N1 * N2 * sizeof(double);
    matriz = (double *) malloc(matriz_bytes);
    matriz_gpu_volta = (double *) malloc(matriz_bytes);

    geraMatriz(matriz, N1, N2);
    CUDA_SAFE_CALL(cudaMalloc((void **) &matriz_gpu, matriz_bytes));
    CUDA_SAFE_CALL(cudaMemcpy(matriz_gpu, matriz, matriz_bytes, cudaMemcpyHostToDevice));
    int tamMemoriaComp = 2 * TAM_BLOCO * TAM_BLOCO * sizeof(double);
    for (int k = 0; k < ITER; k++) {
        gauss_seidel_gpu_par << < blocosGrade, threadsBloco, tamMemoriaComp >> > (matriz_gpu, N1, N2, w);
        gauss_seidel_gpu_impar << < blocosGrade, threadsBloco, tamMemoriaComp >> > (matriz_gpu, N1, N2, w);
    }
    CUDA_SAFE_CALL(cudaMemcpy(matriz_gpu_volta, matriz_gpu, matriz_bytes, cudaMemcpyDeviceToHost));
    imprimeMatriz(matriz_gpu_volta, N1, N2);

    memset(matriz, '\0', N1 * N2 * sizeof(double));
    geraMatriz(matriz, N1, N2);
    CUDA_SAFE_CALL(cudaMemcpy(matriz_gpu, matriz, matriz_bytes, cudaMemcpyHostToDevice));

    for (int k = 0; k < ITER; k++) {
        gauss_seidel_local_gpu_par << < blocosGrade, threadsBloco, tamMemoriaComp >> > (matriz_gpu, N1, N2);
        gauss_seidel_local_gpu_impar << < blocosGrade, threadsBloco, tamMemoriaComp >> > (matriz_gpu, N1, N2);
    }

    CUDA_SAFE_CALL(cudaMemcpy(matriz_gpu_volta, matriz_gpu, matriz_bytes, cudaMemcpyDeviceToHost));




    return 0;
}

void testaResultado(double *resultado_gpu, double *resultado, int N1, int N2) {
    for (int i = 0; i < N1 * N2; i++) {
        if (abs(resultado_gpu[i] - resultado[i]) > 1e-5) {
            fprintf(stderr, "Resultado incorrento para o elemento de indice %d!\n", i);
            //exit(EXIT_FAILURE);
        }
    }
}

void imprimeMatriz(double *a, int N1, int N2) {
    int i, j;
    for (i = 0; i < N1; i++) {
        for (j = 0; j < N2; j++) {
            printf("%.5lf ", a[i * N1 + j]);
        }
        printf("\n");
    }
}

void geraMatriz(double *matriz, int N1, int N2) {
    for (int j = 1; j < N1 - 1; j++) {
        matriz[j] = US; // GERANDO CONTORNO PLACA INFERIOR
        matriz[(N1 - 1) * (N1) + j] = UN; // GERANDO CONTORNO PLACA TOPO
    }

    for (int i = 1; i < N2 - 1; i++) {
        matriz[i * (N1)] = UO; // GERANDO CONTORNO PLACA ESQUERDA
        matriz[i * (N1) + N2 - 1] = UE; // GERANDO CONTORNO PLACA DIREITA
    }


    //Resto do chute inicial vai ser o proprio lixo de memoria? Ainda pensar sobre...
    for (int i = 1; i < N1 - 1; i++) {
        for (int j = 1; j < N2 - 1; j++) {
            matriz[i * N1 + j] = (double) (UO + UE + US + UN) / 4.0;
        }
    }
}

__device__ double n(int i, int j) {
    return (2 - h2 * b(i * h1, j * h2)) / (4 * (1 + ((h2 * h2) / (h1 * h1))));
}

__device__ double s(int i, int j) {
    return (2 + h2 * b(i * h1, j * h2)) / (4 * (1 + ((h2 * h2) / (h1 * h1))));
}

__device__ double e(int i, int j) {
    return (2 - h1 * a(i * h1, j * h2)) / (4 * (1 + ((h1 * h1) / (h2 * h2))));
}

__device__ double o(int i, int j) {
    return (2 + h1 * a(i * h1, j * h2)) / (4 * (1 + ((h1 * h1) / (h2 * h2))));
}

__device__ double b(double x, double y) {
    return 500 * y * (1 - y) * (x - 0.5);
}


__device__ double a(double x, double y) {
    return 500 * x * (1 - x) * (0.5 - y);
}

__device__ double ro (int i, int j){
    return 2*((sqrt(e(i,j)*o(i,j))*cos(h1*PI)) + (sqrt(s(i,j)*n(i,j))*cos(h2*PI)));
}

__global__ void gauss_seidel_gpu_par(double *atual, int N1, int N2, double w) {

    //coordenadas globais da thread
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if ((((i + j) % 2) == 0) && i != 0 && j != 0 && i != (N1 - 1) && j != (N2 - 1)) {
        atual[i * N1 + j] = (1 - w) * atual[i * N1 + j] + w * (o(i, j) * atual[(i - 1) * N1 + j] +
                                                               e(i, j) * atual[(i + 1) * N1 + j] +
                                                               s(i, j) * atual[i * N1 + (j - 1)] +
                                                               n(i, j) * atual[i * N1 + (j + 1)]);

    }

}


__global__ void gauss_seidel_gpu_impar(double *atual, int N1, int N2, double w) {

    //coordenadas globais da thread
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if ((((i + j) % 2) == 1) && i != 0 && j != 0 && i != (N1 - 1) && j != (N2 - 1)) {
        atual[i * N1 + j] = (1 - w) * atual[i * N1 + j] + w * (o(i, j) * atual[(i - 1) * N1 + j] +
                                                               e(i, j) * atual[(i + 1) * N1 + j] +
                                                               s(i, j) * atual[i * N1 + (j - 1)] +
                                                               n(i, j) * atual[i * N1 + (j + 1)]);

    }
}

__global__ void gauss_seidel_gpu(double *atual, int N1, int N2, double w) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i != 0 && j != 0 && i != (N1 - 1) && j != (N2 - 1)) {
        if (((i + j) % 2) == 0) {
            atual[i * N1 + j] = (1 - w) * atual[i * N1 + j] + w * (o(i, j) * atual[(i - 1) * N1 + j] +
                                                                   e(i, j) * atual[(i + 1) * N1 + j] +
                                                                   s(i, j) * atual[i * N1 + (j - 1)] +
                                                                   n(i, j) * atual[i * N1 + (j + 1)]);
        }
        __syncthreads();

        if (((i + j) % 2) == 1) {
            atual[i * N1 + j] = (1 - w) * atual[i * N1 + j] + w * (o(i, j) * atual[(i - 1) * N1 + j] +
                                                                   e(i, j) * atual[(i + 1) * N1 + j] +
                                                                   s(i, j) * atual[i * N1 + (j - 1)] +
                                                                   n(i, j) * atual[i * N1 + (j + 1)]);
        }


    }

}


__global__ void gauss_seidel_gpu_impar_shar(double *atual, int N1, int N2, double w) {

    //coordenadas globais da thread
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
//coordenadas locais da thread
    int i_bloco = threadIdx.x;
    int j_bloco = threadIdx.y;
    extern __shared__ double mat_sub[];
//mem´oria compartilhada para a submatriz de A
    double *Asub = (double *) mat_sub;

    for (int passo = 0; passo < N2; passo += blockDim.y) {
        Asub[i_bloco * blockDim.y + j_bloco] =
                atual[i * N1 + passo + j_bloco];
        __syncthreads();
    }
    if ((((i + j) % 2) == 1) && i != 0 && j != 0 && i != N1 && j != N2) {
        atual[i * N1 + j] = (1 - w) * Asub[i * N1 + j] + w * (o(i, j) * Asub[(i - 1) * N1 + j] +
                                                              e(i, j) * Asub[(i + 1) * N1 + j] +
                                                              s(i, j) * Asub[i * N1 + (j - 1)] +
                                                              n(i, j) * Asub[i * N1 + (j + 1)]);

    }
}

__global__ void gauss_seidel_gpu_par_shar(double *atual, int N1, int N2, double w) {

    //coordenadas globais da thread
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
//coordenadas locais da thread
    int i_bloco = threadIdx.x;
    int j_bloco = threadIdx.y;
    extern __shared__ double mat_sub[];
//mem´oria compartilhada para a submatriz de A
    double *Asub = (double *) mat_sub;
    for (int passo = 0; passo < N2; passo += blockDim.y) {
        Asub[i_bloco * blockDim.y + j_bloco] =
                atual[i * N1 + passo + j_bloco];
        __syncthreads();
    }
    if ((((i + j) % 2) == 0) && i != 0 && j != 0 && i != N1 && j != N2) {
        atual[i * N1 + j] = (1 - w) * Asub[i * N1 + j] + w * (o(i, j) * Asub[(i - 1) * N1 + j] +
                                                              e(i, j) * Asub[(i + 1) * N1 + j] +
                                                              s(i, j) * Asub[i * N1 + (j - 1)] +
                                                              n(i, j) * Asub[i * N1 + (j + 1)]);

    }

}

__global__ void gauss_seidel_local_gpu_par(double *atual, int N1, int N2){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if ((((i + j) % 2) == 0) && i != 0 && j != 0 && i != (N1 - 1) && j != (N2 - 1)) {
        double p = ro(i,j);
        double w = 2 / (1 + sqrt(1 - (p*p)));
        atual[i * N1 + j] = (1 - w) * atual[i * N1 + j] + w * (o(i, j) * atual[(i - 1) * N1 + j] +
                                                               e(i, j) * atual[(i + 1) * N1 + j] +
                                                               s(i, j) * atual[i * N1 + (j - 1)] +
                                                               n(i, j) * atual[i * N1 + (j + 1)]);

    }
}

__global__ void gauss_seidel_local_gpu_impar(double *atual, int N1, int N2){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if ((((i + j) % 2) == 1) && i != 0 && j != 0 && i != (N1 - 1) && j != (N2 - 1)) {
        double p = ro(i,j);
        double w = 2 / (1 + sqrt(1 - (p*p)));
        atual[i * N1 + j] = (1 - w) * atual[i * N1 + j] + w * (o(i, j) * atual[(i - 1) * N1 + j] +
                                                               e(i, j) * atual[(i + 1) * N1 + j] +
                                                               s(i, j) * atual[i * N1 + (j - 1)] +
                                                               n(i, j) * atual[i * N1 + (j + 1)]);

    }
}