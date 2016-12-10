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

__global__ void gauss_seidel_local_gpu_impar_shar(double *atual, int N1, int N2);

__global__ void gauss_seidel_local_gpu_par_shar(double *atual, int N1, int N2);


int main(int argc, char **argv) {
    if (argc < 4) {
        printf("Digite: %s <N1> <N2> <w>\n", argv[0]);
        exit(EXIT_FAILURE);
    }
    N1 = atoi(argv[1]);
    N2 = atoi(argv[2]);
    float tempo_kernel, tempo_kernel_local, tempo_kernel_shared, tempo_kernel_local_shared;
    double w = atof(argv[3]);
    double inicio, fim, tempo_ida, tempo_volta;
    double temp_h1 = 1.0 / (N1 - 1);
    double temp_h2 = 1.0 / (N2 - 1);
    cudaEvent_t start, stop;
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(h1, &temp_h1, sizeof(double)));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(h2, &temp_h2, sizeof(double)));

    dim3 threadsBloco(TAM_BLOCO, TAM_BLOCO);
    dim3 blocosGrade(N1 / threadsBloco.x, N2 / threadsBloco.y);
    double *matriz, *matriz_gpu, *matriz_gpu_volta;
    int matriz_bytes = N1 * N2 * sizeof(double);
    matriz = (double *) malloc(matriz_bytes);
    matriz_gpu_volta = (double *) malloc(matriz_bytes);

    geraMatriz(matriz, N1, N2);
    GET_TIME(inicio);
    CUDA_SAFE_CALL(cudaMalloc((void **) &matriz_gpu, matriz_bytes));
    CUDA_SAFE_CALL(cudaMemcpy(matriz_gpu, matriz, matriz_bytes, cudaMemcpyHostToDevice));
    GET_TIME(fim);
    tempo_ida = fim - inicio;
    int tamMemoriaComp = 2 * TAM_BLOCO * TAM_BLOCO * sizeof(double);

    CUDA_SAFE_CALL(cudaEventCreate(&start));
    CUDA_SAFE_CALL(cudaEventCreate(&stop));
    CUDA_SAFE_CALL(cudaEventRecord(start));
    for (int k = 0; k < ITER; k++) {
        gauss_seidel_gpu_par << < blocosGrade, threadsBloco, tamMemoriaComp >> > (matriz_gpu, N1, N2, w);
        gauss_seidel_gpu_impar << < blocosGrade, threadsBloco, tamMemoriaComp >> > (matriz_gpu, N1, N2, w);
    }
    CUDA_SAFE_CALL(cudaGetLastError());
    CUDA_SAFE_CALL(cudaEventRecord(stop)); // Lembra o tempo de fim
    CUDA_SAFE_CALL(cudaEventSynchronize(stop));
    tempo_kernel = 0;
    CUDA_SAFE_CALL(cudaEventElapsedTime(&tempo_kernel, start, stop));
    tempo_kernel /= 1000; // (tempo em seg)

    memset(matriz, '\0', N1 * N2 * sizeof(double));
    geraMatriz(matriz, N1, N2);
    CUDA_SAFE_CALL(cudaMemcpy(matriz_gpu, matriz, matriz_bytes, cudaMemcpyHostToDevice));

    CUDA_SAFE_CALL(cudaEventCreate(&start));
    CUDA_SAFE_CALL(cudaEventCreate(&stop));
    CUDA_SAFE_CALL(cudaEventRecord(start));
    for (int k = 0; k < ITER; k++) {
        gauss_seidel_local_gpu_par << < blocosGrade, threadsBloco, tamMemoriaComp >> > (matriz_gpu, N1, N2);
        gauss_seidel_local_gpu_impar << < blocosGrade, threadsBloco, tamMemoriaComp >> > (matriz_gpu, N1, N2);
    }
    CUDA_SAFE_CALL(cudaGetLastError());
    CUDA_SAFE_CALL(cudaEventRecord(stop)); // Lembra o tempo de fim
    CUDA_SAFE_CALL(cudaEventSynchronize(stop));
    tempo_kernel_local = 0;
    CUDA_SAFE_CALL(cudaEventElapsedTime(&tempo_kernel_local, start, stop));
    tempo_kernel_local /= 1000; // (tempo em seg)

    CUDA_SAFE_CALL(cudaMemcpy(matriz_gpu_volta, matriz_gpu, matriz_bytes, cudaMemcpyDeviceToHost));
    imprimeMatriz(matriz_gpu_volta, N1, N2);

    memset(matriz, '\0', N1 * N2 * sizeof(double));
    geraMatriz(matriz, N1, N2);
    CUDA_SAFE_CALL(cudaMemcpy(matriz_gpu, matriz, matriz_bytes, cudaMemcpyHostToDevice));

    CUDA_SAFE_CALL(cudaEventCreate(&start));
    CUDA_SAFE_CALL(cudaEventCreate(&stop));
    CUDA_SAFE_CALL(cudaEventRecord(start));
    for (int k = 0; k < ITER; k++) {
        gauss_seidel_gpu_par_shar << < blocosGrade, threadsBloco, tamMemoriaComp >> > (matriz_gpu, N1, N2, w);
        gauss_seidel_gpu_impar_shar << < blocosGrade, threadsBloco, tamMemoriaComp >> > (matriz_gpu, N1, N2, w);
    }
    CUDA_SAFE_CALL(cudaGetLastError());
    CUDA_SAFE_CALL(cudaEventRecord(stop)); // Lembra o tempo de fim
    CUDA_SAFE_CALL(cudaEventSynchronize(stop));
    tempo_kernel_shared = 0;
    CUDA_SAFE_CALL(cudaEventElapsedTime(&tempo_kernel_shared, start, stop));
    tempo_kernel_shared /= 1000; // (tempo em seg)

    memset(matriz, '\0', N1 * N2 * sizeof(double));
    geraMatriz(matriz, N1, N2);
    CUDA_SAFE_CALL(cudaMemcpy(matriz_gpu, matriz, matriz_bytes, cudaMemcpyHostToDevice));

    CUDA_SAFE_CALL(cudaEventCreate(&start));
    CUDA_SAFE_CALL(cudaEventCreate(&stop));
    CUDA_SAFE_CALL(cudaEventRecord(start));
    for (int k = 0; k < ITER; k++) {
        gauss_seidel_local_gpu_par_shar << < blocosGrade, threadsBloco, tamMemoriaComp >> > (matriz_gpu, N1, N2);
        gauss_seidel_local_gpu_impar_shar << < blocosGrade, threadsBloco, tamMemoriaComp >> > (matriz_gpu, N1, N2);
    }
    CUDA_SAFE_CALL(cudaGetLastError());
    CUDA_SAFE_CALL(cudaEventRecord(stop)); // Lembra o tempo de fim
    CUDA_SAFE_CALL(cudaEventSynchronize(stop));
    tempo_kernel_local_shared = 0;
    CUDA_SAFE_CALL(cudaEventElapsedTime(&tempo_kernel_local_shared, start, stop));
    tempo_kernel_local_shared /= 1000; // (tempo em seg)

    double* matriz_teste = (double *) malloc(matriz_bytes);
    GET_TIME(inicio);
    CUDA_SAFE_CALL(cudaMemcpy(matriz_teste, matriz_gpu, matriz_bytes, cudaMemcpyDeviceToHost));
    GET_TIME(fim);
    tempo_volta = fim - inicio;
    printf("\n\n\n\n\n\n\n\n\n\n");
    imprimeMatriz(matriz_teste, N1, N2);
    testaResultado(matriz_gpu_volta, matriz_teste, N1, N2);

    printf(
            "Matriz de %dx%d\n"
            "Blocos de %dx%d\n"
            "Tempo ida = %.6fs\n"
            "Tempo volta = %.6fs\n"
            "Tempo kernel = %.6fs\n"
            "Tempo kernel shared = %.6fs\n"
            "Tempo kernel local = %.6fs\n"
            "Tempo kernel local shared = %.6fs\n"
            "Tempo total de GPU = %.6fs\n"
            "Tempo total de GPU shared = %.6fs\n"
            "Tempo total de GPU local = %.6fs\n"
            "Tempo total de GPU local shared = %.6fs\n",
            N1, N2,
            threadsBloco.x, threadsBloco.y,
            tempo_ida,
            tempo_volta,
            tempo_kernel,
            tempo_kernel_shared,
            tempo_kernel_local,
            tempo_kernel_local_shared,
            tempo_ida+tempo_kernel+tempo_volta,
            tempo_ida+tempo_kernel_shared+tempo_volta,
            tempo_ida+tempo_kernel_local+tempo_volta,
            tempo_ida+tempo_kernel_local_shared+tempo_volta
    );

    return 0;
}

void testaResultado(double *resultado_gpu, double *resultado, int N1, int N2) {
    for (int i = 0; i < N1 * N2; i++) {
        if (abs(resultado_gpu[i] - resultado[i]) > 1e-1) {
            printf("Resultado incorreto para o elemento de indice %d!\n", i);
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

    if ((((i + j) % 2) == 0) && i > 0 && j > 0 && i != (N1 - 1) && j != (N2 - 1)) {
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

    if ((((i + j) % 2) == 1) && i > 0 && j > 0 && i != (N1 - 1) && j != (N2 - 1)) {
        atual[i * N1 + j] = (1 - w) * atual[i * N1 + j] + w * (o(i, j) * atual[(i - 1) * N1 + j] +
                                                               e(i, j) * atual[(i + 1) * N1 + j] +
                                                               s(i, j) * atual[i * N1 + (j - 1)] +
                                                               n(i, j) * atual[i * N1 + (j + 1)]);

    }
}

__global__ void gauss_seidel_gpu_impar_shar(double *atual, int N1, int N2, double w) {

    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    //coordenadas globais da thread
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;

    int i_bloco = threadIdx.x + 1;
    int j_bloco = threadIdx.y + 1;

    __shared__ double Asub[TAM_BLOCO+2][TAM_BLOCO+2];


    //for (int passo = 0; passo < N2; passo += TAM_BLOCO) {
    Asub[i_bloco][ j_bloco] = atual[i * N1 + j];

    if(i_bloco == 1) Asub[i_bloco - 1 ][ j_bloco] = atual[(i-1) * N1 + j];
    if(i_bloco == TAM_BLOCO) Asub[i_bloco + 1 ][ j_bloco] = atual[(i+1) * N1 + j];
    if(j_bloco == 1) Asub[i_bloco ][ j_bloco - 1] = atual[i * N1 + j - 1 ];
    if(j_bloco == TAM_BLOCO) Asub[i_bloco ][ j_bloco + 1] = atual[i * N1 + j + 1 ];



    __syncthreads();
    //}
    if ((((i + j) % 2) == 1) && i > 0 && j > 0 && i < (N1 - 1) && j < (N2 - 1) ) {
        
        atual[i * N1 + j] = (1 - w) * Asub[i_bloco][ j_bloco] + w * (o(i, j) * Asub[(i_bloco - 1)][ j_bloco] +
                                                              e(i, j) * Asub[(i_bloco + 1)][ j_bloco] +
                                                              s(i, j) * Asub[i_bloco][(j_bloco - 1)] +
                                                              n(i, j) * Asub[i_bloco][ (j_bloco + 1)]);

    }


}

__global__ void gauss_seidel_gpu_par_shar(double *atual, int N1, int N2, double w) {

    //coordenadas globais da thread
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
//coordenadas locais da thread
    int i_bloco = threadIdx.x + 1;
    int j_bloco = threadIdx.y + 1;

    __shared__ double Asub[TAM_BLOCO+2][TAM_BLOCO+2];

    Asub[i_bloco][j_bloco] = atual[i * N1 + j];
    if(i_bloco == 1) Asub[i_bloco - 1 ][ j_bloco] = atual[(i-1) * N1 + j];
    if(i_bloco == TAM_BLOCO) Asub[i_bloco + 1 ][ j_bloco] = atual[(i+1) * N1 + j];
    if(j_bloco == 1) Asub[i_bloco ][ j_bloco - 1] = atual[i * N1 + j - 1 ];
    if(j_bloco == TAM_BLOCO) Asub[i_bloco ][ j_bloco + 1] = atual[i * N1 + j + 1 ];

    __syncthreads();
    if ((((i + j) % 2) == 0) && i > 0 && j > 0  && i < (N1 - 1) && j < (N2 - 1) ) {
        atual[i * N1 + j] = (1 - w) * Asub[i_bloco][j_bloco] + w * (o(i, j) * Asub[(i_bloco - 1)][j_bloco] +
                                                              e(i, j) * Asub[(i_bloco + 1) ][ j_bloco] +
                                                              s(i, j) * Asub[i_bloco][(j_bloco - 1)] +
                                                              n(i, j) * Asub[i_bloco][(j_bloco + 1)]);

    }


}

__global__ void gauss_seidel_local_gpu_par(double *atual, int N1, int N2){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if ((((i + j) % 2) == 0) && i > 0 && j > 0 && i < (N1 - 1) && j < (N2 - 1)) {
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

    if ((((i + j) % 2) == 1) && i > 0 && j > 0 && i < (N1 - 1) && j < (N2 - 1)) {
        double p = ro(i,j);
        double w = 2 / (1 + sqrt(1 - (p*p)));
        atual[i * N1 + j] = (1 - w) * atual[i * N1 + j] + w * (o(i, j) * atual[(i - 1) * N1 + j] +
                                                               e(i, j) * atual[(i + 1) * N1 + j] +
                                                               s(i, j) * atual[i * N1 + (j - 1)] +
                                                               n(i, j) * atual[i * N1 + (j + 1)]);

    }
}

__global__ void gauss_seidel_local_gpu_impar_shar(double *atual, int N1, int N2) {

    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    //coordenadas globais da thread
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;

    int i_bloco = threadIdx.x + 1;
    int j_bloco = threadIdx.y + 1;

    __shared__ double Asub[TAM_BLOCO+2][TAM_BLOCO+2];


    Asub[i_bloco][ j_bloco] = atual[i * N1 + j];

    if(i_bloco == 1) Asub[i_bloco - 1 ][ j_bloco] = atual[(i-1) * N1 + j];
    if(i_bloco == TAM_BLOCO) Asub[i_bloco + 1 ][ j_bloco] = atual[(i+1) * N1 + j];
    if(j_bloco == 1) Asub[i_bloco ][ j_bloco - 1] = atual[i * N1 + j - 1 ];
    if(j_bloco == TAM_BLOCO) Asub[i_bloco ][ j_bloco + 1] = atual[i * N1 + j + 1 ];



    __syncthreads();
    if ((((i + j) % 2) == 1) && i > 0 && j > 0 && i < (N1 - 1) && j < (N2 - 1)) {

        double p = ro(i,j);
        double w = 2 / (1 + sqrt(1 - (p*p)));

        atual[i * N1 + j] = (1 - w) * Asub[i_bloco][ j_bloco] + w * (o(i, j) * Asub[(i_bloco - 1)][ j_bloco] +
                                                                     e(i, j) * Asub[(i_bloco + 1)][ j_bloco] +
                                                                     s(i, j) * Asub[i_bloco][(j_bloco - 1)] +
                                                                     n(i, j) * Asub[i_bloco][ (j_bloco + 1)]);

    }


}

__global__ void gauss_seidel_local_gpu_par_shar(double *atual, int N1, int N2) {

    //coordenadas globais da thread
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
//coordenadas locais da thread
    int i_bloco = threadIdx.x + 1;
    int j_bloco = threadIdx.y + 1;

    __shared__ double Asub[TAM_BLOCO+2][TAM_BLOCO+2];

    Asub[i_bloco][j_bloco] = atual[i * N1 + j];
    if(i_bloco == 1) Asub[i_bloco - 1 ][ j_bloco] = atual[(i-1) * N1 + j];
    if(i_bloco == TAM_BLOCO) Asub[i_bloco + 1 ][ j_bloco] = atual[(i+1) * N1 + j];
    if(j_bloco == 1) Asub[i_bloco ][ j_bloco - 1] = atual[i * N1 + j - 1 ];
    if(j_bloco == TAM_BLOCO) Asub[i_bloco ][ j_bloco + 1] = atual[i * N1 + j + 1 ];

    __syncthreads();
    if ((((i + j) % 2) == 0) && i > 0 && j > 0  && i < (N1 - 1) && j < (N2 - 1) ) {
        double p = ro(i,j);
        double w = 2 / (1 + sqrt(1 - (p*p)));
        atual[i * N1 + j] = (1 - w) * Asub[i_bloco][j_bloco] + w * (o(i, j) * Asub[(i_bloco - 1)][j_bloco] +
                                                                    e(i, j) * Asub[(i_bloco + 1) ][ j_bloco] +
                                                                    s(i, j) * Asub[i_bloco][(j_bloco - 1)] +
                                                                    n(i, j) * Asub[i_bloco][(j_bloco + 1)]);

    }

}

