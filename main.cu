#include "GPUSort.cu"
#include "CPUSort.cpp"

#include <random>
#include <omp.h>
#include <cuda.h>
#include <cstdio>
#include <cstdlib>
#include <math.h>


/**
 * @brief Llena una matriz de tamaño n con números aleatorios
 * 
 * @param n Tamaño de la matriz
 * @param m Matriz a llenar
 * @param seed Semilla para el generador de números aleatorios
 * 
 * @return void
 */
void matrandom(int n, uint32_t *m, int seed){
    std::mt19937 mt(seed);
    std::uniform_int_distribution<int>  dist(0, INT16_MAX);
    for(int k=0; k<n; ++k){
        m[k] = dist(mt);
    }
}


/**
 * @brief Imprime los primeros 20 elementos de una matriz
 * 
 * @param n Tamaño de la matriz
 * @param m Matriz a imprimir
 * @param msg Mensaje a imprimir antes de la matriz
 * 
 * @return void
 */
void printmat(int n, uint32_t *m, const char* msg){
    printf("%s\n", msg);
    for(int i=0; i<(n < 20? n : 20); ++i){
        printf("%i ", m[i]);
    }
    printf("\n");
}


/**
 * @brief Crea un vector de tamaño n con números aleatorios
 * 
 * @param n Tamaño del vector
 * @param seed Semilla para el generador de números aleatorios
 * 
 * @return std::vector<uint32_t> Vector de tamaño n con números aleatorios
 */
std :: vector<uint32_t> creaVector(int n, int seed){
    std :: vector<uint32_t> a(n);
    std::mt19937 mt(seed);
    std::uniform_int_distribution<int>  dist(0, INT16_MAX);
    for(int k=0; k<n; ++k){
        a[k] = dist(mt);
    }
    return a;
}

/**
 * @brief Función principal
 * 
 * @param argc Número de argumentos
 * @param argv Argumentos
 * 
 * @return int
 */
int main(int argc, char **argv){
    if(argc != 5){ fprintf(stderr, "run as ./main n modo nt seed\n\n"); exit(EXIT_FAILURE); }
    int n = atoi(argv[1]);
    int modo = atoi(argv[2]);
    int nt = atoi(argv[3]);
    int seed = atoi(argv[4]);
    float msecs = 0.0f;

    // (1) creando matrices en host
    uint32_t * a = (uint32_t *)malloc(n*sizeof(uint32_t));
    uint32_t * o = (uint32_t *)malloc(n*sizeof(uint32_t));
    
    if(modo == 1){
        printf("\nGPU Sorting\n\n");
    }
    else{
        printf("\nCPU Sorting\n\n");
    }

    if(modo == 0){
        printf("initializing A............."); fflush(stdout);
        std :: vector<uint32_t> a = creaVector(n, seed);
        std :: vector<uint32_t> o(n);
        printf("ok\n"); fflush(stdout);

        printmat(n, a.data(), "Arreglo Inicial:");

        printf("\n");
        printf("Calculando.............."); fflush(stdout);

        omp_set_num_threads(nt);
        
        float t1 = omp_get_wtime();
        #pragma omp parallel
        #pragma omp single
        o = recursiveMergeSort(a, 0, n, 0);
        float t2 = omp_get_wtime();

        printf("ok: time: %f secs\n",t2 - t1);
        printf("n° de threads: %i\n", nt);
        printmat(n, o.data(), "Arreglo Ordenado:");

    }
    if(modo == 1){

        printf("initializing A............."); fflush(stdout);
        matrandom(n, a, seed);
        printf("ok\n"); fflush(stdout);
        printmat(n, a, "Arreglo Inicial:");
        printf("\n");

        printf("Calculando..............\n"); fflush(stdout);


        cudaEvent_t start, stop;
        cudaEventCreate(&start); cudaEventCreate(&stop);
        cudaEventRecord(start);
        sort(a, n, o);
        cudaDeviceSynchronize(); cudaEventRecord(stop);
        cudaEventSynchronize(stop); cudaEventElapsedTime(&msecs, start, stop);


        printf("ok: time: %f secs\n", msecs/1000.0f);

        // (4) copiar resultado a host
        printf("copying result to host..........."); fflush(stdout);
        printf("ok\n"); fflush(stdout);
        printf("tamaño blockSize: %i\n", BLOCKSIZE);
        //2 * CTA_SIZE * BLOCKSIZE
        printf("cantidad de bloques: %i\n",GRIDSIZE);
        printf("espacio en memoria: %lu\n", 2*n*sizeof(uint32_t));
        printmat(n, o, "Arreglo Ordenado:");
    }
    if(modo == 2){
        printf("initializing A............."); fflush(stdout);
        std :: vector<uint32_t> a = creaVector(n, seed);
        printf("ok\n"); fflush(stdout);

        printmat(n, a.data(), "Arreglo Inicial:");

        printf("\n");
        printf("Calculando.............."); fflush(stdout);

        float t1 = omp_get_wtime();
        std :: sort(a.begin(), a.end());
        float t2 = omp_get_wtime();

        printf("ok: time: %f secs\n", t2 - t1);
        printmat(n, a.data(), "Arreglo Ordenado:");
    }

    printf("done!\n");
    exit(EXIT_SUCCESS);
}

