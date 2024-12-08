#include <bits/stdc++.h>

#ifndef CTA_SIZE
#define CTA_SIZE 4
#endif


#ifndef K_BITS
#define K_BITS 8
#endif

#ifndef N_STREAMS
#define N_STREAMS 16
#endif

#define N_BINS (1 << K_BITS)
#define LOG_NUM_BANKS 5
#define CONFLICT_FREE_OFFSET(n) ((n) + ((n) >> LOG_NUM_BANKS))
#define ELEMENTS_PER_BLOCK (CTA_SIZE * 2 * BLOCKSIZE)
#define GRIDSIZE 64
#define BLOCKSIZE (n + GRIDSIZE - 1) / GRIDSIZE

#define CHECK(call)                                                            \
{                                                                              \
    const cudaError_t error = call;                                            \
    if (error != cudaSuccess)                                                  \
    {                                                                          \
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);                 \
        fprintf(stderr, "code: %d, reason: %s\n", error,                       \
                cudaGetErrorString(error));                                    \
        exit(1);                                                               \
}                                                                              \
}


__device__ __forceinline__ uint32_t getBin(uint32_t val, uint32_t bit, uint32_t nBins) {
    return (val >> bit) & (nBins - 1);
}


/**
 * @brief Realiza un escaneo (prefix sum) dentro de un bloque de datos, almacenando el resultado en `out`
 *        y las sumas de bloques en `blkSums`.
 *
 * Este kernel calcula un escaneo (prefix sum) dentro de un bloque de elementos de un arreglo de entrada 
 * (`src`) y guarda los resultados en el arreglo de salida (`out`). Al mismo tiempo, el kernel guarda la 
 * suma total de cada bloque en el arreglo `blkSums` para ser usado en fases posteriores del algoritmo 
 * de escaneo global.
 * 
 * El escaneo se realiza en varias fases: la fase de escaneo local (donde se calcula la suma acumulada 
 * dentro de un bloque) y la fase de reducción (donde los resultados se consolidan a nivel de bloque).
 * 
 * El escaneo de los elementos se realiza en memoria compartida para optimizar el acceso a los datos.
 * 
 * @param src Arreglo de entrada en dispositivo que contiene los valores a ser escaneados.
 * @param n Número total de elementos en el arreglo `src` que se deben procesar.
 * @param out Arreglo de salida en dispositivo donde se almacenarán los resultados del escaneo.
 * @param blkSums Arreglo en dispositivo donde se almacenarán las sumas de cada bloque. 
 *                Este arreglo es utilizado para la fase de reducción.
 */
__global__ void scanBlkKernel(uint32_t * src, int n, uint32_t * out, uint32_t * blkSums) {
    extern __shared__ uint32_t s[];
    uint32_t* localScan = s;
    uint32_t* localScanCTA = localScan + CONFLICT_FREE_OFFSET(ELEMENTS_PER_BLOCK);

    int ai = threadIdx.x;
    int bi = threadIdx.x + blockDim.x;
    
    uint32_t first = ELEMENTS_PER_BLOCK * blockIdx.x;
    for (int i = threadIdx.x; i < ELEMENTS_PER_BLOCK; i += blockDim.x) {
        int pos = first + i;
        localScan[CONFLICT_FREE_OFFSET(i)] = pos < n ? src[pos] : 0;
    }
    __syncthreads();

    uint32_t tempA[CTA_SIZE], tempB[CTA_SIZE];
    # pragma unroll
    for (int i = 0; i < CTA_SIZE; ++i) {
        tempA[i] = localScan[CONFLICT_FREE_OFFSET(CTA_SIZE * ai + i)]; 
        tempB[i] = localScan[CONFLICT_FREE_OFFSET(CTA_SIZE * bi + i)]; 
        if (i) {
            tempA[i] += tempA[i - 1];
            tempB[i] += tempB[i - 1];
        }
    }

    // compute scan
    localScanCTA[CONFLICT_FREE_OFFSET(ai)] = tempA[CTA_SIZE - 1];
    localScanCTA[CONFLICT_FREE_OFFSET(bi)] = tempB[CTA_SIZE - 1];
    __syncthreads();

    // reduction phase
    # pragma unroll
    for (int stride = 1, d = BLOCKSIZE; stride <= BLOCKSIZE; stride <<= 1, d >>= 1) {
        if (threadIdx.x < d) {
            int cur = 2 * stride * (threadIdx.x + 1) - 1;
            int prev = cur - stride;
            localScanCTA[CONFLICT_FREE_OFFSET(cur)] += localScanCTA[CONFLICT_FREE_OFFSET(prev)];
        }
        __syncthreads();
    }
    // post-reduction phase
    # pragma unroll
    for (int stride = BLOCKSIZE >> 1, d = 2; stride >= 1; stride >>= 1, d <<= 1) {
        if (threadIdx.x < d - 1) {
            int prev = 2 * stride * (threadIdx.x + 1) - 1;
            int cur = prev + stride;
            localScanCTA[CONFLICT_FREE_OFFSET(cur)] += localScanCTA[CONFLICT_FREE_OFFSET(prev)];
        }
        __syncthreads();
    }
    
    uint32_t lastScanA = ai ? localScanCTA[CONFLICT_FREE_OFFSET(ai - 1)] : 0;
    uint32_t lastScanB = localScanCTA[CONFLICT_FREE_OFFSET(bi - 1)];
    __syncthreads();
    
    #pragma unroll
    for (int i = 0; i < CTA_SIZE; ++i) {
        tempA[i] += lastScanA;
        tempB[i] += lastScanB;
        
        localScan[CONFLICT_FREE_OFFSET(CTA_SIZE * ai + i)] = tempA[i];
        localScan[CONFLICT_FREE_OFFSET(CTA_SIZE * bi + i)] = tempB[i];
    }
    __syncthreads();
    
    for (int i = threadIdx.x; i < ELEMENTS_PER_BLOCK; i += blockDim.x) {
        if (first + i < n)
            out[first + i] = localScan[CONFLICT_FREE_OFFSET(i)];
    }
    
    if (threadIdx.x == blockDim.x - 1) {
        blkSums[blockIdx.x] = tempB[CTA_SIZE - 1];
    }
}


/**
 * @brief Realiza la suma acumulada (prefix sum) en el arreglo de salida, agregando las sumas de bloques previos.
 *
 * Este kernel ajusta los elementos del arreglo de salida `out` sumando la suma de los bloques anteriores. 
 * La suma acumulada se realiza a nivel de bloques, es decir, se suma el valor de la suma total de los bloques 
 * previos (almacenada en `blkSums`) a los elementos correspondientes en el arreglo de salida `out`.
 * 
 * Este kernel se ejecuta después de haber calculado las sumas de bloques individuales, y aplica la 
 * corrección de la suma a los elementos de cada bloque de manera paralela.
 *
 * @param out Arreglo de salida en dispositivo donde se almacenarán los resultados de la suma acumulada.
 * @param n Número total de elementos en el arreglo `out` que se deben procesar.
 * @param blkSums Arreglo de dispositivo que contiene las sumas de cada bloque. Este arreglo es utilizado para
 *               sumar el valor acumulado de los bloques anteriores a los elementos del bloque actual.
 */
__global__ void sumPrefixBlkKernel(uint32_t * out, int n, uint32_t * blkSums) {
    uint32_t lastBlockSum = blockIdx.x > 0 ? blkSums[blockIdx.x - 1] : 0;
    uint32_t first = ELEMENTS_PER_BLOCK * blockIdx.x;
    for (int i = threadIdx.x; i < ELEMENTS_PER_BLOCK; i += blockDim.x) {
        if (first + i < n)
            out[first + i] += lastBlockSum;
    }
}

/**
 * @brief Realiza una reducción en el arreglo de entrada, restando cada elemento del arreglo de entrada al arreglo de salida.
 *
 * Este kernel recibe dos arreglos, uno de entrada `in` y uno de salida `out`, y realiza la operación de resta
 * entre los elementos correspondientes en ambos arreglos. El resultado de la resta de cada elemento de `in`
 * se guarda en el arreglo `out`. La operación se realiza de manera paralela para cada hilo en el bloque de CUDA.
 *
 * @param in Arreglo de entrada en dispositivo con los valores sobre los que se realiza la operación de resta.
 * @param n Número de elementos en el arreglo `in` que se deben procesar.
 * @param out Arreglo de salida en dispositivo donde se almacenarán los resultados de la resta.
 */
__global__ void reduceKernel(uint32_t * in, int n, uint32_t * out) {
    int id_in = blockDim.x * blockIdx.x + threadIdx.x;
    if (id_in < n)
        out[id_in] -= in[id_in];
}


/**
 * @brief Realiza un escaneo (prefix sum) en un arreglo de manera paralela usando CUDA.
 *
 * Esta función implementa un escaneo de prefijos sobre un arreglo de tamaño `n` de manera recursiva,
 * dividiendo el trabajo entre bloques de hilos. Primero, se realiza el escaneo de cada bloque,
 * luego se suman los resultados de los bloques, y finalmente se actualiza el arreglo de salida
 * con el escaneo total.
 *
 * @param d_in Arreglo de entrada en dispositivo que contiene los datos sobre los que se realiza el escaneo.
 * @param d_out Arreglo de salida en dispositivo que contendrá los resultados del escaneo.
 * @param n El tamaño del arreglo de entrada (número de elementos).
 * @param elementsPerBlock Dimensión de cada bloque (cuántos elementos maneja cada bloque).
 * @param blockSize Tamaño del bloque (número de hilos por bloque).
 */
void computeScanArray(uint32_t* d_in, uint32_t* d_out, int n, dim3 elementsPerBlock, dim3 blockSize) {
    dim3 gridSize((n - 1) / elementsPerBlock.x + 1);

    uint32_t * d_blkSums;
    CHECK(cudaMalloc(&d_blkSums, gridSize.x * sizeof(uint32_t)));
    uint32_t * d_sum_blkSums;
    CHECK(cudaMalloc(&d_sum_blkSums, gridSize.x * sizeof(uint32_t)));

    scanBlkKernel<<<gridSize, blockSize, CONFLICT_FREE_OFFSET((2 * CTA_SIZE + 2) * blockSize.x) * sizeof(uint32_t)>>>
        (d_in, n, d_out, d_blkSums);
    if (gridSize.x != 1) {
        computeScanArray(d_blkSums, d_sum_blkSums, gridSize.x, elementsPerBlock, blockSize);
    }
    sumPrefixBlkKernel<<<gridSize, blockSize>>>(d_out, n, d_sum_blkSums);

    CHECK(cudaFree(d_sum_blkSums));
    CHECK(cudaFree(d_blkSums));
}


/**
 * @brief Realiza la operación de dispersión (scatter) en una matriz de enteros de 32 bits.
 * 
 * Esta función toma una matriz de entrada, dispersa sus elementos a las posiciones adecuadas en la matriz
 * de salida según un cálculo basado en el histograma escaneado. Los hilos se encargan de colocar los valores 
 * dispersados en las posiciones correctas en la matriz de destino utilizando un esquema de dispersión 
 * eficiente en paralelo.
 * 
 * @param src Puntero a la matriz de entrada de tipo `uint32_t`, cuyos elementos serán dispersados.
 * @param n Número de elementos en la matriz `src`.
 * @param dst Puntero a la matriz de salida de tipo `uint32_t`, donde se almacenarán los elementos dispersados.
 * @param histScan Puntero a un histograma previamente escaneado que ayuda a calcular las posiciones de dispersión.
 * @param bit El bit actual que se está utilizando para la dispersión.
 * @param count Puntero a la matriz `count` que almacena el número de elementos que ya se han procesado.
 */
__global__ void scatterKernel(uint32_t* src, int n, uint32_t* dst, uint32_t* histScan, int bit, uint32_t* count) {
    extern __shared__ uint32_t start[];
    uint32_t first = N_BINS * blockIdx.x;
    for (int i = threadIdx.x; i < N_BINS; i += blockDim.x) {
        start[CONFLICT_FREE_OFFSET(i)] = histScan[first + i];
    }
    __syncthreads();
    
    first = ELEMENTS_PER_BLOCK * blockIdx.x;
    for (int i = threadIdx.x; i < ELEMENTS_PER_BLOCK; i += blockDim.x) {
        if (first + i < n) {
            uint32_t val = src[first + i];
            uint32_t st = start[CONFLICT_FREE_OFFSET(getBin(val, bit, N_BINS))];
            uint32_t equalsBefore = count[first + i];
            uint32_t pos = st + equalsBefore - 1;
            dst[pos] = val;
        }
    }
}


/**
 * @brief Realiza el ordenamiento local utilizando el algoritmo Radix Sort en la GPU.
 * 
 * Este kernel realiza el ordenamiento local de elementos usando el algoritmo Radix Sort. La implementación
 * utiliza un esquema de memoria compartida para almacenar los elementos de la entrada y el resultado de los 
 * cálculos intermedios, optimizando las fases de escaneo, reducción y dispersión (scatter). El kernel usa 
 * múltiples fases de reducción para procesar los datos de manera eficiente en paralelo.
 * 
 * @param src Puntero a la matriz de entrada de tipo `uint32_t` que contiene los elementos a ordenar.
 * @param n Número de elementos en la matriz `src`.
 * @param bit El bit actual que se está utilizando para el ordenamiento.
 * @param count Puntero a la matriz `count` que lleva el registro de las posiciones de los elementos.
 * @param hist Puntero al histograma que almacena los resultados intermedios del ordenamiento.
 * @param start_pos Posición inicial del bloque de trabajo (valor predeterminado es 0).
 */
__global__ void sortLocalKernel(uint32_t* src, int n, int bit, uint32_t* count, uint32_t* hist, int start_pos = 0) {
    extern __shared__ uint32_t s[];
    uint32_t* localSrc = s;
    uint32_t* localBin = localSrc + CONFLICT_FREE_OFFSET(ELEMENTS_PER_BLOCK);
    uint32_t* localScan = localBin + CONFLICT_FREE_OFFSET(2 * BLOCKSIZE);
    uint32_t* s_hist = localScan + CONFLICT_FREE_OFFSET(ELEMENTS_PER_BLOCK);

    int ai = threadIdx.x;
    int bi = threadIdx.x + blockDim.x;
    uint32_t first = ELEMENTS_PER_BLOCK * blockIdx.x;
    
    for (int i = threadIdx.x; i < ELEMENTS_PER_BLOCK; i += blockDim.x) {
        int pos = first + i;
        localSrc[CONFLICT_FREE_OFFSET(i)] = pos < n ? src[pos] : UINT_MAX;
    }
    __syncthreads();

    // radix sort with k = 1
    uint32_t tempA[CTA_SIZE], tempB[CTA_SIZE];
    #pragma unroll
    for (int b = 0; b < K_BITS; ++b) {
        int blockBit = bit + b;
        uint32_t valA = 0, valB = 0;
        # pragma unroll
        for (int i = 0; i < CTA_SIZE; ++i) {
            uint32_t thisA = getBin(tempA[i] = localSrc[CONFLICT_FREE_OFFSET(CTA_SIZE * ai + i)], bit, N_BINS); 
            valA += (tempA[i] >> blockBit & 1);
            uint32_t thisB = getBin(tempB[i] = localSrc[CONFLICT_FREE_OFFSET(CTA_SIZE * bi + i)], bit, N_BINS); 
            valB += (tempB[i] >> blockBit & 1);
        }

        // compute scan
        localScan[CONFLICT_FREE_OFFSET(ai)] = valA;
        localScan[CONFLICT_FREE_OFFSET(bi)] = valB;
        __syncthreads();
        
        // reduction phase
        # pragma unroll
        for (int stride = 1, d = BLOCKSIZE; stride <= BLOCKSIZE; stride <<= 1, d >>= 1) {
            if (threadIdx.x < d) {
                int cur = 2 * stride * (threadIdx.x + 1) - 1;
                int prev = cur - stride;
                localScan[CONFLICT_FREE_OFFSET(cur)] += localScan[CONFLICT_FREE_OFFSET(prev)];
            }
            __syncthreads();
        }
        // post-reduction phase
        # pragma unroll
        for (int stride = BLOCKSIZE >> 1, d = 2; stride >= 1; stride >>= 1, d <<= 1) {
            if (threadIdx.x < d - 1) {
                int prev = 2 * stride * (threadIdx.x + 1) - 1;
                int cur = prev + stride;
                localScan[CONFLICT_FREE_OFFSET(cur)] += localScan[CONFLICT_FREE_OFFSET(prev)];
            }
            __syncthreads();
        }
        
        // scatter
        int n0 = ELEMENTS_PER_BLOCK - localScan[CONFLICT_FREE_OFFSET(2 * blockDim.x - 1)];
        
        valA = localScan[CONFLICT_FREE_OFFSET(ai)];
        valB = localScan[CONFLICT_FREE_OFFSET(bi)];
        # pragma unroll
        for (int i = CTA_SIZE - 1; i >= 0; --i) {
            if (tempA[i] >> blockBit & 1)
                localSrc[CONFLICT_FREE_OFFSET(n0 + valA - 1)] = tempA[i];
            else
                localSrc[CONFLICT_FREE_OFFSET(CTA_SIZE * ai + i - valA)] = tempA[i];
            valA -= (tempA[i] >> blockBit & 1);
            
            if (tempB[i] >> blockBit & 1)
                localSrc[CONFLICT_FREE_OFFSET(n0 + valB - 1)] = tempB[i];
            else
                localSrc[CONFLICT_FREE_OFFSET(CTA_SIZE * bi + i - valB)] = tempB[i];
            valB -= (tempB[i] >> blockBit & 1);
        }
        
        __syncthreads();
    }
    
    // -------------------------------------------------------------------
    // countEqualsBefore
    uint32_t countA[CTA_SIZE], countB[CTA_SIZE];
    #pragma unroll
    for (int i = 0; i < CTA_SIZE; ++i) {
        tempA[i] = getBin(localSrc[CONFLICT_FREE_OFFSET(CTA_SIZE * ai + i)], bit, N_BINS); 
        tempB[i] = getBin(localSrc[CONFLICT_FREE_OFFSET(CTA_SIZE * bi + i)], bit, N_BINS); 
        countA[i] = countB[i] = 1;
        if (i) {
            if (tempA[i] == tempA[i - 1])
                countA[i] += countA[i - 1];
            if (tempB[i] == tempB[i - 1])
                countB[i] += countB[i - 1];
        }
    }

    localScan[CONFLICT_FREE_OFFSET(ai)] = countA[CTA_SIZE - 1];
    localScan[CONFLICT_FREE_OFFSET(bi)] = countB[CTA_SIZE - 1];
    localBin[CONFLICT_FREE_OFFSET(ai)] = tempA[CTA_SIZE - 1];
    localBin[CONFLICT_FREE_OFFSET(bi)] = tempB[CTA_SIZE - 1];
    __syncthreads();

    // reduction phase
    # pragma unroll
    for (int stride = 1, d = BLOCKSIZE; stride <= BLOCKSIZE; stride <<= 1, d >>= 1) {
        if (threadIdx.x < d) {
            int cur = 2 * stride * (threadIdx.x + 1) - 1;
            int prev = cur - stride;
            cur = CONFLICT_FREE_OFFSET(cur);
            prev = CONFLICT_FREE_OFFSET(prev);
            if (localBin[cur] == localBin[prev])
                localScan[cur] += localScan[prev];
        }
        __syncthreads();
    }
    // post-reduction phase
    # pragma unroll
    for (int stride = BLOCKSIZE >> 1, d = 2; stride >= 1; stride >>= 1, d <<= 1) {
        if (threadIdx.x < d - 1) {
            int prev = 2 * stride * (threadIdx.x + 1) - 1;
            int cur = prev + stride;
            cur = CONFLICT_FREE_OFFSET(cur);
            prev = CONFLICT_FREE_OFFSET(prev);
            if (localBin[cur] == localBin[prev])
                localScan[cur] += localScan[prev];
        }
        __syncthreads();
    }

    uint32_t lastBinA = localBin[CONFLICT_FREE_OFFSET(ai - 1)];
    uint32_t lastBinB = localBin[CONFLICT_FREE_OFFSET(bi - 1)];
    uint32_t lastScanA = ai ? localScan[CONFLICT_FREE_OFFSET(ai - 1)] : 0;
    uint32_t lastScanB = localScan[CONFLICT_FREE_OFFSET(bi - 1)];
    __syncthreads();
    
    #pragma unroll
    for (int i = 0; i < CTA_SIZE; ++i) {
        if (tempA[i] == lastBinA)
            countA[i] += lastScanA;
        
        if (tempB[i] == lastBinB)
            countB[i] += lastScanB;
        
        localScan[CONFLICT_FREE_OFFSET(CTA_SIZE * ai + i)] = countA[i];
        localScan[CONFLICT_FREE_OFFSET(CTA_SIZE * bi + i)] = countB[i];
    }
    __syncthreads();
    
    for (int i = threadIdx.x; i < ELEMENTS_PER_BLOCK; i += blockDim.x) {
        int pos = first + i;
        if (pos < n) {
            count[pos] = localScan[CONFLICT_FREE_OFFSET(i)];
            src[pos] = localSrc[CONFLICT_FREE_OFFSET(i)];
        }
    }
    
    // -------------------------------------------
    // compute hist
    for (int idx = threadIdx.x; idx < N_BINS; idx += blockDim.x)
        s_hist[CONFLICT_FREE_OFFSET(idx)] = 0;
    __syncthreads();
    for (int i = threadIdx.x; i < ELEMENTS_PER_BLOCK; i += blockDim.x) {
        int pos = first + i;
        if (pos < n) {
            uint32_t thisBin = getBin(localSrc[CONFLICT_FREE_OFFSET(i)], bit, N_BINS);
          if (pos == n - 1 || i == ELEMENTS_PER_BLOCK - 1 || thisBin != getBin(localSrc[CONFLICT_FREE_OFFSET(i + 1)], bit, N_BINS))
              s_hist[CONFLICT_FREE_OFFSET(thisBin)] = localScan[CONFLICT_FREE_OFFSET(i)];
        }
    }
    __syncthreads();
    
    first = (blockIdx.x + start_pos) * N_BINS;
    for (int digit = threadIdx.x; digit < N_BINS; digit += blockDim.x)
        hist[first + digit] = s_hist[CONFLICT_FREE_OFFSET(digit)];
}


/**
 * @brief Realiza la transposición de una matriz de enteros de 32 bits en la GPU.
 * 
 * Esta función toma una matriz de entrada y genera su transpuesta. La transposición se realiza
 * utilizando memoria compartida para reducir la latencia y aumentar la eficiencia en el acceso a los datos.
 * La operación se realiza en paralelo con múltiples bloques y hilos de la GPU.
 * 
 * @param iMatrix Puntero a la matriz de entrada de tipo `uint32_t`, que contiene los elementos a transponer.
 * @param oMatrix Puntero a la matriz de salida de tipo `uint32_t`, donde se almacenarán los elementos transpuestos.
 * @param rows Número de filas de la matriz original.
 * @param cols Número de columnas de la matriz original.
 */
__global__ void transpose(uint32_t *iMatrix, uint32_t *oMatrix, int rows, int cols) {
    __shared__ int s_blkData[32][33];
    int iR = blockIdx.x * blockDim.x + threadIdx.y;
    int iC = blockIdx.y * blockDim.y + threadIdx.x;
    s_blkData[threadIdx.y][threadIdx.x] = (iR < rows && iC < cols) ? iMatrix[iR * cols + iC] : 0;
    __syncthreads();
    // Each block write data efficiently from SMEM to GMEM
    int oR = blockIdx.y * blockDim.y + threadIdx.y;
    int oC = blockIdx.x * blockDim.x + threadIdx.x;
    if (oR < cols && oC < rows)
        oMatrix[oR * rows + oC] = s_blkData[threadIdx.x][threadIdx.y];
}


/**
* @brief Ordena un arreglo de enteros de 32 bits con radix paralelo en CUDA.
*
* La función sort implementa el algoritmo radixSort de ordenamiento paralelo en CUDA para un arreglo de enteros de 32 bits (uint32_t).
* El algoritmo se basa en el enfoque de ordenamiento por clasificación (radix sort), que maneja números enteros en sus 
* diferentes bits en iteraciones, distribuyendo los valores en "bins" y realizando operaciones de escaneo (scan) y 
* dispersión (scatter). El proceso se lleva a cabo en múltiples fases de forma paralela, aprovechando múltiples streams 
* de CUDA para mejorar la eficiencia.
*
* @param in: Puntero a un arreglo de enteros de 32 bits a ordenar.
* @param n: Número de elementos en el arreglo.
* @param out: Puntero a un arreglo de enteros de 32 bits donde se almacenarán los elementos ordenados.
*/
void sort(const uint32_t * in, int n, uint32_t * out) {
    uint32_t * d_src;
    uint32_t * d_dst;
    uint32_t * d_hist;
    uint32_t * d_histScan;
    uint32_t * d_count;
    CHECK(cudaMalloc(&d_src, n * sizeof(uint32_t)));
    CHECK(cudaMalloc(&d_count, n * sizeof(uint32_t)));
    CHECK(cudaMalloc(&d_dst, n * sizeof(uint32_t)));

    // Compute block and grid size for scan and scatter phase
    dim3 gridSize(GRIDSIZE);
    dim3 blockSize(BLOCKSIZE);
    dim3 elementsPerBlock(ELEMENTS_PER_BLOCK);

    dim3 blockSizeTranspose(32, 32);
    dim3 gridSizeTransposeHist((gridSize.x - 1) / blockSizeTranspose.x + 1, (N_BINS - 1) / blockSizeTranspose.x + 1);
    dim3 gridSizeTransposeHistScan((N_BINS - 1) / blockSizeTranspose.x + 1, (gridSize.x - 1) / blockSizeTranspose.x + 1);
    
    int histSize = N_BINS * gridSize.x;
    CHECK(cudaMalloc(&d_hist, 2 * histSize * sizeof(uint32_t)));
    CHECK(cudaMalloc(&d_histScan, 2 * histSize * sizeof(uint32_t)));
    dim3 gridSizeScan((histSize - 1) / blockSize.x + 1);

    cudaStream_t *streams = (cudaStream_t *) malloc(N_STREAMS * sizeof(cudaStream_t));    
    for (int i = 0; i < N_STREAMS; ++i) {
        CHECK(cudaStreamCreate(&streams[i]));
    }
    int len = (gridSize.x - 1) / N_STREAMS + 1;
    for (int i = 0; i < N_STREAMS; ++i) {
        int cur_pos = i * len * elementsPerBlock.x;
        if (cur_pos >= n)
            break;
        int cur_len = min(len * elementsPerBlock.x, n - i * len * elementsPerBlock.x);
        dim3 cur_gridSize((cur_len - 1) / elementsPerBlock.x + 1);
        CHECK(cudaMemcpyAsync(d_src + cur_pos, in + cur_pos, cur_len * sizeof(uint32_t), 
                                            cudaMemcpyHostToDevice, streams[i]));
        sortLocalKernel<<<cur_gridSize, blockSize, CONFLICT_FREE_OFFSET((4 * CTA_SIZE + 2) * blockSize.x + N_BINS) * sizeof(uint32_t), streams[i]>>>
            (d_src + cur_pos, cur_len, 0, d_count + cur_pos, d_hist + histSize, i * len);
    }

    for (int bit = 0; bit < 32; bit += K_BITS) {
        if (bit) {
            sortLocalKernel<<<gridSize, blockSize, CONFLICT_FREE_OFFSET((4 * CTA_SIZE + 2) * BLOCKSIZE + N_BINS) * sizeof(uint32_t)>>>
                (d_src, n, bit, d_count, d_hist + histSize);
        }
        
        transpose<<<gridSizeTransposeHist, blockSizeTranspose>>>
          (d_hist + histSize, d_hist, gridSize.x, N_BINS);

        // compute hist scan
        computeScanArray(d_hist, d_histScan + histSize, histSize, elementsPerBlock, blockSize);
        reduceKernel<<<gridSizeScan, blockSize>>>
            (d_hist, histSize, d_histScan + histSize);
        
        transpose<<<gridSizeTransposeHistScan, blockSizeTranspose>>>
          (d_histScan + histSize, d_histScan, N_BINS, gridSize.x);
        
        // scatter
        scatterKernel<<<gridSize, blockSize, CONFLICT_FREE_OFFSET(N_BINS) * sizeof(uint32_t)>>>
            (d_src, n, d_dst, d_histScan, bit, d_count);
        uint32_t * tmp = d_src; d_src = d_dst; d_dst = tmp;
    }

    CHECK(cudaMemcpy(out, d_src, n * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    
    CHECK(cudaFree(d_src));
    CHECK(cudaFree(d_dst));
    CHECK(cudaFree(d_hist));
    CHECK(cudaFree(d_histScan));
}
