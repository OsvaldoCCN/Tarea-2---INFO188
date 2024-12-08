#include <vector>
#include <omp.h>
#include <iostream>

#define DEPTH 14


/**
* @brief Ordena un vector de elementos de tipo T en paralelo usando el algoritmo de MergeSort
*
* Primero divide el arreglo en dos mitades y luego ordena recursivamente cada mitad.
* Para mejorar el rendimiento en sistemas con múltiples núcleos, al alcanzar una profundidad 
* máxima definida por el parámetro DEPTH, utiliza las directivas de OpenMP para ejecutar estas 
* divisiones y ordenaciones en paralelo. Una vez que ambas mitades están ordenadas, 
* las fusiona en un solo arreglo ordenado. Este enfoque aprovecha la paralelización en 
* CPU para optimizar el tiempo de ejecución.
*
* @param arr Vector de elementos de tipo T a ordenar
* @param left Índice del primer elemento del vector a ordenar
* @param right Índice del último elemento del vector a ordenar
* @param depth Profundidad actual de la recursión
*
* @return Vector de elementos de tipo T ordenado
*/
template<typename T> std::vector<T> recursiveMergeSort( std::vector<T> arr, int left, int right, int depth) {
    if(arr.size() == 1){
        return arr;
    }else{
        depth++;
        int size_ = arr.size();
        int halv = size_/2;
        std::vector<T> fHalve(halv);
        std::vector<T> sHalve(size_ - halv);
        for(int i = 0; i < halv;i++){
            fHalve[i] = arr[i];
        }
        for(int j = 0; j < size_ - halv;j++){
            sHalve[j] = arr[halv + j];
        }
        std::vector<T> fHalveS;
        std::vector<T> sHalveS;
        if(!omp_in_final()){
            #pragma omp task shared(fHalveS) final(depth >= DEPTH)
            fHalveS = recursiveMergeSort(fHalve, 0, halv, depth);
            #pragma omp task shared(sHalveS) final(depth >= DEPTH)
            sHalveS = recursiveMergeSort(sHalve, halv, size_, depth);
            #pragma omp taskwait
        }else{
            fHalveS = recursiveMergeSort(fHalve, 0, halv, depth);
            sHalveS = recursiveMergeSort(sHalve, halv, size_, depth);
        }    
        std::vector<T> rs(size_);
        int off1 = 0;
        int off2 = 0;
        for(int i = 0; i < size_; i++){
            if(  (off1 < fHalveS.size()) && (off2 >= sHalveS.size() || (fHalveS[off1] < sHalveS[off2]))  ){
                rs[i] = fHalveS[off1];
                off1++;
            }else{
                rs[i] = sHalveS[off2];
                off2++;
            }
        }
        return rs;
    }    
}
