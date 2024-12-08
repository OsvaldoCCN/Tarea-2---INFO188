# Tarea-2---INFO188


## Descripción
Este proyecto tiene como objetivo comparar los mejores algoritmos de ordenamiento implementados para GPU y CPU, basándose en el estado del arte actual en computación paralela. A través de una revisión de la literatura, se seleccionarán los algoritmos de ordenamiento más eficientes y se implementarán en ambas plataformas para evaluar su rendimiento.

## Requerimientos
- Cuda Toolkit 12.6
- Make
- GCC

## Compilación
Para compilar el proyecto, se debe ejecutar el siguiente comando en la raíz del proyecto:

```bash
make
```

## Ejecución
Para ejecutar el proyecto, se debe ejecutar el siguiente comando en la raíz del proyecto:

```bash
./main <n> <m> <nt> <seed>
```

Donde:
- n: Tamaño del arreglo a ordenar.
- m: Modo de ejecución (0: CPU, 1: GPU, 2: stl Sort).
- nt: Número de threads de CPU.
- seed: Semilla para la generación de números aleatorios.

### Ejemplo
```bash
./main $((2**10)) 0 4 123
```


## Referencias

[1] Satish, N., Harris, M., & Garland, M. (2009). Designing efficient sorting algorithms for manycore GPUs. In 2009 IEEE International Symposium on Parallel & Distributed Processing (pp. 1-10). IEEE. https://doi.org/10.1109/IPDPS.2009.5161005

[2] Blelloch, G. E. (1990). Prefix Sums and Their Applications. CMU School of Computer Science Technical Report. https://www.cs.cmu.edu/~guyb/papers/Ble93.pdf

[3] Marszałek, Z. (2023). Parallelization of Modified Merge Sort Algorithm .https://www.mdpi.com/2073-8994/9/9/176 

[4] IEEE (2020). Performance Analysis of Merge Sort Algorithms.https://ieeexplore.ieee.org/document/9155623 

[5] https://github.com/qcuong98/GPURadixSort.git

[6] https://github.com/BohdanVelikdus/OMP_merge_sort


## Integrantes
- Andres Mardones
- Martin Alvarado
- Isaias Cabrera
- Osvaldo Casas-Cordero