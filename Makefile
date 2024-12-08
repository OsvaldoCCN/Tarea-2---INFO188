CC = nvcc
PROF = nvprof
K = 8
CTA = 2
STREAMS = 16


all: build
build:
	@${CC} main.cu -DK_BITS=${K}  -DCTA_SIZE=${CTA} -DN_STREAMS=${STREAMS} -O2 -Xcompiler -fopenmp -o main