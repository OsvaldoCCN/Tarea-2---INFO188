CC = nvcc
PROF = nvprof
K = 8
BLKSIZE = 256
CTA = 2
STREAMS = 16


all: build
build:
	@${CC} main.cu -DK_BITS=${K} -DBLOCKSIZE=${BLKSIZE} -DCTA_SIZE=${CTA} -DN_STREAMS=${STREAMS} -O2 -Xcompiler -fopenmp -o main