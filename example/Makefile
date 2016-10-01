CC=gcc
CXX=g++
NVCC=nvcc
MAXAS=maxas.pl

CUDA_PATH=/usr/local/cuda
INCLUDE=-I. -I$(CUDA_PATH)/include

CPPFLAGS=-O3 -g -std=c++11 $(INCLUDE) # -Wall -Werror
NVCCFLAGS=-O3 -std=c++11 -lineinfo -Xcompiler $(INCLUDE)
LDFLAGS=-g -L. -L$(CUDA_PATH)/lib64
LDLIBS=-lcuda -lcudart -lcublas

all: test_maxas

test_maxas: test_maxas.o maxas_sgemm.cubin
	$(CXX) $(LDFLAGS) -o $@ $< $(LDLIBS)

test_maxas.o: test_maxas.cpp
	$(CXX) $(CPPFLAGS) -c $< -o $@

maxas_sgemm.cubin: maxas_sgemm.cu maxas_sgemm64.sass maxas_sgemm128.sass
	$(NVCC) -arch sm_50 -cubin $(INCLUDE) maxas_sgemm.cu
	$(MAXAS) -e -k sgemm_kernel_64 maxas_sgemm.cubin maxas_sgemm64-skeleton.asm
	$(MAXAS) -e -k sgemm_kernel_128 maxas_sgemm.cubin maxas_sgemm128-skeleton.asm
	$(MAXAS) -i maxas_sgemm64.sass maxas_sgemm.cubin || (echo "maxas.pl failed $$?"; exit 1)
	$(MAXAS) -i maxas_sgemm128.sass maxas_sgemm.cubin || (echo "maxas.pl failed $$?"; exit 1)
	$(MAXAS) -e -k sgemm_kernel_64 maxas_sgemm.cubin maxas_sgemm64-final.asm
	$(MAXAS) -e -k sgemm_kernel_128 maxas_sgemm.cubin maxas_sgemm128-final.asm

.PHONY: clean

clean:
	rm *~
	rm *.o 
	rm *.cubin
	rm test_maxas
