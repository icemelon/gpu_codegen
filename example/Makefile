CC=gcc
CXX=g++
NVCC=nvcc
MAXAS=maxas.pl

CUDA_PATH=/usr/local/cuda
INCLUDE=-I. -I$(CUDA_PATH)/include

CPPFLAGS=-O3 -g -std=c++11 $(INCLUDE) # -Wall -Werror
NVCCFLAGS=-O3 -arch sm_50 -std=c++11 -lineinfo -Xcompiler $(INCLUDE)
LDFLAGS=-g -L. -L$(CUDA_PATH)/lib64
LDLIBS=-lcuda -lcudart -lcublas

all: test_maxas test_gemm

test_gemm: test_gemm.cpp maxas_sgemm.cubin gemm64_1.cubin gemm64_2.cubin gemm64_3.cubin
	$(CXX) $(CPPFLAGS) -o $@ $< $(LDFLAGS) $(LDLIBS)

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

gemm64_1.cubin: gemm64_1.cu
	$(NVCC) $(NVCCFLAGS) -cubin $<
	$(MAXAS) -e gemm64_1.cubin gemm64_1.asm

gemm64_2.cubin: gemm64_2.cu
	$(NVCC) $(NVCCFLAGS) -cubin $<
	$(MAXAS) -e gemm64_2.cubin gemm64_2.asm

gemm64_3.cubin: gemm64_3.cu
	$(NVCC) $(NVCCFLAGS) -cubin $<
	$(MAXAS) -e gemm64_3.cubin gemm64_3.asm

.PHONY: clean

clean:
	rm *.o 
	rm *.cubin
	rm test_gemm
	rm test_maxas
