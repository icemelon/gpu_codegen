#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda.h>
#include <cuda_runtime.h>

#define REPEAT_BLOCK 2000

#define CUDA_CHECK( fn ) do { \
		int status = (fn); \
		if ( CUDA_SUCCESS != status ) { \
			printf("CUDA Failure (line %d of file %s):\n\t%s returned %d\n", __LINE__, __FILE__, #fn, status); \
			exit(EXIT_FAILURE); \
		} \
	} while (0)

#define CUBLAS_CHECK( fn ) do { \
		cublasStatus_t status = (fn); \
		if ( CUBLAS_STATUS_SUCCESS != status ) { \
			printf("Cublas Failure (line %d of file %s):\n\t%s returned %d\n", __LINE__, __FILE__, #fn, status); \
			exit(EXIT_FAILURE); \
		} \
	} while (0)

float assemblySgemm(const char* kernel, size_t size, CUdeviceptr devC, CUdeviceptr devA, CUdeviceptr devB, int N, int repeat = 1, int printVars = 0);
void gflops(const char* ident, int N, float ms, int repeat);

int main(int argc, char* argv[]) {
    char deviceName[32];
	int count, ordinal, major, minor;
	CUdevice  hDevice;
	CUdeviceptr devA, devB, devC; //, devT, otherDevA, otherDevB;

    cudaSetDevice(1);
	int thread128 = 64;
	int repeat = 1;
    int printVars = 0;

	int N = thread128 * 128;
	float alpha = 1, beta = 0, ms = 1;
	size_t sizeOther = N * N;
	size_t sizeFloat = sizeOther * 4;

	float* A = (float*)malloc(sizeFloat);
	float* B = (float*)malloc(sizeFloat);
	float* C = (float*)malloc(sizeFloat);
	//float* T = (float*)malloc(sizeFloat);  
	//float *otherA, *otherB; 

	//int counter = 0;
	//srand((unsigned int)time(0));
	for(int i = 0; i < N * N; i++) //
	{
		//A[i] = (float)rand() / (float)RAND_MAX;
		//B[i] = (float)rand() / (float)RAND_MAX;
		A[i] = B[i] = 1.0f; // * (i & 3) + 1.0f;
		//A[i] = 1.0f;
		//B[i * N + counter++] = 1.0f; // identity matrix
	}

	//CUDA_CHECK( cuCtxCreate(&hContext, 0, hDevice) );
	//CUBLAS_CHECK( cublasCreate(&hCublas) );
	

	CUDA_CHECK( cudaMalloc((void **)&devA, sizeFloat) );
	CUDA_CHECK( cudaMalloc((void **)&devB, sizeFloat) );
	CUDA_CHECK( cudaMalloc((void **)&devC, sizeFloat) );
	//CUDA_CHECK( cudaAlloc(&devT, sizeFloat) );
	
	CUDA_CHECK( cudaMemcpy((void *)devA, A, sizeFloat, cudaMemcpyHostToDevice) );
    CUDA_CHECK( cudaMemcpy((void *)devB, B, sizeFloat, cudaMemcpyHostToDevice) );
	CUDA_CHECK( cudaMemset((void *)devC, 0, sizeFloat) );
	//CUDA_CHECK( cudaMemset(devT, 0, sizeFloat) );

	// Warm up the clock (unless under nsight)
	//if (!getenv("NSIGHT_LAUNCHED")) // NSIGHT_CUDA_ANALYSIS NSIGHT_CUDA_DEBUGGER 
	//	for (int i = 0; i < 3; i++)
	//		CUBLAS_CHECK( cublasSgemm(hCublas, CUBLAS_OP_N, CUBLAS_OP_T, N, N, N, &alpha, reinterpret_cast<float*>(devA), N, reinterpret_cast<float*>(devB), N, &beta, reinterpret_cast<float*>(devT), N) );
    
	// Launch our kernel
	ms = assemblySgemm("sgemm_kernel_64", sizeOther, devC, devA, devB, N, repeat, printVars);
	gflops("Max64 ", N, ms, repeat);

	/*ms = assemblySgemm("sgemm_kernel_128", sizeOther, devC, devA, devB, N, repeat, printVars);
      gflops("Max128", N, ms, repeat);*/

	//ms = cublasSgemm("maxwell_sgemm_128x64_nt", devT, devA, devB, N, hStart, hStop, repeat);
	//gflops("Cub64 ", N, ms, repeat);

	//ms = cublasSgemm("maxwell_sgemm_128x128_nt", devT, devA, devB, N, hStart, hStop, repeat);
	//gflops("Cub128", N, ms, repeat);

	// Run cublas again for the same repeat count for comparison
	//CUDA_CHECK( cuEventRecord(hStart, NULL) );
	//for (int i = 0; i < repeat; i++)
	//	CUBLAS_CHECK( cublasSgemm(hCublas, CUBLAS_OP_N, CUBLAS_OP_T, N, N, N, &alpha, reinterpret_cast<float*>(devA), N, reinterpret_cast<float*>(devB), N, &beta, reinterpret_cast<float*>(devT), N) );
	//CUDA_CHECK( cuEventRecord(hStop, NULL) );
	//CUDA_CHECK( cuEventSynchronize(hStop) );
	//CUDA_CHECK( cuEventElapsedTime(&ms, hStart, hStop) );
	//gflops("Cublas", N, ms, repeat);

	// Get back our results from each kernel
	CUDA_CHECK( cudaMemcpy(C, (const void *)devC, sizeFloat, cudaMemcpyDeviceToHost) );
	//CUDA_CHECK( cuMemcpyDtoH(T, devT, sizeFloat) );
	
	// Cleanup and shutdown of cuda
	CUDA_CHECK( cudaFree((void *)devA) );
	CUDA_CHECK( cudaFree((void *)devB) );
	CUDA_CHECK( cudaFree((void *)devC) );
	//CUDA_CHECK( cudaFree(devT) );


	//CUBLAS_CHECK( cublasDestroy(hCublas) );
	//hCublas  = 0;
	//CUDA_CHECK( cuCtxDestroy(hContext) );
	//hContext = 0;

	// compare C and T for accuracy
	//test(C, T, N, sizeFloat);

	// And free up host memory
	free(A); free(B); free(C);// free(T);

	return 0;
}


// Our kernel wrapper function
float assemblySgemm(const char* kernel, size_t size, CUdeviceptr devC, CUdeviceptr devA, CUdeviceptr devB, int N, int repeat, int printVars)
{
	// Configure our x and y grid dimensions (assume nice square matrixes).
	// Each block gets 128 tracks from A and 128 tracks from B.
	// Each of the 256 threads calculates 64 elements of that 128x128 sub matrix of C.
	// See Figure 2 here to get the gist of things (we use a different mapping to maximize LDS.128 usage):
	// http://icl.cs.utk.edu/projectsfiles/magma/pubs/fermi_gemm.pdf
	cudaEvent_t hStart, hStop;
	int threads, width;
    
    CUDA_CHECK( cudaEventCreate(&hStart, CU_EVENT_BLOCKING_SYNC) ); // CU_EVENT_DEFAULT 
	CUDA_CHECK( cudaEventCreate(&hStop,  CU_EVENT_BLOCKING_SYNC) );

	if (strcmp(kernel, "sgemm_kernel_64") == 0)
	{
		threads = 64;
		width   = 64;
	}
	else
	{
		threads = 256;
		width   = 128;
	}

	int gridDimXY = N / width + (N % width != 0);
	int blocks    = gridDimXY * gridDimXY;

	// Setup out debug printf output buffer
	//CUdeviceptr devD = NULL;
    CUdeviceptr devD; 
	int* D = NULL;
	int  sizeD = 0;

	if (printVars)
	{
		sizeD = blocks * threads * printVars * sizeof(int);
		D = (int*)malloc(sizeD);

		CUDA_CHECK( cuMemAlloc(&devD, sizeD) );
		CUDA_CHECK( cuMemsetD8(devD, 0, sizeD) );
	}

	// Load the cubin
	CUmodule hModule;
	CUDA_CHECK( cuModuleLoad(&hModule, "maxas_sgemm.cubin") );

	// Load the textures
	CUtexref texA, texB;
	CUDA_CHECK( cuModuleGetTexRef(&texA, hModule, "texA") );
	CUDA_CHECK( cuModuleGetTexRef(&texB, hModule, "texB") );

	// Configure the textures
	CUDA_CHECK( cuTexRefSetFormat(texA, CU_AD_FORMAT_FLOAT, 4) );
	CUDA_CHECK( cuTexRefSetFormat(texB, CU_AD_FORMAT_FLOAT, 4) );

	CUDA_CHECK( cuTexRefSetAddress(NULL, texA, devA, size) );
	CUDA_CHECK( cuTexRefSetAddress(NULL, texB, devB, size) );

	// Load the kernel function
	CUfunction hKernel;
	CUDA_CHECK( cuModuleGetFunction(&hKernel, hModule, kernel) );

	// Setup the params
	float alpha = 1.0f;
	void* params[] = { &devC, &N, &N, &N, &N, &N, &N, &alpha, &devD };

	float totalTime = 0;
	// Launch the kernel repeat times.. but break it up into pieces so as not to lock things up.
	while (repeat > 0)
	{
        printf("repeat left %d\n", repeat);
		float ms;
		int r = repeat > REPEAT_BLOCK ? REPEAT_BLOCK : repeat;
		CUDA_CHECK( cudaEventRecord( hStart, NULL ) );
		
		for (int i = 0; i < r; i++)
			CUDA_CHECK( cuLaunchKernel(hKernel, gridDimXY, gridDimXY, 1, threads, 1, 1, 0, 0, params, 0) );
		
		CUDA_CHECK( cudaEventRecord( hStop, NULL ) );
		CUDA_CHECK( cudaEventSynchronize( hStop ) );
		CUDA_CHECK( cudaEventElapsedTime( &ms, hStart, hStop ) );
        printf("ms %f\n", ms);
		totalTime += ms;
		repeat -= r;
	}

	CUDA_CHECK( cudaEventDestroy(hStart) );
	CUDA_CHECK( cudaEventDestroy(hStop) );
	CUDA_CHECK( cuModuleUnload(hModule) );

	// And here we print out the debug info if requested:
	if (printVars)
	{
		CUDA_CHECK( cuMemcpyDtoH(D, devD, sizeD) );
		CUDA_CHECK( cuMemFree(devD) );
		int   *iD = D;
		float *fD = reinterpret_cast<float*>(D);
		unsigned int *uD = reinterpret_cast<unsigned int*>(D);

		for (int by = 0; by < gridDimXY; by++)
		{
			for (int bx = 0; bx < gridDimXY; bx++)
			{
				unsigned int clock = 0xffffffff, sm = 0;

				for (int tid = 0; tid < threads; tid++)
				{
					//printf("by: %3d, bx: %3d, tid:%3d, rA:%5d, rB:%5d, wr:%5d, rd:%5d, cx:%5d, cy:%5d, ci:%5d, c:%.2f\n", 
					//printf("by: %3d, bx: %3d, tid:%3d, t0:%5d, end:%5d, k:%5d, tid2:%5d, tid15:%5d, ldx:%5d, t2:%5d, t4:%5d\n", 
					//	    by,      bx,      tid,     iD[0],  iD[1],   iD[2], iD[3],    iD[4],     iD[5],   iD[6],  iD[7]
					//);
					if (uD[1] < clock) clock = uD[1];
					sm = uD[0];

					iD += printVars;
					fD += printVars;
					uD += printVars;
				}
				printf("%02d %08u %d %d\n", sm, clock, by, bx);
			}
		}
		free(D);
	}
    
	return totalTime;
}


void gflops(const char* ident, int N, float ms, int repeat)
{
	// Standard sgemm flops formula
	ms /= repeat;
	printf("%s GFLOPS: %.2f (size: %d, iterations: %d)\n", ident, ((double)N * N * N * 2.0 + N * N) / (ms * 1000000.0), N, repeat);
}
