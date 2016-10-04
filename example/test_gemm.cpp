#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
//#include <cuda_texture_types.H>

#define CUDA_CHECK( f ) do { \
    int status = (f); \
	if ( CUDA_SUCCESS != status ) { \
	    printf("CUDA Failure (line %d of file %s):\n\t%s returned %d\n", __LINE__, __FILE__, #f, status); \
		exit(EXIT_FAILURE); \
	} \
} while (0)

void gemm(int M, int N, int K, const float* A, int LDA, const float* BT,
          int LDB, float* C, int LDC, float alpha, float beta) {
  for (int x = 0; x < M; ++x) {
    for (int y = 0; y < N; ++y) {
      C[x + y*LDC] *= beta;
      for (int k = 0; k < K; ++k) {
        C[x + y*LDC] += A[x + k*LDA] * BT[k*LDB + y];
      }
      C[x + y*LDC] *= alpha;
    }
  }
}

// mode: 0 column major, 1 row major
void print_mat(float *A, int M, int N, int mode) {
  if (mode == 0) {
    for (int r = 0; r < M; ++r) {
      for (int c = 0; c < N; ++c) {
        printf("%.2f ", A[r + c*M]);
      }
      printf("\n");
    }
  } else {
    for (int r = 0; r < M; ++r) {
      for (int c = 0; c < N; ++c) {
        printf("%.2f ", A[r*N + c]);
      }
      printf("\n");
    }
  }
}

bool are_rel_close(float a, float b, float tolerance) {
  if (fabs(a) < 0.1) {
    return (b >= (a - tolerance)) && (b <= (a + tolerance));
  }
  float d = fabs(a) * tolerance;
  return (b >= (a - d)) && (b <= (a + d));
}

#define TOLERANCE 1e-2
bool check_rel_close(float *X, float *Y, int M, int N) {
  for (int i = 0; i < M * N; ++i) {
    if (!are_rel_close(X[i], Y[i], TOLERANCE)) {
      printf("[%d] %f <-> %f\n", i, X[i], Y[i]);
      return false;
    }
  }
  return true;
}

int main(int argc, const char **argv) {
  if (argc < 3) {
    printf("%s kernel N\n", argv[0]);
    printf("  kernel:   maxas | gemm64_1 | gemm64_2\n");
    return 0;
  }
  const char* kernel = argv[1];
  int N = atoi(argv[2]);

  int device = 1;
  //cudaSetDevice(device);
  srand(1);
  
  float *A, *B, *C, *hostC;
  //float *devA, *devB, *devC;
  CUdevice hDevice;
  CUcontext hContext;
  CUdeviceptr devA, devB, devC;
  size_t size = N * N * sizeof(float);
  float alpha = 1.0, beta = 0.0;

  CUDA_CHECK( cuInit(0) );
  CUDA_CHECK( cuDeviceGet(&hDevice, device) );
  CUDA_CHECK( cuCtxCreate(&hContext, 0, hDevice) );

  // alloc & init A, B, C
  A = (float*)malloc(size);
  B = (float*)malloc(size);
  C = (float*)malloc(size);
  for (int r = 0; r < N; ++r) {
    for (int c = 0; c < N; ++c) {
      /*
      A[r + c*N] = r*10 + c*0.1;
      B[r*N + c] = -(r*10 + c*0.1);*/
      A[r + c*N] = (float(rand()) / RAND_MAX - 0.5) * 2;
      B[r*N + c] = (float(rand()) / RAND_MAX - 0.5) * 2;
    }
  }

  /*
  print_mat(A, N, N, 0);
  printf("\n");
  print_mat(B, N, N, 1);
  printf("\n");
  return 0;*/

  // alloc & init devA, devB, devC
  /*
  CUDA_CHECK( cudaMalloc((void**)&devA, size) );
  CUDA_CHECK( cudaMalloc((void**)&devB, size) );
  CUDA_CHECK( cudaMalloc((void**)&devC, size) );
  CUDA_CHECK( cudaMemcpy(devA, A, size, cudaMemcpyHostToDevice) );
  CUDA_CHECK( cudaMemcpy(devB, B, size, cudaMemcpyHostToDevice) );*/

  CUDA_CHECK( cuMemAlloc(&devA, size) );
  CUDA_CHECK( cuMemAlloc(&devB, size) );
  CUDA_CHECK( cuMemAlloc(&devC, size) );
  CUDA_CHECK( cuMemcpyHtoD(devA, A, size) );
  CUDA_CHECK( cuMemcpyHtoD(devB, B, size) );

  CUmodule hModule;
  CUtexref texA, texB;
  CUfunction hKernel;
  cudaEvent_t hStart, hStop;
  CUDA_CHECK( cudaEventCreate(&hStart, CU_EVENT_BLOCKING_SYNC) );
  CUDA_CHECK( cudaEventCreate(&hStop, CU_EVENT_BLOCKING_SYNC) );

  cudaProfilerStart();
  if (!strcmp(kernel, "maxas")) {
    // Load the module
    CUDA_CHECK( cuModuleLoad(&hModule, "maxas_sgemm.cubin") );
    // Load the kernel function
    CUDA_CHECK( cuModuleGetFunction(&hKernel, hModule, "sgemm_kernel_64") );
    // Load the textures
    CUDA_CHECK( cuModuleGetTexRef(&texA, hModule, "texA") );
	CUDA_CHECK( cuModuleGetTexRef(&texB, hModule, "texB") );
    // Configure the textures
    CUDA_CHECK( cuTexRefSetFormat(texA, CU_AD_FORMAT_FLOAT, 4) );
	CUDA_CHECK( cuTexRefSetFormat(texB, CU_AD_FORMAT_FLOAT, 4) );
	CUDA_CHECK( cuTexRefSetAddress(NULL, texA, (CUdeviceptr)devA, size) );
	CUDA_CHECK( cuTexRefSetAddress(NULL, texB, (CUdeviceptr)devB, size) );
    // Set up the params and dims
    void* params[] = {&devC, &N, &N, &N, &N, &N, &N, &alpha};
    int block_x, grid_x, grid_y;
    block_x = 64; 
    grid_x = grid_y = (N + 63) / 64;
    // Launch the kernel
    CUDA_CHECK( cudaEventRecord(hStart, NULL) );
    CUDA_CHECK( cuLaunchKernel(hKernel, grid_x, grid_y, 1, block_x, 1, 1, 0, 0, params, 0) );
    CUDA_CHECK( cudaEventRecord(hStop, NULL) );
    CUDA_CHECK( cudaEventSynchronize(hStop) );
  } else if (!strcmp(kernel, "gemm64_1")) {
    // Load the module and kernel function
    CUDA_CHECK( cuModuleLoad(&hModule, "gemm64_1.cubin") );
    CUDA_CHECK( cuModuleGetFunction(&hKernel, hModule, kernel) );
    // Set up the params and dims
    void* params[] = {&N, &N, &N, &devA, &N, &devB, &N, &devC, &N, &alpha, &beta};
    int block_x, block_y, grid_x, grid_y;
    block_x = block_y = 8; 
    grid_x = grid_y = (N + 63) / 64;
    // Launch the kernel
    CUDA_CHECK( cudaEventRecord(hStart, NULL) );
    CUDA_CHECK( cuLaunchKernel(hKernel, grid_x, grid_y, 1, block_x, block_y, 1, 0, 0, params, 0) );
    //gemm64_0<<<grid_size, block_size>>>(N, N, N, (float4*)devA, N, (float4*)devB, N, devC, N, 1.0, 0.0);
    CUDA_CHECK( cudaEventRecord(hStop, NULL) );
    CUDA_CHECK( cudaEventSynchronize(hStop) );
  } else if (!strcmp(kernel, "gemm64_2")) {
    // Load the module
    CUDA_CHECK( cuModuleLoad(&hModule, "gemm64_2.cubin") );
    // Load the kernel function
    CUDA_CHECK( cuModuleGetFunction(&hKernel, hModule, "gemm64_2") );
    // Load the textures
    CUDA_CHECK( cuModuleGetTexRef(&texA, hModule, "texA") );
	CUDA_CHECK( cuModuleGetTexRef(&texB, hModule, "texB") );
    // Configure the textures
    CUDA_CHECK( cuTexRefSetFormat(texA, CU_AD_FORMAT_FLOAT, 4) );
	CUDA_CHECK( cuTexRefSetFormat(texB, CU_AD_FORMAT_FLOAT, 4) );
	CUDA_CHECK( cuTexRefSetAddress(NULL, texA, devA, size) );
	CUDA_CHECK( cuTexRefSetAddress(NULL, texB, devB, size) );
    // Set up the params and dims
    void* params[] = {&N, &N, &N, &N, &N, &devC, &N, &alpha, &beta};
    int block_x, grid_x, grid_y;
    block_x = 64; 
    grid_x = grid_y = (N + 63) / 64;
    CUDA_CHECK( cudaEventRecord(hStart, NULL) );
    CUDA_CHECK( cuLaunchKernel(hKernel, grid_x, grid_y, 1, block_x, 1, 1, 0, 0, params, 0) );
    CUDA_CHECK( cudaEventRecord(hStop, NULL) );
    CUDA_CHECK( cudaEventSynchronize(hStop) );
    /*
    dim3 block_size, grid_size;
    block_size.x = 64;
    grid_size.x = N / 64;
    grid_size.y = N / 64;
    CUDA_CHECK( cudaBindTexture(NULL, texA, devA, size) );
    CUDA_CHECK( cudaBindTexture(NULL, texB, devB, size) );
    gemm64_2<<<grid_size, block_size>>>(N, N, N, N, N, devC, N, 1.0, 0.0);*/
  } else if (!strcmp(kernel, "gemm64_3")) {
    // Load the module
    CUDA_CHECK( cuModuleLoad(&hModule, "gemm64_3.cubin") );
    // Load the kernel function
    CUDA_CHECK( cuModuleGetFunction(&hKernel, hModule, "gemm64_3") );
    // Load the textures
    CUDA_CHECK( cuModuleGetTexRef(&texA, hModule, "texA") );
	CUDA_CHECK( cuModuleGetTexRef(&texB, hModule, "texB") );
    // Configure the textures
    CUDA_CHECK( cuTexRefSetFormat(texA, CU_AD_FORMAT_FLOAT, 4) );
	CUDA_CHECK( cuTexRefSetFormat(texB, CU_AD_FORMAT_FLOAT, 4) );
	CUDA_CHECK( cuTexRefSetAddress(NULL, texA, devA, size) );
	CUDA_CHECK( cuTexRefSetAddress(NULL, texB, devB, size) );
    // Set up the params and dims
    void* params[] = {&N, &N, &N, &N, &N, &devC, &N, &alpha, &beta};
    int block_x, block_y, grid_x, grid_y;
    block_x = block_y = 8; 
    grid_x = grid_y = (N + 63) / 64;
    CUDA_CHECK( cudaEventRecord(hStart, NULL) );
    CUDA_CHECK( cuLaunchKernel(hKernel, grid_x, grid_y, 1, block_x, block_y, 1, 0, 0, params, 0) );
    CUDA_CHECK( cudaEventRecord(hStop, NULL) );
    CUDA_CHECK( cudaEventSynchronize(hStop) );
    /*
    dim3 block_size, grid_size;
    block_size.x = 64;
    grid_size.x = N / 64;
    grid_size.y = N / 64;
    CUDA_CHECK( cudaBindTexture(NULL, texA, devA, size) );
    CUDA_CHECK( cudaBindTexture(NULL, texB, devB, size) );
    gemm64_2<<<grid_size, block_size>>>(N, N, N, N, N, devC, N, 1.0, 0.0);*/
  } else {
    printf("Wrong kernel!\n");
    return 1;
  }

  float ms;
  CUDA_CHECK( cudaEventElapsedTime(&ms, hStart, hStop) );
  printf("Kernel %s latency %f\n", kernel, ms);

  hostC = (float*)malloc(size);
  //CUDA_CHECK( cudaMemcpy(hostC, devC, size, cudaMemcpyDeviceToHost) );
  CUDA_CHECK( cuMemcpyDtoH(hostC, devC, size) );
  cudaProfilerStop();
  return 0;

  // verify the result
  gemm(N, N, N, A, N, B, N, C, N, 1.0, 0.0);
  printf("correct: %d\n", check_rel_close(C, hostC, N, N));
  /*print_mat(C, N, N, 0);
  printf("\n");
  print_mat(hostC, N, N, 0);*/

  return 0;
}
