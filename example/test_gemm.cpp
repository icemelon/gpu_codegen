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
/*
extern "C" __global__ void k0(size_t M,
                              size_t N,
                              size_t K,
                              float* C_pointer,
                              size_t C_ld,
                              float alpha,
                              float4* A_pointer,
                              size_t A_ld,
                              float4* B_pointer,
                              size_t B_ld,
                              float beta)
{
    A_ld /= 4;
    B_ld /= 4;

    float rC[64] = {0};
    float4 rA[2][2];
    float4 rB[2][2];
    __shared__ float lA[2*512];
    __shared__ float lB[2*1024];
    
    unsigned int lAstart = 0;
    unsigned int lBstart = 0;
    
    size_t gidx = blockIdx.x;
    size_t gidy = blockIdx.y;
    size_t idx =  threadIdx.x;
    size_t idy =  threadIdx.y;
    
    size_t idt = 8*idy + idx;
    size_t idxT = idt % 16;
    size_t idyT = idt / 16;
    
    A_pointer += (gidx*16 + idxT) + idyT*A_ld;
    B_pointer += (gidy*32 + idxT) + idyT*B_ld;

    float* lAstore = lA + idyT*64 + idxT*4;
    float* lBstore = lB + idyT*128 + idxT*4;

    reinterpret_cast<float4*>(lAstore + lAstart + 0)[0] = A_pointer[0*A_ld + 0];
    reinterpret_cast<float4*>(lBstore + lBstart +0)[0] = B_pointer[0*B_ld + 0];
    reinterpret_cast<float4*>(lBstore + lBstart +64)[0] = B_pointer[0*B_ld + 16];

    //Outer loop
    for(unsigned int block_k=0; block_k <= K - 8; block_k+=8)
    {
        __syncthreads();
        float* lAread = lA + lAstart + 4*idx;
        float* lBread = lB + lBstart + 4*idy;
        //Inner loop
#pragma unroll
        for(unsigned int k = 0; k < 8; k+=2)
        {
            //Fetch A to registers
            rA[0][0] = reinterpret_cast<float4*>(lAread + k*64 + 0*32+ 0*64)[0];
            rA[0][1] = reinterpret_cast<float4*>(lAread + k*64 + 1*32+ 0*64)[0];
            rA[1][0] = reinterpret_cast<float4*>(lAread + k*64 + 0*32+ 1*64)[0];
            rA[1][1] = reinterpret_cast<float4*>(lAread + k*64 + 1*32+ 1*64)[0];
            
            //Fetch B to registers
            rB[0][0] = reinterpret_cast<float4*>(lBread + k*128 + 0*64+ 0*128)[0];
            rB[0][1] = reinterpret_cast<float4*>(lBread + k*128 + 1*64+ 0*128)[0];
            rB[1][0] = reinterpret_cast<float4*>(lBread + k*128 + 0*64+ 1*128)[0];
            rB[1][1] = reinterpret_cast<float4*>(lBread + k*128 + 1*64+ 1*128)[0];

            //FMA computations
            rC[0]=fma(rA[0][0].x,rB[0][0].x,rC[0]);
            rC[0]=fma(rA[1][0].x,rB[1][0].x,rC[0]);
            rC[8]=fma(rA[0][0].y,rB[0][0].x,rC[8]);
            rC[8]=fma(rA[1][0].y,rB[1][0].x,rC[8]);
            rC[16]=fma(rA[0][0].z,rB[0][0].x,rC[16]);
            rC[16]=fma(rA[1][0].z,rB[1][0].x,rC[16]);
            rC[24]=fma(rA[0][0].w,rB[0][0].x,rC[24]);
            rC[24]=fma(rA[1][0].w,rB[1][0].x,rC[24]);
            rC[32]=fma(rA[0][1].x,rB[0][0].x,rC[32]);
            rC[32]=fma(rA[1][1].x,rB[1][0].x,rC[32]);
            rC[40]=fma(rA[0][1].y,rB[0][0].x,rC[40]);
            rC[40]=fma(rA[1][1].y,rB[1][0].x,rC[40]);
            rC[48]=fma(rA[0][1].z,rB[0][0].x,rC[48]);
            rC[48]=fma(rA[1][1].z,rB[1][0].x,rC[48]);
            rC[56]=fma(rA[0][1].w,rB[0][0].x,rC[56]);
            rC[56]=fma(rA[1][1].w,rB[1][0].x,rC[56]);
            rC[1]=fma(rA[0][0].x,rB[0][0].y,rC[1]);
            rC[1]=fma(rA[1][0].x,rB[1][0].y,rC[1]);
            rC[9]=fma(rA[0][0].y,rB[0][0].y,rC[9]);
            rC[9]=fma(rA[1][0].y,rB[1][0].y,rC[9]);
            rC[17]=fma(rA[0][0].z,rB[0][0].y,rC[17]);
            rC[17]=fma(rA[1][0].z,rB[1][0].y,rC[17]);
            rC[25]=fma(rA[0][0].w,rB[0][0].y,rC[25]);
            rC[25]=fma(rA[1][0].w,rB[1][0].y,rC[25]);
            rC[33]=fma(rA[0][1].x,rB[0][0].y,rC[33]);
            rC[33]=fma(rA[1][1].x,rB[1][0].y,rC[33]);
            rC[41]=fma(rA[0][1].y,rB[0][0].y,rC[41]);
            rC[41]=fma(rA[1][1].y,rB[1][0].y,rC[41]);
            rC[49]=fma(rA[0][1].z,rB[0][0].y,rC[49]);
            rC[49]=fma(rA[1][1].z,rB[1][0].y,rC[49]);
            rC[57]=fma(rA[0][1].w,rB[0][0].y,rC[57]);
            rC[57]=fma(rA[1][1].w,rB[1][0].y,rC[57]);
            rC[2]=fma(rA[0][0].x,rB[0][0].z,rC[2]);
            rC[2]=fma(rA[1][0].x,rB[1][0].z,rC[2]);
            rC[10]=fma(rA[0][0].y,rB[0][0].z,rC[10]);
            rC[10]=fma(rA[1][0].y,rB[1][0].z,rC[10]);
            rC[18]=fma(rA[0][0].z,rB[0][0].z,rC[18]);
            rC[18]=fma(rA[1][0].z,rB[1][0].z,rC[18]);
            rC[26]=fma(rA[0][0].w,rB[0][0].z,rC[26]);
            rC[26]=fma(rA[1][0].w,rB[1][0].z,rC[26]);
            rC[34]=fma(rA[0][1].x,rB[0][0].z,rC[34]);
            rC[34]=fma(rA[1][1].x,rB[1][0].z,rC[34]);
            rC[42]=fma(rA[0][1].y,rB[0][0].z,rC[42]);
            rC[42]=fma(rA[1][1].y,rB[1][0].z,rC[42]);
            rC[50]=fma(rA[0][1].z,rB[0][0].z,rC[50]);
            rC[50]=fma(rA[1][1].z,rB[1][0].z,rC[50]);
            rC[58]=fma(rA[0][1].w,rB[0][0].z,rC[58]);
            rC[58]=fma(rA[1][1].w,rB[1][0].z,rC[58]);
            rC[3]=fma(rA[0][0].x,rB[0][0].w,rC[3]);
            rC[3]=fma(rA[1][0].x,rB[1][0].w,rC[3]);
            rC[11]=fma(rA[0][0].y,rB[0][0].w,rC[11]);
            rC[11]=fma(rA[1][0].y,rB[1][0].w,rC[11]);
            rC[19]=fma(rA[0][0].z,rB[0][0].w,rC[19]);
            rC[19]=fma(rA[1][0].z,rB[1][0].w,rC[19]);
            rC[27]=fma(rA[0][0].w,rB[0][0].w,rC[27]);
            rC[27]=fma(rA[1][0].w,rB[1][0].w,rC[27]);
            rC[35]=fma(rA[0][1].x,rB[0][0].w,rC[35]);
            rC[35]=fma(rA[1][1].x,rB[1][0].w,rC[35]);
            rC[43]=fma(rA[0][1].y,rB[0][0].w,rC[43]);
            rC[43]=fma(rA[1][1].y,rB[1][0].w,rC[43]);
            rC[51]=fma(rA[0][1].z,rB[0][0].w,rC[51]);
            rC[51]=fma(rA[1][1].z,rB[1][0].w,rC[51]);
            rC[59]=fma(rA[0][1].w,rB[0][0].w,rC[59]);
            rC[59]=fma(rA[1][1].w,rB[1][0].w,rC[59]);
            rC[4]=fma(rA[0][0].x,rB[0][1].x,rC[4]);
            rC[4]=fma(rA[1][0].x,rB[1][1].x,rC[4]);
            rC[12]=fma(rA[0][0].y,rB[0][1].x,rC[12]);
            rC[12]=fma(rA[1][0].y,rB[1][1].x,rC[12]);
            rC[20]=fma(rA[0][0].z,rB[0][1].x,rC[20]);
            rC[20]=fma(rA[1][0].z,rB[1][1].x,rC[20]);
            rC[28]=fma(rA[0][0].w,rB[0][1].x,rC[28]);
            rC[28]=fma(rA[1][0].w,rB[1][1].x,rC[28]);
            rC[36]=fma(rA[0][1].x,rB[0][1].x,rC[36]);
            rC[36]=fma(rA[1][1].x,rB[1][1].x,rC[36]);
            rC[44]=fma(rA[0][1].y,rB[0][1].x,rC[44]);
            rC[44]=fma(rA[1][1].y,rB[1][1].x,rC[44]);
            rC[52]=fma(rA[0][1].z,rB[0][1].x,rC[52]);
            rC[52]=fma(rA[1][1].z,rB[1][1].x,rC[52]);
            rC[60]=fma(rA[0][1].w,rB[0][1].x,rC[60]);
            rC[60]=fma(rA[1][1].w,rB[1][1].x,rC[60]);
            rC[5]=fma(rA[0][0].x,rB[0][1].y,rC[5]);
            rC[5]=fma(rA[1][0].x,rB[1][1].y,rC[5]);
            rC[13]=fma(rA[0][0].y,rB[0][1].y,rC[13]);
            rC[13]=fma(rA[1][0].y,rB[1][1].y,rC[13]);
            rC[21]=fma(rA[0][0].z,rB[0][1].y,rC[21]);
            rC[21]=fma(rA[1][0].z,rB[1][1].y,rC[21]);
            rC[29]=fma(rA[0][0].w,rB[0][1].y,rC[29]);
            rC[29]=fma(rA[1][0].w,rB[1][1].y,rC[29]);
            rC[37]=fma(rA[0][1].x,rB[0][1].y,rC[37]);
            rC[37]=fma(rA[1][1].x,rB[1][1].y,rC[37]);
            rC[45]=fma(rA[0][1].y,rB[0][1].y,rC[45]);
            rC[45]=fma(rA[1][1].y,rB[1][1].y,rC[45]);
            rC[53]=fma(rA[0][1].z,rB[0][1].y,rC[53]);
            rC[53]=fma(rA[1][1].z,rB[1][1].y,rC[53]);
            rC[61]=fma(rA[0][1].w,rB[0][1].y,rC[61]);
            rC[61]=fma(rA[1][1].w,rB[1][1].y,rC[61]);
            rC[6]=fma(rA[0][0].x,rB[0][1].z,rC[6]);
            rC[6]=fma(rA[1][0].x,rB[1][1].z,rC[6]);
            rC[14]=fma(rA[0][0].y,rB[0][1].z,rC[14]);
            rC[14]=fma(rA[1][0].y,rB[1][1].z,rC[14]);
            rC[22]=fma(rA[0][0].z,rB[0][1].z,rC[22]);
            rC[22]=fma(rA[1][0].z,rB[1][1].z,rC[22]);
            rC[30]=fma(rA[0][0].w,rB[0][1].z,rC[30]);
            rC[30]=fma(rA[1][0].w,rB[1][1].z,rC[30]);
            rC[38]=fma(rA[0][1].x,rB[0][1].z,rC[38]);
            rC[38]=fma(rA[1][1].x,rB[1][1].z,rC[38]);
            rC[46]=fma(rA[0][1].y,rB[0][1].z,rC[46]);
            rC[46]=fma(rA[1][1].y,rB[1][1].z,rC[46]);
            rC[54]=fma(rA[0][1].z,rB[0][1].z,rC[54]);
            rC[54]=fma(rA[1][1].z,rB[1][1].z,rC[54]);
            rC[62]=fma(rA[0][1].w,rB[0][1].z,rC[62]);
            rC[62]=fma(rA[1][1].w,rB[1][1].z,rC[62]);
            rC[7]=fma(rA[0][0].x,rB[0][1].w,rC[7]);
            rC[7]=fma(rA[1][0].x,rB[1][1].w,rC[7]);
            rC[15]=fma(rA[0][0].y,rB[0][1].w,rC[15]);
            rC[15]=fma(rA[1][0].y,rB[1][1].w,rC[15]);
            rC[23]=fma(rA[0][0].z,rB[0][1].w,rC[23]);
            rC[23]=fma(rA[1][0].z,rB[1][1].w,rC[23]);
            rC[31]=fma(rA[0][0].w,rB[0][1].w,rC[31]);
            rC[31]=fma(rA[1][0].w,rB[1][1].w,rC[31]);
            rC[39]=fma(rA[0][1].x,rB[0][1].w,rC[39]);
            rC[39]=fma(rA[1][1].x,rB[1][1].w,rC[39]);
            rC[47]=fma(rA[0][1].y,rB[0][1].w,rC[47]);
            rC[47]=fma(rA[1][1].y,rB[1][1].w,rC[47]);
            rC[55]=fma(rA[0][1].z,rB[0][1].w,rC[55]);
            rC[55]=fma(rA[1][1].z,rB[1][1].w,rC[55]);
            rC[63]=fma(rA[0][1].w,rB[0][1].w,rC[63]);
            rC[63]=fma(rA[1][1].w,rB[1][1].w,rC[63]);
        }
        lAstart ^= 512;
        lBstart ^= 1024;
        A_pointer += 8*A_ld;
        B_pointer += 8*B_ld;
        //Fetch A to local memory
        reinterpret_cast<float4*>(lAstore + lAstart + 0)[0] = A_pointer[0*A_ld + 0];
        //Fetch B to local memory
        reinterpret_cast<float4*>(lBstore + lBstart + 0)[0] = B_pointer[0*B_ld + 0];
        reinterpret_cast<float4*>(lBstore + lBstart + 64)[0] = B_pointer[0*B_ld + 16];
    }
    //Write back C
    C_pointer += (gidx*64 + idx*4) + (gidy*128 + idy*4)*C_ld;
    C_pointer[0+0*C_ld] = rC[0]*alpha;
    C_pointer[0+1*C_ld] = rC[1]*alpha;
    C_pointer[0+2*C_ld] = rC[2]*alpha;
    C_pointer[0+3*C_ld] = rC[3]*alpha;
    C_pointer[0+64*C_ld] = rC[4]*alpha;
    C_pointer[0+65*C_ld] = rC[5]*alpha;
    C_pointer[0+66*C_ld] = rC[6]*alpha;
    C_pointer[0+67*C_ld] = rC[7]*alpha;
    C_pointer[1+0*C_ld] = rC[8]*alpha;
    C_pointer[1+1*C_ld] = rC[9]*alpha;
    C_pointer[1+2*C_ld] = rC[10]*alpha;
    C_pointer[1+3*C_ld] = rC[11]*alpha;
    C_pointer[1+64*C_ld] = rC[12]*alpha;
    C_pointer[1+65*C_ld] = rC[13]*alpha;
    C_pointer[1+66*C_ld] = rC[14]*alpha;
    C_pointer[1+67*C_ld] = rC[15]*alpha;
    C_pointer[2+0*C_ld] = rC[16]*alpha;
    C_pointer[2+1*C_ld] = rC[17]*alpha;
    C_pointer[2+2*C_ld] = rC[18]*alpha;
    C_pointer[2+3*C_ld] = rC[19]*alpha;
    C_pointer[2+64*C_ld] = rC[20]*alpha;
    C_pointer[2+65*C_ld] = rC[21]*alpha;
    C_pointer[2+66*C_ld] = rC[22]*alpha;
    C_pointer[2+67*C_ld] = rC[23]*alpha;
    C_pointer[3+0*C_ld] = rC[24]*alpha;
    C_pointer[3+1*C_ld] = rC[25]*alpha;
    C_pointer[3+2*C_ld] = rC[26]*alpha;
    C_pointer[3+3*C_ld] = rC[27]*alpha;
    C_pointer[3+64*C_ld] = rC[28]*alpha;
    C_pointer[3+65*C_ld] = rC[29]*alpha;
    C_pointer[3+66*C_ld] = rC[30]*alpha;
    C_pointer[3+67*C_ld] = rC[31]*alpha;
    C_pointer[32+0*C_ld] = rC[32]*alpha;
    C_pointer[32+1*C_ld] = rC[33]*alpha;
    C_pointer[32+2*C_ld] = rC[34]*alpha;
    C_pointer[32+3*C_ld] = rC[35]*alpha;
    C_pointer[32+64*C_ld] = rC[36]*alpha;
    C_pointer[32+65*C_ld] = rC[37]*alpha;
    C_pointer[32+66*C_ld] = rC[38]*alpha;
    C_pointer[32+67*C_ld] = rC[39]*alpha;
    C_pointer[33+0*C_ld] = rC[40]*alpha;
    C_pointer[33+1*C_ld] = rC[41]*alpha;
    C_pointer[33+2*C_ld] = rC[42]*alpha;
    C_pointer[33+3*C_ld] = rC[43]*alpha;
    C_pointer[33+64*C_ld] = rC[44]*alpha;
    C_pointer[33+65*C_ld] = rC[45]*alpha;
    C_pointer[33+66*C_ld] = rC[46]*alpha;
    C_pointer[33+67*C_ld] = rC[47]*alpha;
    C_pointer[34+0*C_ld] = rC[48]*alpha;
    C_pointer[34+1*C_ld] = rC[49]*alpha;
    C_pointer[34+2*C_ld] = rC[50]*alpha;
    C_pointer[34+3*C_ld] = rC[51]*alpha;
    C_pointer[34+64*C_ld] = rC[52]*alpha;
    C_pointer[34+65*C_ld] = rC[53]*alpha;
    C_pointer[34+66*C_ld] = rC[54]*alpha;
    C_pointer[34+67*C_ld] = rC[55]*alpha;
    C_pointer[35+0*C_ld] = rC[56]*alpha;
    C_pointer[35+1*C_ld] = rC[57]*alpha;
    C_pointer[35+2*C_ld] = rC[58]*alpha;
    C_pointer[35+3*C_ld] = rC[59]*alpha;
    C_pointer[35+64*C_ld] = rC[60]*alpha;
    C_pointer[35+65*C_ld] = rC[61]*alpha;
    C_pointer[35+66*C_ld] = rC[62]*alpha;
    C_pointer[35+67*C_ld] = rC[63]*alpha;
}
*/

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
    printf("  kernel:   gemm64_1 | gemm64_2\n");
    return 0;
  }
  const char* kernel = argv[1];
  int N = atoi(argv[2]);

  int device = 0;
  cudaSetDevice(device);
  srand(1);
  
  float *A, *B, *C, *hostC;
  float *devA, *devB, *devC;
  size_t size = N * N * sizeof(float);
  float alpha = 1.0, beta = 0.0;

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
  CUDA_CHECK( cudaMalloc((void**)&devA, size) );
  CUDA_CHECK( cudaMalloc((void**)&devB, size) );
  CUDA_CHECK( cudaMalloc((void**)&devC, size) );
  CUDA_CHECK( cudaMemcpy(devA, A, size, cudaMemcpyHostToDevice) );
  CUDA_CHECK( cudaMemcpy(devB, B, size, cudaMemcpyHostToDevice) );

  /*CUDA_CHECK( cuTexRefSetFormat(texA, CU_AD_FORMAT_FLOAT, 4) );
  CUDA_CHECK( cuTexRefSetFormat(texB, CU_AD_FORMAT_FLOAT, 4) );
  CUDA_CHECK( cuTexRefSetAddress(NULL, texA, devA, size) );
  CUDA_CHECK( cuTexRefSetAddress(NULL, texB, devB, size) );*/
  //CUDA_CHECK( cudaBindTexture(NULL, texA, devA, size) );
  //CUDA_CHECK( cudaBindTexture(NULL, texB, devB, size) );

  CUmodule hModule;
  CUtexref texA, texB;
  CUfunction hKernel;

  cudaProfilerStart();
  if (!strcmp(kernel, "gemm64_1")) {
    CUDA_CHECK( cuModuleLoad(&hModule, "gemm64_1.cubin") );
    CUDA_CHECK( cuModuleGetFunction(&hKernel, hModule, kernel) );
    void* params[] = {&N, &N, &N, &devA, &N, &devB, &N, &devC, &N, &alpha, &beta};
    int block_x, block_y, grid_x, grid_y;
    block_x = 8; 
    block_y = 8;
    grid_x = N / 64;
    grid_y = N / 64;
    CUDA_CHECK( cuLaunchKernel(hKernel, grid_x, grid_y, 1, block_x, block_y, 1, 0, 0, params, 0) );
    //gemm64_0<<<grid_size, block_size>>>(N, N, N, (float4*)devA, N, (float4*)devB, N, devC, N, 1.0, 0.0);
  } else if (!strcmp(kernel, "gemm64_2")) {
    /*dim3 block_size, grid_size;
    block_size.x = 64;
    grid_size.x = N / 64;
    grid_size.y = N / 64;
    gemm64_1<<<grid_size, block_size>>>(N, N, N, N, N, devC, N, 1.0, 0.0);*/
  } /*else if (kernel == 2) {
    dim3 block_size, grid_size;
    block_size.x = 8;
    block_size.y = 16;
    grid_size.x = N / 64;
    grid_size.y = N / 128;
    k0<<<grid_size, block_size>>>(N, N, N, devC, N, 1, (float4*)devA, N, (float4*)devB, N, 0);
  }*/ else {
    printf("Wrong kernel!\n");
    return 1;
  }

  hostC = (float*)malloc(size);
  CUDA_CHECK( cudaMemcpy(hostC, devC, size, cudaMemcpyDeviceToHost) );
  cudaProfilerStop();

  //sleep(1);
  //return 0;
  gemm(N, N, N, A, N, B, N, C, N, 1.0, 0.0);

 
  /*print_mat(C, N, N, 0);
  printf("\n");
  print_mat(hostC, N, N, 0);*/
  printf("%d\n", check_rel_close(C, hostC, N, N));

  return 0;
}
