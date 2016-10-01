#include <cstdio>
#include <cuda.h>
#include <cuda_profiler_api.h>
//#include <cuda_texture_types.H>

#define CUDA_CHECK( fn ) do { \
		int status = (fn); \
		if ( CUDA_SUCCESS != status ) { \
			printf("CUDA Failure (line %d of file %s):\n\t%s returned %d\n", __LINE__, __FILE__, #fn, status); \
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


extern "C"
__global__ void __launch_bounds__(64) gemm64_0(
    size_t M, size_t N, size_t K,
    float4* A, int LDA,
    float4* B, int LDB,
    float* C, int LDC,
    float alpha, float beta) {
  
  LDA /= 4;
  LDB /= 4;
  float rC[64] = {0};
  float4 rA[2][2];
  float4 rB[2][2];
  __shared__ float lA[2*512];
  __shared__ float lB[2*512];
    
  unsigned int lAstart = 0;
  unsigned int lBstart = 0;
    
  size_t gidx = blockIdx.x;
  size_t gidy = blockIdx.y;
  size_t idx =  threadIdx.x;
  size_t idy =  threadIdx.y;
    
  A += (gidx*16 + idx) + idy*LDA;
  B += (gidy*16 + idx) + idy*LDB;
  
  float* lAstore = lA + idy*64 + idx*4;
  float* lBstore = lB + idy*64 + idx*4;

  reinterpret_cast<float4*>(lAstore + lAstart +  0)[0] = A[0];
  reinterpret_cast<float4*>(lAstore + lAstart + 32)[0] = A[8];
  reinterpret_cast<float4*>(lBstore + lBstart +  0)[0] = B[0];
  reinterpret_cast<float4*>(lBstore + lBstart + 32)[0] = B[8];
  
  //Outer loop
  for(unsigned int block_k = 0; block_k <= K - 8; block_k += 8)
  {
    __syncthreads();
    float* lAread = lA + lAstart + 4*idx;
    float* lBread = lB + lBstart + 4*idy;
    //Inner loop
#pragma unroll
    for (unsigned int k = 0; k < 8; k += 2)
    {
      //Fetch A to registers
      rA[0][0] = reinterpret_cast<float4*>(lAread + k*64 + 0*64 +  0)[0];
      rA[0][1] = reinterpret_cast<float4*>(lAread + k*64 + 0*64 + 32)[0];
      rA[1][0] = reinterpret_cast<float4*>(lAread + k*64 + 1*64 +  0)[0];
      rA[1][1] = reinterpret_cast<float4*>(lAread + k*64 + 1*64 + 32)[0];
      
      //Fetch B to registers
      rB[0][0] = reinterpret_cast<float4*>(lBread + k*64 + 0*64 +  0)[0];
      rB[0][1] = reinterpret_cast<float4*>(lBread + k*64 + 0*64 + 32)[0];
      rB[1][0] = reinterpret_cast<float4*>(lBread + k*64 + 1*64 +  0)[0];
      rB[1][1] = reinterpret_cast<float4*>(lBread + k*64 + 1*64 + 32)[0];

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
    lBstart ^= 512;
    A += 8 * LDA;
    B += 8 * LDB;
    //Fetch A / B to local memory
    reinterpret_cast<float4*>(lAstore + lAstart +  0)[0] = A[0];
    reinterpret_cast<float4*>(lAstore + lAstart + 32)[0] = A[8];
    reinterpret_cast<float4*>(lBstore + lBstart +  0)[0] = B[0];
    reinterpret_cast<float4*>(lBstore + lBstart + 32)[0] = B[8];
  }
  
  //Write back C
  C += (gidx*64 + idx*4) + (gidy*64 + idy*4)*LDC;
  C[ 0 +  0*LDC] = rC[ 0] * alpha;
  C[ 0 +  1*LDC] = rC[ 1] * alpha;
  C[ 0 +  2*LDC] = rC[ 2] * alpha;
  C[ 0 +  3*LDC] = rC[ 3] * alpha;
  C[ 0 + 32*LDC] = rC[ 4] * alpha;
  C[ 0 + 33*LDC] = rC[ 5] * alpha;
  C[ 0 + 34*LDC] = rC[ 6] * alpha;
  C[ 0 + 35*LDC] = rC[ 7] * alpha;
  C[ 1 +  0*LDC] = rC[ 8] * alpha;
  C[ 1 +  1*LDC] = rC[ 9] * alpha;
  C[ 1 +  2*LDC] = rC[10] * alpha;
  C[ 1 +  3*LDC] = rC[11] * alpha;
  C[ 1 + 32*LDC] = rC[12] * alpha;
  C[ 1 + 33*LDC] = rC[13] * alpha;
  C[ 1 + 34*LDC] = rC[14] * alpha;
  C[ 1 + 35*LDC] = rC[15] * alpha;
  C[ 2 +  0*LDC] = rC[16] * alpha;
  C[ 2 +  1*LDC] = rC[17] * alpha;
  C[ 2 +  2*LDC] = rC[18] * alpha;
  C[ 2 +  3*LDC] = rC[19] * alpha;
  C[ 2 + 32*LDC] = rC[20] * alpha;
  C[ 2 + 33*LDC] = rC[21] * alpha;
  C[ 2 + 34*LDC] = rC[22] * alpha;
  C[ 2 + 35*LDC] = rC[23] * alpha;
  C[ 3 +  0*LDC] = rC[24] * alpha;
  C[ 3 +  1*LDC] = rC[25] * alpha;
  C[ 3 +  2*LDC] = rC[26] * alpha;
  C[ 3 +  3*LDC] = rC[27] * alpha;
  C[ 3 + 32*LDC] = rC[28] * alpha;
  C[ 3 + 33*LDC] = rC[29] * alpha;
  C[ 3 + 34*LDC] = rC[30] * alpha;
  C[ 3 + 35*LDC] = rC[31] * alpha;
  C[32 +  0*LDC] = rC[32] * alpha;
  C[32 +  1*LDC] = rC[33] * alpha;
  C[32 +  2*LDC] = rC[34] * alpha;
  C[32 +  3*LDC] = rC[35] * alpha;
  C[32 + 32*LDC] = rC[36] * alpha;
  C[32 + 33*LDC] = rC[37] * alpha;
  C[32 + 34*LDC] = rC[38] * alpha;
  C[32 + 35*LDC] = rC[39] * alpha;
  C[33 +  0*LDC] = rC[40] * alpha;
  C[33 +  1*LDC] = rC[41] * alpha;
  C[33 +  2*LDC] = rC[42] * alpha;
  C[33 +  3*LDC] = rC[43] * alpha;
  C[33 + 32*LDC] = rC[44] * alpha;
  C[33 + 33*LDC] = rC[45] * alpha;
  C[33 + 34*LDC] = rC[46] * alpha;
  C[33 + 35*LDC] = rC[47] * alpha;
  C[34 +  0*LDC] = rC[48] * alpha;
  C[34 +  1*LDC] = rC[49] * alpha;
  C[34 +  2*LDC] = rC[50] * alpha;
  C[34 +  3*LDC] = rC[51] * alpha;
  C[34 + 32*LDC] = rC[52] * alpha;
  C[34 + 33*LDC] = rC[53] * alpha;
  C[34 + 34*LDC] = rC[54] * alpha;
  C[34 + 35*LDC] = rC[55] * alpha;
  C[35 +  0*LDC] = rC[56] * alpha;
  C[35 +  1*LDC] = rC[57] * alpha;
  C[35 +  2*LDC] = rC[58] * alpha;
  C[35 +  3*LDC] = rC[59] * alpha;
  C[35 + 32*LDC] = rC[60] * alpha;
  C[35 + 33*LDC] = rC[61] * alpha;
  C[35 + 34*LDC] = rC[62] * alpha;
  C[35 + 35*LDC] = rC[63] * alpha;

}

typedef texture<float4, cudaTextureType1D, cudaReadModeElementType> FloatTex;

FloatTex texA(0, cudaFilterModePoint, cudaAddressModeBorder);
FloatTex texB(0, cudaFilterModePoint, cudaAddressModeBorder);

extern "C"
__global__ void __launch_bounds__(64) gemm64_1(
    int M, int N, int K,
    int LDA, int LDB,
    float* C, int LDC,
    float alpha, float beta) {
  // 2x512 float for strip A/B, double buffering
  __shared__ float4 shareA[256];
  __shared__ float4 shareB[256];
  // registers
  float rC[64] = {0};
  //float rC[64];
  float4 rA[2][2];
  float4 rB[2][2];
  float4 loadX0, loadX2, loadX4, loadX6;
  
  int tid = threadIdx.x;
  int bx = blockIdx.x;
  int by = blockIdx.y;
  
  int blk, ldx, ldx4, ldx8;
  FloatTex tex = (tid > 31) ? texB : texA;
  float4* share = (tid > 31) ? shareB : shareA;
  if (tid > 31) {
    blk = by;
    ldx4 = LDB;
  } else {
    blk = bx;
    ldx4 = LDA;
  }
  
  // store the zeros in the share buffer
  // share[128].x = share[128].y = share[128].z = share[128].w = 0.;
  
  int tid2 = (tid >> 4) & 1;
  int tid15 = tid & 15;
  //int tid31 = tid & 31;
  //int tid32 = tid & 32;
  ldx = ldx4 >> 2;
  ldx8 = ldx4 + ldx4;

  // track0 is location to read from texture (texA/texB) [tid2, 64*blk+4*tid15]
  // track0 = (64 * blk + 4 * tid15 + ldx4 * tid2) / 4 (divide 4 due to float4)
  int track0 = (blk << 4) + tid15 + (ldx * tid2);
  int track2 = track0 + ldx + ldx;
  int track4 = track0 + ldx4;
  int track6 = track2 + ldx4;
  
  // end is the boundary of track0
  int end = track0 + (K - 8) * ldx;

  // writeS is location to write into shared memory (shareA/shareB) [tid2, 4*tid15]
  // writeS = (4 * tid15 + 64 * tid2) / 4
  int writeS = tid15 + tid2 * 16;

  // readAs/readBs is location to read from shared memory (shareA/shareB) for multiply
  int readAs = (tid >> 1) & 7;
  int readBs = ((tid & 0x30) >> 3) | (tid & 1);

  // load texture (texA/texB) to registers loadX0 - loadX6
  loadX0 = tex1Dfetch(tex, track0);
  loadX2 = tex1Dfetch(tex, track2);
  loadX4 = tex1Dfetch(tex, track4);
  loadX6 = tex1Dfetch(tex, track6);

  
  // init rC to 0
// #pragma unroll
//   for (int i = 0; i < 16; ++i) {
//     rC[i*4 + 0] = share[128].x;
//     rC[i*4 + 1] = share[128].y;
//     rC[i*4 + 2] = share[128].z;
//     rC[i*4 + 3] = share[128].w;
//     }

  // store loadX0 - loadX6 to shared memory
  share[writeS + 0*16] = loadX0;
  share[writeS + 2*16] = loadX2;
  share[writeS + 4*16] = loadX4;
  share[writeS + 6*16] = loadX6;

  track0 += ldx8;
  track2 += ldx8;
  track4 += ldx8;
  track6 += ldx8;

  __syncthreads();

  writeS ^= 128;
  
  rA[0][0] = shareA[readAs + 0*16 + 0];
  rB[0][0] = shareB[readBs + 0*16 + 0];
  rA[0][1] = shareA[readAs + 0*16 + 8];
  rB[0][1] = shareB[readBs + 0*16 + 8];
  
  //while (track0 < end) {
  while (track0 <= end)
  //for (int block_k = 0; block_k <= K - 8; block_k += 8)
  {
    // inner loop
    // auto generated code
    // Iter k = 0
    rC[ 0] = fma(rA[0][0].x, rB[0][0].x, rC[0]);
    rA[1][0] = shareA[readAs + 1*16 + 0]; // load smem to regs
    rC[ 1] = fma(rA[0][0].y, rB[0][0].x, rC[1]);
    rC[ 2] = fma(rA[0][0].z, rB[0][0].x, rC[2]);
    rB[1][0] = shareB[readBs + 1*16 + 0]; // load smem to regs
    rC[ 3] = fma(rA[0][0].w, rB[0][0].x, rC[3]);
    rC[ 4] = fma(rA[0][1].x, rB[0][0].x, rC[4]);
    rA[1][1] = shareA[readAs + 1*16 + 8]; // load smem to regs
    rC[ 5] = fma(rA[0][1].y, rB[0][0].x, rC[5]);
    rC[ 6] = fma(rA[0][1].z, rB[0][0].x, rC[6]);
    rB[1][1] = shareB[readBs + 1*16 + 8]; // load smem to regs
    rC[ 7] = fma(rA[0][1].w, rB[0][0].x, rC[7]);
    rC[ 8] = fma(rA[0][0].x, rB[0][0].y, rC[8]);
    rC[ 9] = fma(rA[0][0].y, rB[0][0].y, rC[9]);
    rC[10] = fma(rA[0][0].z, rB[0][0].y, rC[10]);
    rC[11] = fma(rA[0][0].w, rB[0][0].y, rC[11]);
    rC[12] = fma(rA[0][1].x, rB[0][0].y, rC[12]);
    rC[13] = fma(rA[0][1].y, rB[0][0].y, rC[13]);
    rC[14] = fma(rA[0][1].z, rB[0][0].y, rC[14]);
    rC[15] = fma(rA[0][1].w, rB[0][0].y, rC[15]);
    rC[16] = fma(rA[0][0].x, rB[0][0].z, rC[16]);
    rC[17] = fma(rA[0][0].y, rB[0][0].z, rC[17]);
    rC[18] = fma(rA[0][0].z, rB[0][0].z, rC[18]);
    rC[19] = fma(rA[0][0].w, rB[0][0].z, rC[19]);
    rC[20] = fma(rA[0][1].x, rB[0][0].z, rC[20]);
    rC[21] = fma(rA[0][1].y, rB[0][0].z, rC[21]);
    rC[22] = fma(rA[0][1].z, rB[0][0].z, rC[22]);
    rC[23] = fma(rA[0][1].w, rB[0][0].z, rC[23]);
    rC[24] = fma(rA[0][0].x, rB[0][0].w, rC[24]);
    rC[25] = fma(rA[0][0].y, rB[0][0].w, rC[25]);
    rC[26] = fma(rA[0][0].z, rB[0][0].w, rC[26]);
    rC[27] = fma(rA[0][0].w, rB[0][0].w, rC[27]);
    rC[28] = fma(rA[0][1].x, rB[0][0].w, rC[28]);
    rC[29] = fma(rA[0][1].y, rB[0][0].w, rC[29]);
    rC[30] = fma(rA[0][1].z, rB[0][0].w, rC[30]);
    rC[31] = fma(rA[0][1].w, rB[0][0].w, rC[31]);
    loadX0 = tex1Dfetch(tex, track0); // load next strip to register
    rC[32] = fma(rA[0][0].x, rB[0][1].x, rC[32]);
    rC[33] = fma(rA[0][0].y, rB[0][1].x, rC[33]);
    loadX2 = tex1Dfetch(tex, track2); // load next strip to register
    rC[34] = fma(rA[0][0].z, rB[0][1].x, rC[34]);
    rC[35] = fma(rA[0][0].w, rB[0][1].x, rC[35]);
    rC[36] = fma(rA[0][1].x, rB[0][1].x, rC[36]);
    rC[37] = fma(rA[0][1].y, rB[0][1].x, rC[37]);
    rC[38] = fma(rA[0][1].z, rB[0][1].x, rC[38]);
    rC[39] = fma(rA[0][1].w, rB[0][1].x, rC[39]);
    rC[40] = fma(rA[0][0].x, rB[0][1].y, rC[40]);
    rC[41] = fma(rA[0][0].y, rB[0][1].y, rC[41]);
    rC[42] = fma(rA[0][0].z, rB[0][1].y, rC[42]);
    rC[43] = fma(rA[0][0].w, rB[0][1].y, rC[43]);
    rC[44] = fma(rA[0][1].x, rB[0][1].y, rC[44]);
    rC[45] = fma(rA[0][1].y, rB[0][1].y, rC[45]);
    rC[46] = fma(rA[0][1].z, rB[0][1].y, rC[46]);
    rC[47] = fma(rA[0][1].w, rB[0][1].y, rC[47]);
    rC[48] = fma(rA[0][0].x, rB[0][1].z, rC[48]);
    rC[49] = fma(rA[0][0].y, rB[0][1].z, rC[49]);
    rC[50] = fma(rA[0][0].z, rB[0][1].z, rC[50]);
    rC[51] = fma(rA[0][0].w, rB[0][1].z, rC[51]);
    rC[52] = fma(rA[0][1].x, rB[0][1].z, rC[52]);
    rC[53] = fma(rA[0][1].y, rB[0][1].z, rC[53]);
    rC[54] = fma(rA[0][1].z, rB[0][1].z, rC[54]);
    rC[55] = fma(rA[0][1].w, rB[0][1].z, rC[55]);
    rC[56] = fma(rA[0][0].x, rB[0][1].w, rC[56]);
    rC[57] = fma(rA[0][0].y, rB[0][1].w, rC[57]);
    rC[58] = fma(rA[0][0].z, rB[0][1].w, rC[58]);
    rC[59] = fma(rA[0][0].w, rB[0][1].w, rC[59]);
    rC[60] = fma(rA[0][1].x, rB[0][1].w, rC[60]);
    rC[61] = fma(rA[0][1].y, rB[0][1].w, rC[61]);
    rC[62] = fma(rA[0][1].z, rB[0][1].w, rC[62]);
    rC[63] = fma(rA[0][1].w, rB[0][1].w, rC[63]);
    // Iter k = 1
    rC[ 0] = fma(rA[1][0].x, rB[1][0].x, rC[0]);
    rA[0][0] = shareA[readAs + 2*16 + 0]; // load smem to regs
    rC[ 1] = fma(rA[1][0].y, rB[1][0].x, rC[1]);
    rC[ 2] = fma(rA[1][0].z, rB[1][0].x, rC[2]);
    rB[0][0] = shareB[readBs + 2*16 + 0]; // load smem to regs
    rC[ 3] = fma(rA[1][0].w, rB[1][0].x, rC[3]);
    rC[ 4] = fma(rA[1][1].x, rB[1][0].x, rC[4]);
    rA[0][1] = shareA[readAs + 2*16 + 8]; // load smem to regs
    rC[ 5] = fma(rA[1][1].y, rB[1][0].x, rC[5]);
    rC[ 6] = fma(rA[1][1].z, rB[1][0].x, rC[6]);
    rB[0][1] = shareB[readBs + 2*16 + 8]; // load smem to regs
    rC[ 7] = fma(rA[1][1].w, rB[1][0].x, rC[7]);
    rC[ 8] = fma(rA[1][0].x, rB[1][0].y, rC[8]);
    rC[ 9] = fma(rA[1][0].y, rB[1][0].y, rC[9]);
    rC[10] = fma(rA[1][0].z, rB[1][0].y, rC[10]);
    rC[11] = fma(rA[1][0].w, rB[1][0].y, rC[11]);
    rC[12] = fma(rA[1][1].x, rB[1][0].y, rC[12]);
    rC[13] = fma(rA[1][1].y, rB[1][0].y, rC[13]);
    rC[14] = fma(rA[1][1].z, rB[1][0].y, rC[14]);
    rC[15] = fma(rA[1][1].w, rB[1][0].y, rC[15]);
    rC[16] = fma(rA[1][0].x, rB[1][0].z, rC[16]);
    rC[17] = fma(rA[1][0].y, rB[1][0].z, rC[17]);
    rC[18] = fma(rA[1][0].z, rB[1][0].z, rC[18]);
    rC[19] = fma(rA[1][0].w, rB[1][0].z, rC[19]);
    rC[20] = fma(rA[1][1].x, rB[1][0].z, rC[20]);
    rC[21] = fma(rA[1][1].y, rB[1][0].z, rC[21]);
    rC[22] = fma(rA[1][1].z, rB[1][0].z, rC[22]);
    rC[23] = fma(rA[1][1].w, rB[1][0].z, rC[23]);
    rC[24] = fma(rA[1][0].x, rB[1][0].w, rC[24]);
    rC[25] = fma(rA[1][0].y, rB[1][0].w, rC[25]);
    rC[26] = fma(rA[1][0].z, rB[1][0].w, rC[26]);
    rC[27] = fma(rA[1][0].w, rB[1][0].w, rC[27]);
    rC[28] = fma(rA[1][1].x, rB[1][0].w, rC[28]);
    rC[29] = fma(rA[1][1].y, rB[1][0].w, rC[29]);
    rC[30] = fma(rA[1][1].z, rB[1][0].w, rC[30]);
    rC[31] = fma(rA[1][1].w, rB[1][0].w, rC[31]);
    loadX4 = tex1Dfetch(tex, track4); // load next strip to register
    rC[32] = fma(rA[1][0].x, rB[1][1].x, rC[32]);
    rC[33] = fma(rA[1][0].y, rB[1][1].x, rC[33]);
    loadX6 = tex1Dfetch(tex, track6); // load next strip to register
    rC[34] = fma(rA[1][0].z, rB[1][1].x, rC[34]);
    rC[35] = fma(rA[1][0].w, rB[1][1].x, rC[35]);
    rC[36] = fma(rA[1][1].x, rB[1][1].x, rC[36]);
    rC[37] = fma(rA[1][1].y, rB[1][1].x, rC[37]);
    rC[38] = fma(rA[1][1].z, rB[1][1].x, rC[38]);
    rC[39] = fma(rA[1][1].w, rB[1][1].x, rC[39]);
    rC[40] = fma(rA[1][0].x, rB[1][1].y, rC[40]);
    rC[41] = fma(rA[1][0].y, rB[1][1].y, rC[41]);
    rC[42] = fma(rA[1][0].z, rB[1][1].y, rC[42]);
    rC[43] = fma(rA[1][0].w, rB[1][1].y, rC[43]);
    rC[44] = fma(rA[1][1].x, rB[1][1].y, rC[44]);
    rC[45] = fma(rA[1][1].y, rB[1][1].y, rC[45]);
    rC[46] = fma(rA[1][1].z, rB[1][1].y, rC[46]);
    rC[47] = fma(rA[1][1].w, rB[1][1].y, rC[47]);
    rC[48] = fma(rA[1][0].x, rB[1][1].z, rC[48]);
    rC[49] = fma(rA[1][0].y, rB[1][1].z, rC[49]);
    rC[50] = fma(rA[1][0].z, rB[1][1].z, rC[50]);
    rC[51] = fma(rA[1][0].w, rB[1][1].z, rC[51]);
    rC[52] = fma(rA[1][1].x, rB[1][1].z, rC[52]);
    rC[53] = fma(rA[1][1].y, rB[1][1].z, rC[53]);
    rC[54] = fma(rA[1][1].z, rB[1][1].z, rC[54]);
    rC[55] = fma(rA[1][1].w, rB[1][1].z, rC[55]);
    rC[56] = fma(rA[1][0].x, rB[1][1].w, rC[56]);
    rC[57] = fma(rA[1][0].y, rB[1][1].w, rC[57]);
    rC[58] = fma(rA[1][0].z, rB[1][1].w, rC[58]);
    rC[59] = fma(rA[1][0].w, rB[1][1].w, rC[59]);
    rC[60] = fma(rA[1][1].x, rB[1][1].w, rC[60]);
    rC[61] = fma(rA[1][1].y, rB[1][1].w, rC[61]);
    rC[62] = fma(rA[1][1].z, rB[1][1].w, rC[62]);
    rC[63] = fma(rA[1][1].w, rB[1][1].w, rC[63]);
    // Iter k = 2
    rC[ 0] = fma(rA[0][0].x, rB[0][0].x, rC[0]);
    rA[1][0] = shareA[readAs + 3*16 + 0]; // load smem to regs
    rC[ 1] = fma(rA[0][0].y, rB[0][0].x, rC[1]);
    rC[ 2] = fma(rA[0][0].z, rB[0][0].x, rC[2]);
    rB[1][0] = shareB[readBs + 3*16 + 0]; // load smem to regs
    rC[ 3] = fma(rA[0][0].w, rB[0][0].x, rC[3]);
    rC[ 4] = fma(rA[0][1].x, rB[0][0].x, rC[4]);
    rA[1][1] = shareA[readAs + 3*16 + 8]; // load smem to regs
    rC[ 5] = fma(rA[0][1].y, rB[0][0].x, rC[5]);
    rC[ 6] = fma(rA[0][1].z, rB[0][0].x, rC[6]);
    rB[1][1] = shareB[readBs + 3*16 + 8]; // load smem to regs
    rC[ 7] = fma(rA[0][1].w, rB[0][0].x, rC[7]);
    rC[ 8] = fma(rA[0][0].x, rB[0][0].y, rC[8]);
    rC[ 9] = fma(rA[0][0].y, rB[0][0].y, rC[9]);
    rC[10] = fma(rA[0][0].z, rB[0][0].y, rC[10]);
    rC[11] = fma(rA[0][0].w, rB[0][0].y, rC[11]);
    rC[12] = fma(rA[0][1].x, rB[0][0].y, rC[12]);
    rC[13] = fma(rA[0][1].y, rB[0][0].y, rC[13]);
    rC[14] = fma(rA[0][1].z, rB[0][0].y, rC[14]);
    rC[15] = fma(rA[0][1].w, rB[0][0].y, rC[15]);
    rC[16] = fma(rA[0][0].x, rB[0][0].z, rC[16]);
    rC[17] = fma(rA[0][0].y, rB[0][0].z, rC[17]);
    rC[18] = fma(rA[0][0].z, rB[0][0].z, rC[18]);
    rC[19] = fma(rA[0][0].w, rB[0][0].z, rC[19]);
    rC[20] = fma(rA[0][1].x, rB[0][0].z, rC[20]);
    rC[21] = fma(rA[0][1].y, rB[0][0].z, rC[21]);
    rC[22] = fma(rA[0][1].z, rB[0][0].z, rC[22]);
    rC[23] = fma(rA[0][1].w, rB[0][0].z, rC[23]);
    rC[24] = fma(rA[0][0].x, rB[0][0].w, rC[24]);
    rC[25] = fma(rA[0][0].y, rB[0][0].w, rC[25]);
    rC[26] = fma(rA[0][0].z, rB[0][0].w, rC[26]);
    rC[27] = fma(rA[0][0].w, rB[0][0].w, rC[27]);
    rC[28] = fma(rA[0][1].x, rB[0][0].w, rC[28]);
    rC[29] = fma(rA[0][1].y, rB[0][0].w, rC[29]);
    rC[30] = fma(rA[0][1].z, rB[0][0].w, rC[30]);
    rC[31] = fma(rA[0][1].w, rB[0][0].w, rC[31]);
    rC[32] = fma(rA[0][0].x, rB[0][1].x, rC[32]);
    rC[33] = fma(rA[0][0].y, rB[0][1].x, rC[33]);
    rC[34] = fma(rA[0][0].z, rB[0][1].x, rC[34]);
    rC[35] = fma(rA[0][0].w, rB[0][1].x, rC[35]);
    rC[36] = fma(rA[0][1].x, rB[0][1].x, rC[36]);
    rC[37] = fma(rA[0][1].y, rB[0][1].x, rC[37]);
    rC[38] = fma(rA[0][1].z, rB[0][1].x, rC[38]);
    rC[39] = fma(rA[0][1].w, rB[0][1].x, rC[39]);
    rC[40] = fma(rA[0][0].x, rB[0][1].y, rC[40]);
    rC[41] = fma(rA[0][0].y, rB[0][1].y, rC[41]);
    rC[42] = fma(rA[0][0].z, rB[0][1].y, rC[42]);
    rC[43] = fma(rA[0][0].w, rB[0][1].y, rC[43]);
    rC[44] = fma(rA[0][1].x, rB[0][1].y, rC[44]);
    rC[45] = fma(rA[0][1].y, rB[0][1].y, rC[45]);
    rC[46] = fma(rA[0][1].z, rB[0][1].y, rC[46]);
    rC[47] = fma(rA[0][1].w, rB[0][1].y, rC[47]);
    rC[48] = fma(rA[0][0].x, rB[0][1].z, rC[48]);
    rC[49] = fma(rA[0][0].y, rB[0][1].z, rC[49]);
    rC[50] = fma(rA[0][0].z, rB[0][1].z, rC[50]);
    rC[51] = fma(rA[0][0].w, rB[0][1].z, rC[51]);
    rC[52] = fma(rA[0][1].x, rB[0][1].z, rC[52]);
    rC[53] = fma(rA[0][1].y, rB[0][1].z, rC[53]);
    rC[54] = fma(rA[0][1].z, rB[0][1].z, rC[54]);
    rC[55] = fma(rA[0][1].w, rB[0][1].z, rC[55]);
    rC[56] = fma(rA[0][0].x, rB[0][1].w, rC[56]);
    rC[57] = fma(rA[0][0].y, rB[0][1].w, rC[57]);
    rC[58] = fma(rA[0][0].z, rB[0][1].w, rC[58]);
    rC[59] = fma(rA[0][0].w, rB[0][1].w, rC[59]);
    rC[60] = fma(rA[0][1].x, rB[0][1].w, rC[60]);
    rC[61] = fma(rA[0][1].y, rB[0][1].w, rC[61]);
    rC[62] = fma(rA[0][1].z, rB[0][1].w, rC[62]);
    rC[63] = fma(rA[0][1].w, rB[0][1].w, rC[63]);
    // Iter k = 3
    rC[ 0] = fma(rA[1][0].x, rB[1][0].x, rC[0]);
    rA[0][0] = shareA[readAs + 4*16 + 0]; // load smem to regs
    rC[ 1] = fma(rA[1][0].y, rB[1][0].x, rC[1]);
    rC[ 2] = fma(rA[1][0].z, rB[1][0].x, rC[2]);
    rB[0][0] = shareB[readBs + 4*16 + 0]; // load smem to regs
    rC[ 3] = fma(rA[1][0].w, rB[1][0].x, rC[3]);
    rC[ 4] = fma(rA[1][1].x, rB[1][0].x, rC[4]);
    rA[0][1] = shareA[readAs + 4*16 + 8]; // load smem to regs
    rC[ 5] = fma(rA[1][1].y, rB[1][0].x, rC[5]);
    rC[ 6] = fma(rA[1][1].z, rB[1][0].x, rC[6]);
    rB[0][1] = shareB[readBs + 4*16 + 8]; // load smem to regs
    rC[ 7] = fma(rA[1][1].w, rB[1][0].x, rC[7]);
    rC[ 8] = fma(rA[1][0].x, rB[1][0].y, rC[8]);
    rC[ 9] = fma(rA[1][0].y, rB[1][0].y, rC[9]);
    rC[10] = fma(rA[1][0].z, rB[1][0].y, rC[10]);
    rC[11] = fma(rA[1][0].w, rB[1][0].y, rC[11]);
    rC[12] = fma(rA[1][1].x, rB[1][0].y, rC[12]);
    rC[13] = fma(rA[1][1].y, rB[1][0].y, rC[13]);
    rC[14] = fma(rA[1][1].z, rB[1][0].y, rC[14]);
    rC[15] = fma(rA[1][1].w, rB[1][0].y, rC[15]);
    rC[16] = fma(rA[1][0].x, rB[1][0].z, rC[16]);
    rC[17] = fma(rA[1][0].y, rB[1][0].z, rC[17]);
    rC[18] = fma(rA[1][0].z, rB[1][0].z, rC[18]);
    rC[19] = fma(rA[1][0].w, rB[1][0].z, rC[19]);
    rC[20] = fma(rA[1][1].x, rB[1][0].z, rC[20]);
    rC[21] = fma(rA[1][1].y, rB[1][0].z, rC[21]);
    rC[22] = fma(rA[1][1].z, rB[1][0].z, rC[22]);
    rC[23] = fma(rA[1][1].w, rB[1][0].z, rC[23]);
    rC[24] = fma(rA[1][0].x, rB[1][0].w, rC[24]);
    rC[25] = fma(rA[1][0].y, rB[1][0].w, rC[25]);
    rC[26] = fma(rA[1][0].z, rB[1][0].w, rC[26]);
    rC[27] = fma(rA[1][0].w, rB[1][0].w, rC[27]);
    rC[28] = fma(rA[1][1].x, rB[1][0].w, rC[28]);
    rC[29] = fma(rA[1][1].y, rB[1][0].w, rC[29]);
    rC[30] = fma(rA[1][1].z, rB[1][0].w, rC[30]);
    rC[31] = fma(rA[1][1].w, rB[1][0].w, rC[31]);
    rC[32] = fma(rA[1][0].x, rB[1][1].x, rC[32]);
    rC[33] = fma(rA[1][0].y, rB[1][1].x, rC[33]);
    rC[34] = fma(rA[1][0].z, rB[1][1].x, rC[34]);
    rC[35] = fma(rA[1][0].w, rB[1][1].x, rC[35]);
    rC[36] = fma(rA[1][1].x, rB[1][1].x, rC[36]);
    rC[37] = fma(rA[1][1].y, rB[1][1].x, rC[37]);
    rC[38] = fma(rA[1][1].z, rB[1][1].x, rC[38]);
    rC[39] = fma(rA[1][1].w, rB[1][1].x, rC[39]);
    rC[40] = fma(rA[1][0].x, rB[1][1].y, rC[40]);
    rC[41] = fma(rA[1][0].y, rB[1][1].y, rC[41]);
    rC[42] = fma(rA[1][0].z, rB[1][1].y, rC[42]);
    rC[43] = fma(rA[1][0].w, rB[1][1].y, rC[43]);
    rC[44] = fma(rA[1][1].x, rB[1][1].y, rC[44]);
    rC[45] = fma(rA[1][1].y, rB[1][1].y, rC[45]);
    rC[46] = fma(rA[1][1].z, rB[1][1].y, rC[46]);
    rC[47] = fma(rA[1][1].w, rB[1][1].y, rC[47]);
    rC[48] = fma(rA[1][0].x, rB[1][1].z, rC[48]);
    rC[49] = fma(rA[1][0].y, rB[1][1].z, rC[49]);
    rC[50] = fma(rA[1][0].z, rB[1][1].z, rC[50]);
    rC[51] = fma(rA[1][0].w, rB[1][1].z, rC[51]);
    rC[52] = fma(rA[1][1].x, rB[1][1].z, rC[52]);
    rC[53] = fma(rA[1][1].y, rB[1][1].z, rC[53]);
    rC[54] = fma(rA[1][1].z, rB[1][1].z, rC[54]);
    rC[55] = fma(rA[1][1].w, rB[1][1].z, rC[55]);
    rC[56] = fma(rA[1][0].x, rB[1][1].w, rC[56]);
    rC[57] = fma(rA[1][0].y, rB[1][1].w, rC[57]);
    rC[58] = fma(rA[1][0].z, rB[1][1].w, rC[58]);
    rC[59] = fma(rA[1][0].w, rB[1][1].w, rC[59]);
    rC[60] = fma(rA[1][1].x, rB[1][1].w, rC[60]);
    rC[61] = fma(rA[1][1].y, rB[1][1].w, rC[61]);
    rC[62] = fma(rA[1][1].z, rB[1][1].w, rC[62]);
    rC[63] = fma(rA[1][1].w, rB[1][1].w, rC[63]);
    // Iter k = 4
    rC[ 0] = fma(rA[0][0].x, rB[0][0].x, rC[0]);
    rA[1][0] = shareA[readAs + 5*16 + 0]; // load smem to regs
    rC[ 1] = fma(rA[0][0].y, rB[0][0].x, rC[1]);
    rC[ 2] = fma(rA[0][0].z, rB[0][0].x, rC[2]);
    rB[1][0] = shareB[readBs + 5*16 + 0]; // load smem to regs
    rC[ 3] = fma(rA[0][0].w, rB[0][0].x, rC[3]);
    rC[ 4] = fma(rA[0][1].x, rB[0][0].x, rC[4]);
    rA[1][1] = shareA[readAs + 5*16 + 8]; // load smem to regs
    rC[ 5] = fma(rA[0][1].y, rB[0][0].x, rC[5]);
    rC[ 6] = fma(rA[0][1].z, rB[0][0].x, rC[6]);
    rB[1][1] = shareB[readBs + 5*16 + 8]; // load smem to regs
    rC[ 7] = fma(rA[0][1].w, rB[0][0].x, rC[7]);
    rC[ 8] = fma(rA[0][0].x, rB[0][0].y, rC[8]);
    rC[ 9] = fma(rA[0][0].y, rB[0][0].y, rC[9]);
    rC[10] = fma(rA[0][0].z, rB[0][0].y, rC[10]);
    rC[11] = fma(rA[0][0].w, rB[0][0].y, rC[11]);
    rC[12] = fma(rA[0][1].x, rB[0][0].y, rC[12]);
    rC[13] = fma(rA[0][1].y, rB[0][0].y, rC[13]);
    rC[14] = fma(rA[0][1].z, rB[0][0].y, rC[14]);
    rC[15] = fma(rA[0][1].w, rB[0][0].y, rC[15]);
    rC[16] = fma(rA[0][0].x, rB[0][0].z, rC[16]);
    rC[17] = fma(rA[0][0].y, rB[0][0].z, rC[17]);
    rC[18] = fma(rA[0][0].z, rB[0][0].z, rC[18]);
    rC[19] = fma(rA[0][0].w, rB[0][0].z, rC[19]);
    rC[20] = fma(rA[0][1].x, rB[0][0].z, rC[20]);
    rC[21] = fma(rA[0][1].y, rB[0][0].z, rC[21]);
    rC[22] = fma(rA[0][1].z, rB[0][0].z, rC[22]);
    rC[23] = fma(rA[0][1].w, rB[0][0].z, rC[23]);
    rC[24] = fma(rA[0][0].x, rB[0][0].w, rC[24]);
    rC[25] = fma(rA[0][0].y, rB[0][0].w, rC[25]);
    rC[26] = fma(rA[0][0].z, rB[0][0].w, rC[26]);
    rC[27] = fma(rA[0][0].w, rB[0][0].w, rC[27]);
    rC[28] = fma(rA[0][1].x, rB[0][0].w, rC[28]);
    rC[29] = fma(rA[0][1].y, rB[0][0].w, rC[29]);
    rC[30] = fma(rA[0][1].z, rB[0][0].w, rC[30]);
    rC[31] = fma(rA[0][1].w, rB[0][0].w, rC[31]);
    rC[32] = fma(rA[0][0].x, rB[0][1].x, rC[32]);
    rC[33] = fma(rA[0][0].y, rB[0][1].x, rC[33]);
    rC[34] = fma(rA[0][0].z, rB[0][1].x, rC[34]);
    rC[35] = fma(rA[0][0].w, rB[0][1].x, rC[35]);
    rC[36] = fma(rA[0][1].x, rB[0][1].x, rC[36]);
    rC[37] = fma(rA[0][1].y, rB[0][1].x, rC[37]);
    rC[38] = fma(rA[0][1].z, rB[0][1].x, rC[38]);
    rC[39] = fma(rA[0][1].w, rB[0][1].x, rC[39]);
    rC[40] = fma(rA[0][0].x, rB[0][1].y, rC[40]);
    rC[41] = fma(rA[0][0].y, rB[0][1].y, rC[41]);
    rC[42] = fma(rA[0][0].z, rB[0][1].y, rC[42]);
    rC[43] = fma(rA[0][0].w, rB[0][1].y, rC[43]);
    rC[44] = fma(rA[0][1].x, rB[0][1].y, rC[44]);
    rC[45] = fma(rA[0][1].y, rB[0][1].y, rC[45]);
    rC[46] = fma(rA[0][1].z, rB[0][1].y, rC[46]);
    rC[47] = fma(rA[0][1].w, rB[0][1].y, rC[47]);
    rC[48] = fma(rA[0][0].x, rB[0][1].z, rC[48]);
    rC[49] = fma(rA[0][0].y, rB[0][1].z, rC[49]);
    rC[50] = fma(rA[0][0].z, rB[0][1].z, rC[50]);
    rC[51] = fma(rA[0][0].w, rB[0][1].z, rC[51]);
    rC[52] = fma(rA[0][1].x, rB[0][1].z, rC[52]);
    rC[53] = fma(rA[0][1].y, rB[0][1].z, rC[53]);
    rC[54] = fma(rA[0][1].z, rB[0][1].z, rC[54]);
    rC[55] = fma(rA[0][1].w, rB[0][1].z, rC[55]);
    rC[56] = fma(rA[0][0].x, rB[0][1].w, rC[56]);
    rC[57] = fma(rA[0][0].y, rB[0][1].w, rC[57]);
    rC[58] = fma(rA[0][0].z, rB[0][1].w, rC[58]);
    rC[59] = fma(rA[0][0].w, rB[0][1].w, rC[59]);
    rC[60] = fma(rA[0][1].x, rB[0][1].w, rC[60]);
    rC[61] = fma(rA[0][1].y, rB[0][1].w, rC[61]);
    rC[62] = fma(rA[0][1].z, rB[0][1].w, rC[62]);
    rC[63] = fma(rA[0][1].w, rB[0][1].w, rC[63]);
    // Iter k = 5
    rC[ 0] = fma(rA[1][0].x, rB[1][0].x, rC[0]);
    rA[0][0] = shareA[readAs + 6*16 + 0]; // load smem to regs
    rC[ 1] = fma(rA[1][0].y, rB[1][0].x, rC[1]);
    rC[ 2] = fma(rA[1][0].z, rB[1][0].x, rC[2]);
    rB[0][0] = shareB[readBs + 6*16 + 0]; // load smem to regs
    rC[ 3] = fma(rA[1][0].w, rB[1][0].x, rC[3]);
    rC[ 4] = fma(rA[1][1].x, rB[1][0].x, rC[4]);
    rA[0][1] = shareA[readAs + 6*16 + 8]; // load smem to regs
    rC[ 5] = fma(rA[1][1].y, rB[1][0].x, rC[5]);
    rC[ 6] = fma(rA[1][1].z, rB[1][0].x, rC[6]);
    rB[0][1] = shareB[readBs + 6*16 + 8]; // load smem to regs
    rC[ 7] = fma(rA[1][1].w, rB[1][0].x, rC[7]);
    rC[ 8] = fma(rA[1][0].x, rB[1][0].y, rC[8]);
    rC[ 9] = fma(rA[1][0].y, rB[1][0].y, rC[9]);
    rC[10] = fma(rA[1][0].z, rB[1][0].y, rC[10]);
    rC[11] = fma(rA[1][0].w, rB[1][0].y, rC[11]);
    rC[12] = fma(rA[1][1].x, rB[1][0].y, rC[12]);
    rC[13] = fma(rA[1][1].y, rB[1][0].y, rC[13]);
    rC[14] = fma(rA[1][1].z, rB[1][0].y, rC[14]);
    rC[15] = fma(rA[1][1].w, rB[1][0].y, rC[15]);
    rC[16] = fma(rA[1][0].x, rB[1][0].z, rC[16]);
    rC[17] = fma(rA[1][0].y, rB[1][0].z, rC[17]);
    rC[18] = fma(rA[1][0].z, rB[1][0].z, rC[18]);
    rC[19] = fma(rA[1][0].w, rB[1][0].z, rC[19]);
    rC[20] = fma(rA[1][1].x, rB[1][0].z, rC[20]);
    rC[21] = fma(rA[1][1].y, rB[1][0].z, rC[21]);
    rC[22] = fma(rA[1][1].z, rB[1][0].z, rC[22]);
    rC[23] = fma(rA[1][1].w, rB[1][0].z, rC[23]);
    rC[24] = fma(rA[1][0].x, rB[1][0].w, rC[24]);
    rC[25] = fma(rA[1][0].y, rB[1][0].w, rC[25]);
    rC[26] = fma(rA[1][0].z, rB[1][0].w, rC[26]);
    rC[27] = fma(rA[1][0].w, rB[1][0].w, rC[27]);
    rC[28] = fma(rA[1][1].x, rB[1][0].w, rC[28]);
    rC[29] = fma(rA[1][1].y, rB[1][0].w, rC[29]);
    rC[30] = fma(rA[1][1].z, rB[1][0].w, rC[30]);
    share[writeS + 0*16] = loadX0; // store register to shared memory
    rC[31] = fma(rA[1][1].w, rB[1][0].w, rC[31]);
    rC[32] = fma(rA[1][0].x, rB[1][1].x, rC[32]);
    rC[33] = fma(rA[1][0].y, rB[1][1].x, rC[33]);
    rC[34] = fma(rA[1][0].z, rB[1][1].x, rC[34]);
    share[writeS + 2*16] = loadX2; // store register to shared memory
    rC[35] = fma(rA[1][0].w, rB[1][1].x, rC[35]);
    rC[36] = fma(rA[1][1].x, rB[1][1].x, rC[36]);
    rC[37] = fma(rA[1][1].y, rB[1][1].x, rC[37]);
    rC[38] = fma(rA[1][1].z, rB[1][1].x, rC[38]);
    rC[39] = fma(rA[1][1].w, rB[1][1].x, rC[39]);
    rC[40] = fma(rA[1][0].x, rB[1][1].y, rC[40]);
    rC[41] = fma(rA[1][0].y, rB[1][1].y, rC[41]);
    rC[42] = fma(rA[1][0].z, rB[1][1].y, rC[42]);
    rC[43] = fma(rA[1][0].w, rB[1][1].y, rC[43]);
    rC[44] = fma(rA[1][1].x, rB[1][1].y, rC[44]);
    rC[45] = fma(rA[1][1].y, rB[1][1].y, rC[45]);
    rC[46] = fma(rA[1][1].z, rB[1][1].y, rC[46]);
    rC[47] = fma(rA[1][1].w, rB[1][1].y, rC[47]);
    rC[48] = fma(rA[1][0].x, rB[1][1].z, rC[48]);
    rC[49] = fma(rA[1][0].y, rB[1][1].z, rC[49]);
    rC[50] = fma(rA[1][0].z, rB[1][1].z, rC[50]);
    rC[51] = fma(rA[1][0].w, rB[1][1].z, rC[51]);
    rC[52] = fma(rA[1][1].x, rB[1][1].z, rC[52]);
    rC[53] = fma(rA[1][1].y, rB[1][1].z, rC[53]);
    rC[54] = fma(rA[1][1].z, rB[1][1].z, rC[54]);
    rC[55] = fma(rA[1][1].w, rB[1][1].z, rC[55]);
    rC[56] = fma(rA[1][0].x, rB[1][1].w, rC[56]);
    rC[57] = fma(rA[1][0].y, rB[1][1].w, rC[57]);
    rC[58] = fma(rA[1][0].z, rB[1][1].w, rC[58]);
    rC[59] = fma(rA[1][0].w, rB[1][1].w, rC[59]);
    rC[60] = fma(rA[1][1].x, rB[1][1].w, rC[60]);
    rC[61] = fma(rA[1][1].y, rB[1][1].w, rC[61]);
    rC[62] = fma(rA[1][1].z, rB[1][1].w, rC[62]);
    rC[63] = fma(rA[1][1].w, rB[1][1].w, rC[63]);
    // Iter k = 6
    rC[ 0] = fma(rA[0][0].x, rB[0][0].x, rC[0]);
    rA[1][0] = shareA[readAs + 7*16 + 0]; // load smem to regs
    rC[ 1] = fma(rA[0][0].y, rB[0][0].x, rC[1]);
    rC[ 2] = fma(rA[0][0].z, rB[0][0].x, rC[2]);
    rB[1][0] = shareB[readBs + 7*16 + 0]; // load smem to regs
    rC[ 3] = fma(rA[0][0].w, rB[0][0].x, rC[3]);
    rC[ 4] = fma(rA[0][1].x, rB[0][0].x, rC[4]);
    rA[1][1] = shareA[readAs + 7*16 + 8]; // load smem to regs
    rC[ 5] = fma(rA[0][1].y, rB[0][0].x, rC[5]);
    rC[ 6] = fma(rA[0][1].z, rB[0][0].x, rC[6]);
    rB[1][1] = shareB[readBs + 7*16 + 8]; // load smem to regs
    rC[ 7] = fma(rA[0][1].w, rB[0][0].x, rC[7]);
    rC[ 8] = fma(rA[0][0].x, rB[0][0].y, rC[8]);
    rC[ 9] = fma(rA[0][0].y, rB[0][0].y, rC[9]);
    rC[10] = fma(rA[0][0].z, rB[0][0].y, rC[10]);
    rC[11] = fma(rA[0][0].w, rB[0][0].y, rC[11]);
    rC[12] = fma(rA[0][1].x, rB[0][0].y, rC[12]);
    rC[13] = fma(rA[0][1].y, rB[0][0].y, rC[13]);
    rC[14] = fma(rA[0][1].z, rB[0][0].y, rC[14]);
    rC[15] = fma(rA[0][1].w, rB[0][0].y, rC[15]);
    rC[16] = fma(rA[0][0].x, rB[0][0].z, rC[16]);
    rC[17] = fma(rA[0][0].y, rB[0][0].z, rC[17]);
    rC[18] = fma(rA[0][0].z, rB[0][0].z, rC[18]);
    rC[19] = fma(rA[0][0].w, rB[0][0].z, rC[19]);
    rC[20] = fma(rA[0][1].x, rB[0][0].z, rC[20]);
    rC[21] = fma(rA[0][1].y, rB[0][0].z, rC[21]);
    rC[22] = fma(rA[0][1].z, rB[0][0].z, rC[22]);
    rC[23] = fma(rA[0][1].w, rB[0][0].z, rC[23]);
    rC[24] = fma(rA[0][0].x, rB[0][0].w, rC[24]);
    rC[25] = fma(rA[0][0].y, rB[0][0].w, rC[25]);
    rC[26] = fma(rA[0][0].z, rB[0][0].w, rC[26]);
    rC[27] = fma(rA[0][0].w, rB[0][0].w, rC[27]);
    rC[28] = fma(rA[0][1].x, rB[0][0].w, rC[28]);
    rC[29] = fma(rA[0][1].y, rB[0][0].w, rC[29]);
    rC[30] = fma(rA[0][1].z, rB[0][0].w, rC[30]);
    share[writeS + 4*16] = loadX4; // store register to shared memory
    rC[31] = fma(rA[0][1].w, rB[0][0].w, rC[31]);
    rC[32] = fma(rA[0][0].x, rB[0][1].x, rC[32]);
    rC[33] = fma(rA[0][0].y, rB[0][1].x, rC[33]);
    rC[34] = fma(rA[0][0].z, rB[0][1].x, rC[34]);
    share[writeS + 6*16] = loadX6; // store register to shared memory
    rC[35] = fma(rA[0][0].w, rB[0][1].x, rC[35]);
    rC[36] = fma(rA[0][1].x, rB[0][1].x, rC[36]);
    rC[37] = fma(rA[0][1].y, rB[0][1].x, rC[37]);
    rC[38] = fma(rA[0][1].z, rB[0][1].x, rC[38]);
    rC[39] = fma(rA[0][1].w, rB[0][1].x, rC[39]);
    rC[40] = fma(rA[0][0].x, rB[0][1].y, rC[40]);
    rC[41] = fma(rA[0][0].y, rB[0][1].y, rC[41]);
    rC[42] = fma(rA[0][0].z, rB[0][1].y, rC[42]);
    rC[43] = fma(rA[0][0].w, rB[0][1].y, rC[43]);
    rC[44] = fma(rA[0][1].x, rB[0][1].y, rC[44]);
    rC[45] = fma(rA[0][1].y, rB[0][1].y, rC[45]);
    rC[46] = fma(rA[0][1].z, rB[0][1].y, rC[46]);
    rC[47] = fma(rA[0][1].w, rB[0][1].y, rC[47]);
    rC[48] = fma(rA[0][0].x, rB[0][1].z, rC[48]);
    rC[49] = fma(rA[0][0].y, rB[0][1].z, rC[49]);
    rC[50] = fma(rA[0][0].z, rB[0][1].z, rC[50]);
    rC[51] = fma(rA[0][0].w, rB[0][1].z, rC[51]);
    rC[52] = fma(rA[0][1].x, rB[0][1].z, rC[52]);
    rC[53] = fma(rA[0][1].y, rB[0][1].z, rC[53]);
    rC[54] = fma(rA[0][1].z, rB[0][1].z, rC[54]);
    rC[55] = fma(rA[0][1].w, rB[0][1].z, rC[55]);
    rC[56] = fma(rA[0][0].x, rB[0][1].w, rC[56]);
    rC[57] = fma(rA[0][0].y, rB[0][1].w, rC[57]);
    rC[58] = fma(rA[0][0].z, rB[0][1].w, rC[58]);
    rC[59] = fma(rA[0][0].w, rB[0][1].w, rC[59]);
    rC[60] = fma(rA[0][1].x, rB[0][1].w, rC[60]);
    rC[61] = fma(rA[0][1].y, rB[0][1].w, rC[61]);
    rC[62] = fma(rA[0][1].z, rB[0][1].w, rC[62]);
    __syncthreads(); // sync till next strip is stored in shared memory
    readAs ^= 128; // togger readAs to read next A strip
    readBs ^= 128; // togger readBs to read next B strip
    writeS ^= 128; // togger writeS to write to the other shared memory buffer
    rC[63] = fma(rA[0][1].w, rB[0][1].w, rC[63]);
    // Iter k = 7
    rC[ 0] = fma(rA[1][0].x, rB[1][0].x, rC[0]);
    rA[0][0] = shareA[readAs + 0*16 + 0]; // load smem to regs
    rC[ 1] = fma(rA[1][0].y, rB[1][0].x, rC[1]);
    rC[ 2] = fma(rA[1][0].z, rB[1][0].x, rC[2]);
    rB[0][0] = shareB[readBs + 0*16 + 0]; // load smem to regs
    rC[ 3] = fma(rA[1][0].w, rB[1][0].x, rC[3]);
    rC[ 4] = fma(rA[1][1].x, rB[1][0].x, rC[4]);
    rA[0][1] = shareA[readAs + 0*16 + 8]; // load smem to regs
    rC[ 5] = fma(rA[1][1].y, rB[1][0].x, rC[5]);
    rC[ 6] = fma(rA[1][1].z, rB[1][0].x, rC[6]);
    rB[0][1] = shareB[readBs + 0*16 + 8]; // load smem to regs
    rC[ 7] = fma(rA[1][1].w, rB[1][0].x, rC[7]);
    rC[ 8] = fma(rA[1][0].x, rB[1][0].y, rC[8]);
    rC[ 9] = fma(rA[1][0].y, rB[1][0].y, rC[9]);
    rC[10] = fma(rA[1][0].z, rB[1][0].y, rC[10]);
    rC[11] = fma(rA[1][0].w, rB[1][0].y, rC[11]);
    rC[12] = fma(rA[1][1].x, rB[1][0].y, rC[12]);
    rC[13] = fma(rA[1][1].y, rB[1][0].y, rC[13]);
    rC[14] = fma(rA[1][1].z, rB[1][0].y, rC[14]);
    rC[15] = fma(rA[1][1].w, rB[1][0].y, rC[15]);
    rC[16] = fma(rA[1][0].x, rB[1][0].z, rC[16]);
    rC[17] = fma(rA[1][0].y, rB[1][0].z, rC[17]);
    rC[18] = fma(rA[1][0].z, rB[1][0].z, rC[18]);
    rC[19] = fma(rA[1][0].w, rB[1][0].z, rC[19]);
    rC[20] = fma(rA[1][1].x, rB[1][0].z, rC[20]);
    rC[21] = fma(rA[1][1].y, rB[1][0].z, rC[21]);
    rC[22] = fma(rA[1][1].z, rB[1][0].z, rC[22]);
    rC[23] = fma(rA[1][1].w, rB[1][0].z, rC[23]);
    rC[24] = fma(rA[1][0].x, rB[1][0].w, rC[24]);
    rC[25] = fma(rA[1][0].y, rB[1][0].w, rC[25]);
    rC[26] = fma(rA[1][0].z, rB[1][0].w, rC[26]);
    rC[27] = fma(rA[1][0].w, rB[1][0].w, rC[27]);
    rC[28] = fma(rA[1][1].x, rB[1][0].w, rC[28]);
    rC[29] = fma(rA[1][1].y, rB[1][0].w, rC[29]);
    rC[30] = fma(rA[1][1].z, rB[1][0].w, rC[30]);
    rC[31] = fma(rA[1][1].w, rB[1][0].w, rC[31]);
    rC[32] = fma(rA[1][0].x, rB[1][1].x, rC[32]);
    rC[33] = fma(rA[1][0].y, rB[1][1].x, rC[33]);
    rC[34] = fma(rA[1][0].z, rB[1][1].x, rC[34]);
    rC[35] = fma(rA[1][0].w, rB[1][1].x, rC[35]);
    rC[36] = fma(rA[1][1].x, rB[1][1].x, rC[36]);
    rC[37] = fma(rA[1][1].y, rB[1][1].x, rC[37]);
    rC[38] = fma(rA[1][1].z, rB[1][1].x, rC[38]);
    rC[39] = fma(rA[1][1].w, rB[1][1].x, rC[39]);
    rC[40] = fma(rA[1][0].x, rB[1][1].y, rC[40]);
    rC[41] = fma(rA[1][0].y, rB[1][1].y, rC[41]);
    rC[42] = fma(rA[1][0].z, rB[1][1].y, rC[42]);
    rC[43] = fma(rA[1][0].w, rB[1][1].y, rC[43]);
    rC[44] = fma(rA[1][1].x, rB[1][1].y, rC[44]);
    rC[45] = fma(rA[1][1].y, rB[1][1].y, rC[45]);
    rC[46] = fma(rA[1][1].z, rB[1][1].y, rC[46]);
    rC[47] = fma(rA[1][1].w, rB[1][1].y, rC[47]);
    rC[48] = fma(rA[1][0].x, rB[1][1].z, rC[48]);
    rC[49] = fma(rA[1][0].y, rB[1][1].z, rC[49]);
    rC[50] = fma(rA[1][0].z, rB[1][1].z, rC[50]);
    rC[51] = fma(rA[1][0].w, rB[1][1].z, rC[51]);
    rC[52] = fma(rA[1][1].x, rB[1][1].z, rC[52]);
    rC[53] = fma(rA[1][1].y, rB[1][1].z, rC[53]);
    rC[54] = fma(rA[1][1].z, rB[1][1].z, rC[54]);
    rC[55] = fma(rA[1][1].w, rB[1][1].z, rC[55]);
    rC[56] = fma(rA[1][0].x, rB[1][1].w, rC[56]);
    rC[57] = fma(rA[1][0].y, rB[1][1].w, rC[57]);
    rC[58] = fma(rA[1][0].z, rB[1][1].w, rC[58]);
    rC[59] = fma(rA[1][0].w, rB[1][1].w, rC[59]);
    rC[60] = fma(rA[1][1].x, rB[1][1].w, rC[60]);
    rC[61] = fma(rA[1][1].y, rB[1][1].w, rC[61]);
    rC[62] = fma(rA[1][1].z, rB[1][1].w, rC[62]);
    rC[63] = fma(rA[1][1].w, rB[1][1].w, rC[63]);
    track0 += ldx8;
    track2 += ldx8;
    track4 += ldx8;
    track6 += ldx8;
    
  }
  
  // write back to C
  int cx = (readAs & 0x7f) * 4;
  int cy = (readBs & 0x7f) * 4;
  C += (bx*64 + cx) + (by*64 + cy) * LDC;

  // auto generated code
  C[ 0 +  0*LDC] = rC[ 0] * alpha;
  C[ 1 +  0*LDC] = rC[ 1] * alpha;
  C[ 2 +  0*LDC] = rC[ 2] * alpha;
  C[ 3 +  0*LDC] = rC[ 3] * alpha;
  C[32 +  0*LDC] = rC[ 4] * alpha;
  C[33 +  0*LDC] = rC[ 5] * alpha;
  C[34 +  0*LDC] = rC[ 6] * alpha;
  C[35 +  0*LDC] = rC[ 7] * alpha;
  C[ 0 +  1*LDC] = rC[ 8] * alpha;
  C[ 1 +  1*LDC] = rC[ 9] * alpha;
  C[ 2 +  1*LDC] = rC[10] * alpha;
  C[ 3 +  1*LDC] = rC[11] * alpha;
  C[32 +  1*LDC] = rC[12] * alpha;
  C[33 +  1*LDC] = rC[13] * alpha;
  C[34 +  1*LDC] = rC[14] * alpha;
  C[35 +  1*LDC] = rC[15] * alpha;
  C[ 0 +  2*LDC] = rC[16] * alpha;
  C[ 1 +  2*LDC] = rC[17] * alpha;
  C[ 2 +  2*LDC] = rC[18] * alpha;
  C[ 3 +  2*LDC] = rC[19] * alpha;
  C[32 +  2*LDC] = rC[20] * alpha;
  C[33 +  2*LDC] = rC[21] * alpha;
  C[34 +  2*LDC] = rC[22] * alpha;
  C[35 +  2*LDC] = rC[23] * alpha;
  C[ 0 +  3*LDC] = rC[24] * alpha;
  C[ 1 +  3*LDC] = rC[25] * alpha;
  C[ 2 +  3*LDC] = rC[26] * alpha;
  C[ 3 +  3*LDC] = rC[27] * alpha;
  C[32 +  3*LDC] = rC[28] * alpha;
  C[33 +  3*LDC] = rC[29] * alpha;
  C[34 +  3*LDC] = rC[30] * alpha;
  C[35 +  3*LDC] = rC[31] * alpha;
  C[ 0 + 32*LDC] = rC[32] * alpha;
  C[ 1 + 32*LDC] = rC[33] * alpha;
  C[ 2 + 32*LDC] = rC[34] * alpha;
  C[ 3 + 32*LDC] = rC[35] * alpha;
  C[32 + 32*LDC] = rC[36] * alpha;
  C[33 + 32*LDC] = rC[37] * alpha;
  C[34 + 32*LDC] = rC[38] * alpha;
  C[35 + 32*LDC] = rC[39] * alpha;
  C[ 0 + 33*LDC] = rC[40] * alpha;
  C[ 1 + 33*LDC] = rC[41] * alpha;
  C[ 2 + 33*LDC] = rC[42] * alpha;
  C[ 3 + 33*LDC] = rC[43] * alpha;
  C[32 + 33*LDC] = rC[44] * alpha;
  C[33 + 33*LDC] = rC[45] * alpha;
  C[34 + 33*LDC] = rC[46] * alpha;
  C[35 + 33*LDC] = rC[47] * alpha;
  C[ 0 + 34*LDC] = rC[48] * alpha;
  C[ 1 + 34*LDC] = rC[49] * alpha;
  C[ 2 + 34*LDC] = rC[50] * alpha;
  C[ 3 + 34*LDC] = rC[51] * alpha;
  C[32 + 34*LDC] = rC[52] * alpha;
  C[33 + 34*LDC] = rC[53] * alpha;
  C[34 + 34*LDC] = rC[54] * alpha;
  C[35 + 34*LDC] = rC[55] * alpha;
  C[ 0 + 35*LDC] = rC[56] * alpha;
  C[ 1 + 35*LDC] = rC[57] * alpha;
  C[ 2 + 35*LDC] = rC[58] * alpha;
  C[ 3 + 35*LDC] = rC[59] * alpha;
  C[32 + 35*LDC] = rC[60] * alpha;
  C[33 + 35*LDC] = rC[61] * alpha;
  C[34 + 35*LDC] = rC[62] * alpha;
  C[35 + 35*LDC] = rC[63] * alpha;
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

int main() {
  int device = 0;
  cudaSetDevice(device);
  srand(1);
  
  float *A, *B, *C, *hostC;
  float *devA, *devB, *devC;
  //int N = 64;
  int N = 128;
  //int N = 4096;
  size_t size = N * N * sizeof(float);

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
  CUDA_CHECK( cudaBindTexture(NULL, texA, devA, size) );
  CUDA_CHECK( cudaBindTexture(NULL, texB, devB, size) );

  cudaProfilerStart();

#define KERNEL 1
  
#if KERNEL == 0
  dim3 block_size, grid_size;
  block_size.x = 8;
  block_size.y = 8;
  grid_size.x = N / 64;
  grid_size.y = N / 64;
  gemm64_0<<<grid_size, block_size>>>(N, N, N, (float4*)devA, N, (float4*)devB, N, devC, N, 1.0, 0.0);
#elif KERNEL == 1
  dim3 block_size, grid_size;
  block_size.x = 64;
  grid_size.x = N / 64;
  grid_size.y = N / 64;
  gemm64_1<<<grid_size, block_size>>>(N, N, N, N, N, devC, N, 1.0, 0.0);
#elif KERNEL == 2
  dim3 block_size, grid_size;
  block_size.x = 8;
  block_size.y = 16;
  grid_size.x = N / 64;
  grid_size.y = N / 128;
  k0<<<grid_size, block_size>>>(N, N, N, devC, N, 1, (float4*)devA, N, (float4*)devB, N, 0);
#else
  return 1;
#endif

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