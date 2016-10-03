#include <cuda.h>

extern "C"
__global__ void __launch_bounds__(64) gemm64_1(
    int M, int N, int K,
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