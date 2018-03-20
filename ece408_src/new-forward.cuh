#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
#define MXNET_OPERATOR_NEW_FORWARD_CUH_
#define TILE_WIDTH 24
#include <mxnet/base.h>


namespace mxnet
{
namespace op
{




__global__ void forward_kernel(float *y, const float *x, const float *k, const int B, const int M, const int C, const int H, const int W, const int K) {

    /*
    Modify this function to implement the forward pass described in Chapter 16.
    We have added an additional dimension to the tensors to support an entire mini-batch
    The goal here is to be correct AND fast.
    We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    */

    const int H_out = H - K + 1;
    const int W_out = W - K + 1;
    #define y4d(i3,i2,i1,i0) y[(i3) * (M * H_out * W_out) + (i2)*(H_out * W_out) + (i1)*(W_out) + i0]
    #define x4d(i3,i2,i1,i0) x[(i3) * (C * H * W) + (i2)*(H * W) + (i1)*(W) + i0]
    #define k4d(i3,i2,i1,i0) k[(i3) * (C * K * K) + (i2)*(K * K) + (i1)*(K) + i0]
    #define x3d(i2, i1, i0) X_unroll[(i2) * (C * K * K * W_out * H_out)  + (i1) * (W_out * H_out) + i0]

  __shared__ float subTileA[TILE_WIDTH][TILE_WIDTH];
  __shared__ float subTileB[TILE_WIDTH][TILE_WIDTH];


  int bx = blockIdx.x;  int by = blockIdx.y;
  int tx = threadIdx.x; int ty = threadIdx.y;
  int batchidx=blockIdx.z;
  int numARows = M;
  int numAColumns = C*K*K;
  int numBRows = C*K*K;
  int numBColumns = H_out * W_out;
  float Pvalue = 0.0;

  int Row = by * TILE_WIDTH + ty;
  int Col = bx * TILE_WIDTH + tx;



  for (int m = 0; m < (int)(ceil(numAColumns*1.0/TILE_WIDTH)); ++m) {
      if(Row < numARows && m*TILE_WIDTH + tx < numAColumns && batchidx < B)
        subTileA[ty][tx] = k[Row*numAColumns + m*TILE_WIDTH + tx];//A[Row*numAColumns + m*TILE_WIDTH + tx];
      else
        subTileA[ty][tx] = 0;
      if(m*TILE_WIDTH+ty < numBRows && Col < numBColumns && batchidx < B)
        subTileB[ty][tx] = x[batchidx*C*K*K*W_out*H_out +(m*TILE_WIDTH+ty)*numBColumns + Col];
      else
        subTileB[ty][tx] = 0;

     __syncthreads();
  if(Row < numARows && Col < numBColumns && batchidx < B){
  for (int i = 0; i < TILE_WIDTH; ++i){

      Pvalue += subTileA[ty][i] * subTileB[i][tx];

      // if(Row == 0 && Col == 0 && batchidx == 0) {
      //   printf("Pvalue = %f\n", Pvalue);
      // }
  }
  y[batchidx*numARows*numBColumns+Row*numBColumns + Col] = Pvalue;
  }
      __syncthreads();
}
//   if(Row < numARows && Col < numBColumns && batchidx < B) {
//   //y4d(batchidx, Row, Col/W_out, Col%W_out) = Pvalue;
// 	y[batchidx*numARows*numBColumns+Row*numBColumns + Col] = Pvalue;
//   //printf("PVALUE = %f\n", Pvalue);
// }

    #undef y4d
    #undef x4d
    #undef k4d
    #undef x3d
}
// __global__ void unroll_Kernel(int C, int H, int W, int K, float* x, float* X_unroll) {
//     int c, s, h_out, w_out, h_unroll, w_base, p, q;
//     int b = blockIdx.x;
//     int t = blockIdx.x * blockDim.x + threadIdx.x;
//     int H_out = H - K + 1;
//     int W_out = W - K + 1;
//     int W_unroll = H_out * W_out;
//
//
//     #define y4d(i3,i2,i1,i0) y[(i3) * (M * H_out * W_out) + (i2)*(H_out * W_out) + (i1)*(W_out) + i0]
//     #define x4d(i3,i2,i1,i0) x[(i3) * (C * H * W) + (i2)*(H * W) + (i1)*(W) + i0]
//     #define k4d(i3,i2,i1,i0) k[(i3) * (C * K * K) + (i2)*(K * K) + (i1)*(K) + i0]
//     #define x3d(i2, i1, i0) X_unroll[(i2) * (C * K * K * W_out * H_out)  + (i1) * (W_out * H_out) + i0]
//
//     if (t < C * W_unroll) {
//     c = t / W_unroll;
//     s = t % W_unroll;
//     h_out = s / W_out;
//     w_out = s % W_out;
//     h_unroll = h_out * W_out + w_out;
//     w_base = c * K * K;
//     for(p = 0; p < K; p++)
//     for(q = 0; q < K; q++) {
//     	int w_unroll = w_base + p * K + q;
//     	//X_unroll(b, h_unroll, w_unroll) = X(b, c, h_out + p, w_out + q);
//       x3d(b, w_unroll, h_unroll) = x4d(b, c, h_out + p, w_out + q);
//         }
//     }
//
//     #undef y4d
//     #undef x4d
//     #undef k4d
//     #undef x3d
//
// }

__global__ void unroll_Kernel(int B, int C, int H, int W, int K, float* x, float* X_unroll) {
    int c, s, h_out, w_out, h_unroll, w_base, p, q;
    int b = blockIdx.x;
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int t = threadIdx.x;
    int H_out = H - K + 1;
    int W_out = W - K + 1;
    int W_unroll = H_out * W_out;


    #define y4d(i3,i2,i1,i0) y[(i3) * (M * H_out * W_out) + (i2)*(H_out * W_out) + (i1)*(W_out) + i0]
    #define x4d(i3,i2,i1,i0) x[(i3) * (C * H * W) + (i2)*(H * W) + (i1)*(W) + i0]
    #define k4d(i3,i2,i1,i0) k[(i3) * (C * K * K) + (i2)*(K * K) + (i1)*(K) + i0]
    #define x3d(i2, i1, i0) X_unroll[(i2) * (C * K * K * W_out * H_out)  + (i1) * (W_out * H_out) + i0]


    if (tx < C * W_unroll * B) {
    c = t / W_unroll;
    s = t % W_unroll;
    h_out = s / W_out;
    w_out = s % W_out;
    h_unroll = h_out * W_out + w_out;
    w_base = c * K * K;
    for(p = 0; p < K; p++)
    for(q = 0; q < K; q++) {
    	int w_unroll = w_base + p * K + q;
    	//X_unroll(b, h_unroll, w_unroll) = X(b, c, h_out + p, w_out + q);
      x3d(b, w_unroll, h_unroll) = x4d(b, c, h_out + p, w_out + q);
      //  if(b == 6582 && w_unroll == 20 && h_unroll == 125)
      //    printf("BAOQI\n");
          }
        __syncthreads();
    }

    #undef y4d
    #undef x4d
    #undef k4d
    #undef x3d

}




/*
   This function is called by new-inl.h
   Any code you write should be executed by this function.
   For ECE408, we only expect the float version of the operator to be called, so here we specialize with only floats.
*/
template<>
void forward<gpu, float>(mshadow::Tensor<gpu, 4, float> &y, const mshadow::Tensor<gpu, 4, float> &x, const mshadow::Tensor<gpu, 4, float> &w) {


    // Use mxnet's CHECK_EQ to do assertions.
    // Remove this assertion when you do your implementation!
    //CHECK_EQ(0, 1) << "Missing an ECE408 GPU implementation!";

    // You'll probably need to launch kernels against the right stream to keep MXNet happy
    cudaStream_t s = y.stream_->stream_;

    const int B = y.shape_[0];
    const int M = y.shape_[1];
    const int C = x.shape_[1];
    const int H = x.shape_[2];
    const int W = x.shape_[3];
    const int K = w.shape_[3];



    int W_out = W-K+1;
    int H_out = H-K+1;

    float *deviceY;
    float *deviceX;
    float *deviceW;
    float *deviceX_unroll;
    //float *localY;



    //float *deviceW_unroll;
    //float *hostY;

    int size_x = B*C*H*W*sizeof(float);
    int size_y = B*M*H_out*W_out*sizeof(float);
    int size_w = M*C*K*K*sizeof(float);
    int size_x_unroll = C*K*K*H_out*W_out;


    //localY = (float*) malloc(size_y);
    //int size_w_unroll = M*C*K*K;
    size_t shmem_size = sizeof(float)*((TILE_WIDTH+K-1)*(TILE_WIDTH+K-1)+K*K);

    MSHADOW_CUDA_CALL(cudaMalloc((void**) &deviceX, size_x));
    MSHADOW_CUDA_CALL(cudaMalloc((void**) &deviceY, size_y));
    MSHADOW_CUDA_CALL(cudaMalloc((void**) &deviceW, size_w));
    MSHADOW_CUDA_CALL(cudaMalloc((void**) &deviceX_unroll, size_x_unroll));
    //MSHADOW_CUDA_CALL(cudaMalloc((void**) &deviceW_unroll, size_w_unroll));


    MSHADOW_CUDA_CALL(cudaMemcpy(deviceX, x.dptr_, size_x, cudaMemcpyHostToDevice));
    MSHADOW_CUDA_CALL(cudaMemcpy(deviceW, w.dptr_, size_w, cudaMemcpyHostToDevice));

    // int W_grid = (int)ceil(W_out*1.0 / TILE_WIDTH);
    // int H_grid = (int)ceil(H_out*1.0 / TILE_WIDTH);
    // int BLK = W_grid * H_grid;
    int numBC=W_out*H_out;
    int numAR=M;

    //int BLK = (int)ceil(1.0*W_out/TILE_WIDTH*H_out/TILE_WIDTH);
    //dim3 gridDim(ceil(numBC*1.0/TILE_WIDTH),ceil(numAR*1.0/TILE_WIDTH) , B);
    dim3 gridDim(ceil(numBC*1.0/TILE_WIDTH),ceil(numAR*1.0/TILE_WIDTH) , B);
    dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1);


    dim3 gridUnroll(B, 1, 1);
    dim3 blockUnroll(C*H_out*W_out, 1, 1);

    //forward_kernel<<<gridDim, blockDim, 0, s>>>(y.dptr_,x.dptr_,w.dptr_, B,M,C,H,W,K);
  unroll_Kernel<<<gridUnroll, blockUnroll>>>(B, C, H, W, K, deviceX, deviceX_unroll);
	forward_kernel<<<gridDim, blockDim, shmem_size, s>>>(deviceY, deviceX_unroll, deviceW, B,M,C,H,W,K);
	MSHADOW_CUDA_CALL(cudaMemcpy(y.dptr_, deviceY, size_y, cudaMemcpyDeviceToHost));
  // MSHADOW_CUDA_CALL(cudaMemcpy(localY, deviceY, size_y, cudaMemcpyDeviceToHost));
  //
  // for (int k = 147542400; k < 147543400; k++) {
	//   printf("%f ,", &(y.dptr_ + k));
  //   if(k % 24 == 0)
  //     printf("\n");
  //   if(k % 576 == 0)
  //     printf("----------------------------------\n");
	// }
  // printf("\n");
  //
  //
  // printf("B = %d\n", B);

    MSHADOW_CUDA_CALL(cudaFree(deviceX));
    MSHADOW_CUDA_CALL(cudaFree(deviceY));
    MSHADOW_CUDA_CALL(cudaFree(deviceW));


    // Extract the tensor dimensions into B,M,C,H,W,K
    // ...

    // Set the kernel dimensions
    // dim3 gridDim(0);
    // dim3 blockDim(0);
    // Call the kernel
    //forward_kernel<<<gridDim, blockDim, 0, s>>>(deviceY, deviceX, deviceW, B,M,C,H,W,K);

    // Use MSHADOW_CUDA_CALL to check for CUDA runtime errors.
    MSHADOW_CUDA_CALL(cudaDeviceSynchronize());

}


/*
    This tells mxnet how to do an op when it's not a float.
    This is not used in the ECE408 project
*/
template<typename gpu, typename DType>
void forward(mshadow::Tensor<gpu, 4, DType> &y, const mshadow::Tensor<gpu, 4, DType> &x, const mshadow::Tensor<gpu, 4, DType> &w) {
    assert( 0 && "No forward implementation for other datatypes needed for ECE408");
}

}
}

#endif
