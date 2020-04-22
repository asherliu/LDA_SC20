#ifndef _UTILITY_H_
#define _UTILITY_H_


#include<iostream>
#include<string>
#include <stdio.h>
#include <stdlib.h>

#include<vector>
#include<map>
#include<numeric>

#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <curand.h>
#include <curand_kernel.h>
#include <ctime> 
//#include <windows.h>  

#include <fstream>
#include<algorithm>
#include "Argument.cuh"



using namespace std;

//static void HandleError(cudaError_t err,
//	const char *file,
//	int line);
//#define H_ERR( err ) \
//  (HandleError( err, __FILE__, __LINE__ ))

__device__ void prefix_sum(volatile int* p);
__device__ void radix_sort(volatile int* p);
__device__ void index_value_count(volatile int* p, volatile int *index, volatile int *value);
__device__ void dense_sparse_kernel(volatile int *p, int* index, int* value, int* count, int *d_sparse_slotcount, int *d_sparse_slotoffset, int *d_counter_0);
__device__ void dense_sparse_kernel2(volatile int *p, unsigned short int* index, unsigned short int* value, int* count, int *d_sparse_slotcount, int *d_slotoffset, int *d_counter_0, int numOfwordD);
__device__ void warp_prefix_sum(int& p);
__device__ void warp_radix_sort(volatile int* p);
__device__ void warp_index_value_count(volatile int* p, volatile int *index, volatile int *value);
__global__ void tokenlist_to_matrix_warp(int *d_a, int *d_count, int *d_index, int *d_value, int *d_slotcount, int *d_slotoffset, int *d_row_sum, int *d_counter_1, int* d_token_amount_0, int* d_token_amount_1, int numOfTokenD);
//__global__ void tokenlist_to_matrix_warp(int *d_a, int *d_count, int *d_index, int *d_value, int *d_slotcount, int *d_slotoffset, int* d_sparse_slotcount, int* d_sparse_slotoffset, int *d_row_sum, int *d_counter_1, int* d_token_amount_0, int* d_token_amount_1);
__global__ void tokenlist_to_matrix(int *d_a, int *d_count, int *d_index, int *d_value, int *d_slotcount, int *d_slotoffset, int *d_row_sum, int *d_counter_0, int *d_token_amount_0, int *d_dense, int numOfTokenD);
//__global__ void tokenlist_to_matrix(int *d_a, int *d_count, int *d_index, int *d_value, int *d_slotcount, int *d_slotoffset, int *d_sparse_slotcount, int *d_sparse_slotoffset, int *d_row_sum, int *d_counter_0, int *d_token_amount_0, int *d_dense);

__global__ void WT_Update_Kernel(unsigned short int *d_a, int *d_count, unsigned short int *d_index, unsigned short int *d_value, int *d_slotcount, int *d_slotoffset, int *d_row_sum, unsigned int *d_counter_0, int d_token_amount_0, int *d_dense, int numOfTokenD);

__global__ void DT_Update_Kernel(int *d_Index, unsigned short int *d_a, int *d_count, unsigned short int *d_index, int *d_value, int *d_slotcount, int *d_slotoffset, int *d_sparse_slotcount, int *d_sparse_slotoffset, unsigned int *d_counter_0, int argD, int *d_dense);
__global__ void WTDen_Update_Kernel(unsigned short int *deviceTopic, int *deviceWTDense, int *deviceTLCount, int *deviceTLOffset, int *deviceWTOffset, int numOfWordD, unsigned int* deviceCounter);

__global__ void sparseMatrixAdd(int* argCount0, int* argOffset0, int* argNZCount0, unsigned short int* argIndex0, unsigned short int* argValue0, int* argCount1, int* argOffset1, int* argNZCount1, unsigned short int* argIndex1, unsigned short int* argValue1, int* argDense, int argNumRows, unsigned int* deviceCounter, int* argWTRowSum, int numOfWordD);

//__global__ void LDAKernelTrain(float alpha, float beta, int* d_Index, int* d_TopicIndex, int* d_SparseDTCount, int* d_SparseDTIndex, int* d_SparseDTValue, int* d_TokenCountDT, int* d_TokenOffsetDT, int* d_DocListCount, int* d_DocListOffset, int* d_SparseWTCount, int* d_SparseWTIndex, int* d_SparseWTValue, int* d_TokenCount, int* d_TokenOffset, int* d_WordListCount, int* d_WordListOffset, int* d_WTRowSum, int* d_blockCounter, int*d_DocIndex, int D, int W, float* d_Perplexity, curandState *randState, float *WTHeadDense);
__global__ void LDAKernelTrain(float alpha, float beta, int* d_Index, unsigned short int* d_TopicIndex, int* d_SparseDTCount, unsigned short int* d_SparseDTIndex, int* d_SparseDTValue, int* d_TokenCountDT, int* d_TokenOffsetDT, int* d_DocListCount, int* d_DocListOffset, int* d_SparseWTCount, unsigned short int* d_SparseWTIndex, unsigned short int* d_SparseWTValue, int* d_TokenCount, int* d_TokenOffset, int* d_WordListCount, int* d_WordListOffset, int* d_WTRowSum, unsigned int* d_blockCounter, int*d_DocIndex, int D, int W, float* d_Perplexity, curandState *randState, float *WTHeadDense, int numOfWordD, int numOfWordS);
__global__ void LDAKernelTrainD(float alpha, float beta, int* d_Index, unsigned short int* d_TopicIndex, int* d_SparseDTCount, unsigned short int* d_SparseDTIndex, int* d_SparseDTValue, int* d_TokenCountDT, int* d_TokenOffsetDT, int* d_DocListCount, int* d_DocListOffset, int* d_WTDense, int* d_WTDenseCopy, int* d_TokenCount, int* d_TokenOffset, int* d_WordListCount, int* d_WordListOffset, int* d_WTRowSum, unsigned int* d_blockCounter, int*d_DocIndex, int D, int W, float* d_Perplexity, curandState *randState, float *WTHeadDense, int numOfWordD,int tokenSegment);
__global__ void LDATrainPerplexityReduce1(float *perplexity, float *perplexityMid, int numVals);
__global__ void LDATrainPerplexityReduce(float *perplexity, float numOfTokens, float *sumPerlexity);
__global__ void initRandState(curandState *state);

__global__ void WTDen_Sum_Update_Kernel(int *deviceWTDense, int *deviceWTRowSum, int *deviceWTOffset, int numOfWordD);





#endif
