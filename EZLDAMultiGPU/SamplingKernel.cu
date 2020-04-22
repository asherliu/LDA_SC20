


#include "SamplingKernel.cuh"


#define gpuErr(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}


void SampleKernelD(WTD &argWTDen, WTAll &argWT, DTChunk &argDT, Document &argDoc, curandState* randState, int argChunkId, int argGPUId, cudaStream_t &stream)
{

	unsigned int blockCounter = 0;
	int iterWT = (argWTDen.numOfWordD - 1) / GridDim + 1;
	float Perplexity = 0.0;
	unsigned int* deviceCounter;
	cudaSetDevice(argGPUId);
	cudaMalloc(&deviceCounter, sizeof(unsigned int));
	// srand(time(NULL));

	// curandState* randState;
	// cudaMalloc(&randState, sizeof(curandState)*GridDim*BlockDim);
	// H_ERR(cudaDeviceSynchronize());
   
 //    gpuErr(cudaPeekAtLastError());
	
	initRandState << <GridDim, BlockDim >> >(randState);
	H_ERR(cudaDeviceSynchronize());

	// for (int i = 0; i < iterWT; i++) {

	cudaMemcpy(deviceCounter, &blockCounter, sizeof(unsigned int), cudaMemcpyHostToDevice);

	LDAKernelTrainD << <GridDim, BlockDim>> > (alpha, beta, argDoc.GPUChunkVec[argGPUId].deviceMapWord2Doc, argDoc.GPUChunkVec[argGPUId].deviceTLTopic, argDT.DTGPUChunkVec[argGPUId].deviceNZDTCount, argDT.DTGPUChunkVec[argGPUId].deviceDTIndex, argDT.DTGPUChunkVec[argGPUId].deviceDTValue, argDoc.GPUChunkVec[argGPUId].deviceTLDocCount, argDoc.GPUChunkVec[argGPUId].deviceTLDocOffset, argDT.DTGPUChunkVec[argGPUId].deviceDTCount, argDT.DTGPUChunkVec[argGPUId].deviceDTOffset, argWTDen.WTDenseGPUChunkVec[argGPUId].deviceWTDense, argWTDen.WTDenseGPUChunkVec[argGPUId].deviceWTDenseCopy, argDoc.GPUChunkVec[argGPUId].deviceTLWordCount, argDoc.GPUChunkVec[argGPUId].deviceTLWordOffset, argWT.WTGPUChunkVec[argGPUId].deviceWTCount, argWT.WTGPUChunkVec[argGPUId].deviceWTOffset, argWT.WTGPUChunkVec[argGPUId].deviceWTRowSum, deviceCounter, argDoc.GPUChunkVec[argGPUId].deviceMapDoc2Word, argDoc.docLengthVec[argGPUId], argWT.wordLength, argDoc.GPUChunkVec[argGPUId].devicePerplexity, randState, argDoc.GPUChunkVec[argGPUId].deviceWTHeadDense, argWTDen.numOfWordD);
	// H_ERR(cudaDeviceSynchronize());
	// 	blockCounter++;

	// }
	
	H_ERR(cudaDeviceSynchronize());

}
//(double alpha, double beta, int* d_Index, int* d_TopicIndex, int* d_SparseDTCount, int* d_SparseDTIndex, int* d_SparseDTValue, int* d_TokenCountDT, int* d_TokenOffsetDT, int* d_DocListCount, int* d_DocListOffset, int* d_WTDense, int* d_WTDenseCopy, int* d_TokenCount, int* d_TokenOffset, int* d_WordListCount, int* d_WordListOffset, int* d_WTRowSum, int* d_blockCounter, int*d_DocIndex, int D, int W, double* d_Perplexity, curandState *randState, double *WTHeadDense, int numOfWordD);


void SampleKernel(WTAll &argWT, DTChunk &argDT, Document &argDoc, curandState* randState, int argChunkId, int argGPUId, cudaStream_t &stream) {

	int blockCounter = 0;
	int iterWT = (argWT.numOfWordS - 1) / GridDim + 1;
	float Perplexity = 0.0;
	int numOfWordD = argWT.wordLength- argWT.numOfWordS;
	// srand(time(NULL));

	// curandState* randState;
	// cudaMalloc(&randState, sizeof(curandState)*GridDim*BlockDim);
	// H_ERR(cudaDeviceSynchronize());
 //    gpuErr(cudaPeekAtLastError());
	cudaSetDevice(argGPUId);
	initRandState << <GridDim, BlockDim >> >(randState);
	H_ERR(cudaDeviceSynchronize());

	for (int i = 0; i < iterWT; i++) {

		/*cudaMemcpy(argDoc.d_blockCounter, &blockCounter, sizeof(int), cudaMemcpyHostToDevice);*/

		LDAKernelTrain << <GridDim, BlockDim >> > (alpha, beta, argDoc.GPUChunkVec[argGPUId].deviceMapWord2Doc, argDoc.GPUChunkVec[argGPUId].deviceTLTopic, argDT.DTGPUChunkVec[argGPUId].deviceNZDTCount, argDT.DTGPUChunkVec[argGPUId].deviceDTIndex, argDT.DTGPUChunkVec[argGPUId].deviceDTValue, argDoc.GPUChunkVec[argGPUId].deviceTLDocCount, argDoc.GPUChunkVec[argGPUId].deviceTLDocOffset, argDT.DTGPUChunkVec[argGPUId].deviceDTCount, argDT.DTGPUChunkVec[argGPUId].deviceDTOffset, argWT.WTGPUChunkVec[argGPUId].deviceNZWTCount, argWT.WTGPUChunkVec[argGPUId].deviceWTIndex, argWT.WTGPUChunkVec[argGPUId].deviceWTValue, argDoc.GPUChunkVec[argGPUId].deviceTLWordCount, argDoc.GPUChunkVec[argGPUId].deviceTLWordOffset, argWT.WTGPUChunkVec[argGPUId].deviceWTCount, argWT.WTGPUChunkVec[argGPUId].deviceWTOffset, argWT.WTGPUChunkVec[argGPUId].deviceWTRowSum, blockCounter, argDoc.GPUChunkVec[argGPUId].deviceMapDoc2Word, argDoc.docLengthVec[argGPUId], argWT.wordLength, argDoc.GPUChunkVec[argGPUId].devicePerplexity, randState, argDoc.GPUChunkVec[argGPUId].deviceWTHeadDense, numOfWordD, argWT.numOfWordS);
		H_ERR(cudaDeviceSynchronize());
		blockCounter++;

	}
	cudaSetDevice(argGPUId);
	LDATrainPerplexityReduce1 << <GridDim, BlockDim>> > (argDoc.GPUChunkVec[argGPUId].devicePerplexity, argDoc.GPUChunkVec[argGPUId].devicePerplexityMid, argDoc.TLLengthVec[argGPUId]);

	H_ERR(cudaDeviceSynchronize());


}








