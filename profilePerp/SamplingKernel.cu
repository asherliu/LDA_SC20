


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


void SampleKernelD(WTD &argWTDen, WTAll &argWT, DTChunk &argDT, Document &argDoc, curandState* randState)
{

	//unsigned int blockCounter = 0;
	unsigned int* deviceCounter;
	cudaMalloc(&deviceCounter, sizeof(unsigned int));
	cudaMemset(deviceCounter, 0, sizeof(unsigned int));
	// srand(time(NULL));

	// curandState* randState;
	// cudaMalloc(&randState, sizeof(curandState)*GridDim*BlockDim);
	// H_ERR(cudaDeviceSynchronize());
   
 //    gpuErr(cudaPeekAtLastError());

	initRandState << <GridDim, BlockDim >> >(randState);
	H_ERR(cudaDeviceSynchronize());

	// for (int i = 0; i < iterWT; i++) {

	//cudaMemcpy(deviceCounter, &blockCounter, sizeof(unsigned int), cudaMemcpyHostToDevice);

	LDAKernelTrainD << <GridDim, BlockDim >> > (alpha, beta, argDoc.deviceMapWord2Doc, argDoc.deviceTLTopic, argDT.deviceNZDTCount, argDT.deviceDTIndex, argDT.deviceDTValue, argDoc.deviceTLDocCount, argDoc.deviceTLDocOffset, argDT.deviceDTCount, argDT.deviceDTOffset, argWTDen.deviceWTDense, argWTDen.deviceWTDenseCopy, argDoc.deviceTLWordCount, argDoc.deviceTLWordOffset, argWT.deviceWTCount, argWT.deviceWTOffset, argWT.deviceWTRowSum, deviceCounter, argDoc.deviceMapDoc2Word, argDoc.docLengthVec[argDT.chunkId], argWT.wordLength, argDoc.devicePerplexityMid, randState, argDoc.deviceWTHeadDense, argWTDen.numOfWordD, argDoc.tokenSegment);

	
	H_ERR(cudaDeviceSynchronize());

}
//(double alpha, double beta, int* d_Index, int* d_TopicIndex, int* d_SparseDTCount, int* d_SparseDTIndex, int* d_SparseDTValue, int* d_TokenCountDT, int* d_TokenOffsetDT, int* d_DocListCount, int* d_DocListOffset, int* d_WTDense, int* d_WTDenseCopy, int* d_TokenCount, int* d_TokenOffset, int* d_WordListCount, int* d_WordListOffset, int* d_WTRowSum, int* d_blockCounter, int*d_DocIndex, int D, int W, double* d_Perplexity, curandState *randState, double *WTHeadDense, int numOfWordD);


void SampleKernel(WTAll &argWT, DTChunk &argDT, Document &argDoc, curandState* randState) {

	int numOfWordD = argWT.wordLength - argWT.numOfWordS;
	unsigned int* deviceCounter;
	cudaMalloc(&deviceCounter, sizeof(unsigned int));
	cudaMemset(deviceCounter, 0, sizeof(unsigned int));

	initRandState << <GridDim, BlockDim >> >(randState);
	H_ERR(cudaDeviceSynchronize());

	LDAKernelTrain << <GridDim, BlockDim >> > (alpha, beta, argDoc.deviceMapWord2Doc, argDoc.deviceTLTopic, argDT.deviceNZDTCount, argDT.deviceDTIndex, argDT.deviceDTValue, argDoc.deviceTLDocCount, argDoc.deviceTLDocOffset, argDT.deviceDTCount, argDT.deviceDTOffset, argWT.deviceNZWTCount, argWT.deviceWTIndex, argWT.deviceWTValue, argDoc.deviceTLWordCount, argDoc.deviceTLWordOffset, argWT.deviceWTCount, argWT.deviceWTOffset, argWT.deviceWTRowSum, deviceCounter, argDoc.deviceMapDoc2Word, argDoc.docLengthVec[argDT.chunkId], argWT.wordLength, argDoc.devicePerplexityMid, randState, argDoc.deviceWTHeadDense, numOfWordD, argWT.numOfWordS);

	H_ERR(cudaDeviceSynchronize());

	


}



void PerplexityKernel(Document &argDoc) {

	float* sumPerplexity;

	cudaMalloc(&sumPerplexity, sizeof(float));
	LDATrainPerplexityReduce << <1, BlockDim >> > (argDoc.devicePerplexityMid, argDoc.totalNumOfTokens, sumPerplexity);
	cudaMemcpy(&argDoc.sumPerplexity, sumPerplexity, sizeof(float), cudaMemcpyDeviceToHost);
	H_ERR(cudaDeviceSynchronize());
}













//
//void SampleKernel(WTAll &argWT, DTChunk &argDT, Document &argDoc, curandState* randState) {
//
//	int blockCounter = 0;
//	int iterWT = (argWT.numOfWordS - 1) / GridDim + 1;
//	float Perplexity = 0.0;
//	int numOfWordD = argWT.wordLength - argWT.numOfWordS;
//	// srand(time(NULL));
//
//	// curandState* randState;
//	// cudaMalloc(&randState, sizeof(curandState)*GridDim*BlockDim);
//	// H_ERR(cudaDeviceSynchronize());
//	//    gpuErr(cudaPeekAtLastError());
//
//	initRandState << <GridDim, BlockDim >> >(randState);
//	H_ERR(cudaDeviceSynchronize());
//
//	for (int i = 0; i < iterWT; i++) {
//
//		cudaMemcpy(argDoc.d_blockCounter, &blockCounter, sizeof(int), cudaMemcpyHostToDevice);
//
//		LDAKernelTrain << <GridDim, BlockDim >> > (alpha, beta, argDoc.deviceMapWord2Doc, argDoc.deviceTLTopic, argDT.deviceNZDTCount, argDT.deviceDTIndex, argDT.deviceDTValue, argDoc.deviceTLDocCount, argDoc.deviceTLDocOffset, argDT.deviceDTCount, argDT.deviceDTOffset, argWT.deviceNZWTCount, argWT.deviceWTIndex, argWT.deviceWTValue, argDoc.deviceTLWordCount, argDoc.deviceTLWordOffset, argWT.deviceWTCount, argWT.deviceWTOffset, argWT.deviceWTRowSum, argDoc.d_blockCounter, argDoc.deviceMapDoc2Word, argDoc.docLengthVec[argDT.chunkId], argWT.wordLength, argDoc.devicePerplexity, randState, argDoc.deviceWTHeadDense, numOfWordD, argWT.numOfWordS);
//		H_ERR(cudaDeviceSynchronize());
//		blockCounter++;
//
//	}
//	LDATrainPerplexityReduce1 << <GridDim, BlockDim >> > (argDoc.devicePerplexity, argDoc.devicePerplexityMid, argDoc.TLLengthVec[argDT.chunkId]);
//
//	H_ERR(cudaDeviceSynchronize());
//
//
//}
//
//
//
















