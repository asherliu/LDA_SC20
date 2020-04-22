


#include "SamplingKernel.cuh"

void SampleKernelD(WTD &argWTDen, WTAll &argWT, DTChunk &argDT, Document &argDoc)
{

	int blockCounter = 0;
	int iterWT = (argWTDen.numOfWordD - 1) / GridDim + 1;
	float Perplexity = 0.0;
	srand(time(NULL));

	curandState* randState;
	cudaMalloc(&randState, sizeof(curandState)*GridDim*BlockDim);
	initRandState << <GridDim, BlockDim >> >(randState);

	for (int i = 0; i < iterWT; i++) {

		cudaMemcpy(argDoc.d_blockCounter, &blockCounter, sizeof(int), cudaMemcpyHostToDevice);

		LDAKernelTrainD << <GridDim, BlockDim >> > (alpha, beta, argDoc.deviceMapWord2Doc, argDoc.deviceTLTopic, argDT.deviceNZDTCount, argDT.deviceDTIndex, argDT.deviceDTValue, argDoc.deviceTLDocCount, argDoc.deviceTLDocOffset, argDT.deviceDTCount, argDT.deviceDTOffset, argWTDen.deviceWTDense, argWTDen.deviceWTDenseCopy, argDoc.deviceTLWordCount, argDoc.deviceTLWordOffset, argWT.deviceWTCount, argWT.deviceWTOffset, argWT.deviceWTRowSum, argDoc.d_blockCounter, argDoc.deviceMapDoc2Word, argDoc.docLengthVec[argDT.chunkId], argWT.wordLength, argDoc.devicePerplexity, randState, argDoc.deviceWTHeadDense, argWTDen.numOfWordD);
		H_ERR(cudaDeviceSynchronize());
		blockCounter++;

	}
	
	H_ERR(cudaDeviceSynchronize());

}
//(double alpha, double beta, int* d_Index, int* d_TopicIndex, int* d_SparseDTCount, int* d_SparseDTIndex, int* d_SparseDTValue, int* d_TokenCountDT, int* d_TokenOffsetDT, int* d_DocListCount, int* d_DocListOffset, int* d_WTDense, int* d_WTDenseCopy, int* d_TokenCount, int* d_TokenOffset, int* d_WordListCount, int* d_WordListOffset, int* d_WTRowSum, int* d_blockCounter, int*d_DocIndex, int D, int W, double* d_Perplexity, curandState *randState, double *WTHeadDense, int numOfWordD);


void SampleKernel(WTAll &argWT, DTChunk &argDT, Document &argDoc) {

	int blockCounter = 0;
	int iterWT = (argWT.numOfWordS - 1) / GridDim + 1;
	float Perplexity = 0.0;
	int numOfWordD = argWT.wordLength- argWT.numOfWordS;
	srand(time(NULL));

	curandState* randState;
	cudaMalloc(&randState, sizeof(curandState)*GridDim*BlockDim);
	initRandState << <GridDim, BlockDim >> >(randState);

	for (int i = 0; i < iterWT; i++) {

		cudaMemcpy(argDoc.d_blockCounter, &blockCounter, sizeof(int), cudaMemcpyHostToDevice);

		LDAKernelTrain << <GridDim, BlockDim >> > (alpha, beta, argDoc.deviceMapWord2Doc, argDoc.deviceTLTopic, argDT.deviceNZDTCount, argDT.deviceDTIndex, argDT.deviceDTValue, argDoc.deviceTLDocCount, argDoc.deviceTLDocOffset, argDT.deviceDTCount, argDT.deviceDTOffset, argWT.deviceNZWTCount, argWT.deviceWTIndex, argWT.deviceWTValue, argDoc.deviceTLWordCount, argDoc.deviceTLWordOffset, argWT.deviceWTCount, argWT.deviceWTOffset, argWT.deviceWTRowSum, argDoc.d_blockCounter, argDoc.deviceMapDoc2Word, argDoc.docLengthVec[argDT.chunkId], argWT.wordLength, argDoc.devicePerplexity, randState, argDoc.deviceWTHeadDense, numOfWordD, argWT.numOfWordS);
		H_ERR(cudaDeviceSynchronize());
		blockCounter++;

	}
	LDATrainPerplexityReduce1 << <GridDim, BlockDim >> > (argDoc.devicePerplexity, argDoc.devicePerplexityMid, argDoc.TLLengthVec[argDT.chunkId]);

	H_ERR(cudaDeviceSynchronize());


}








