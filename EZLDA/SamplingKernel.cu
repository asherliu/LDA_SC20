


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


void SampleKernelD(WTD &argWTDen, WTAll &argWT, DTChunk &argDT, Document &argDoc, curandState* randState, int argStreamId, cudaStream_t& stream)
{

	////unsigned int blockCounter = 0;
	//unsigned int* deviceCounter;
	//cudaMalloc(&deviceCounter, sizeof(unsigned int));
	cudaMemsetAsync(argDoc.deviceCounterSampleKernelD[argStreamId], 0, sizeof(unsigned int), stream);

	/*cudaMemcpyAsync(argDoc.deviceCounterSampleKernelD[argStreamId], &argDoc.counterSampleKernelD, sizeof(unsigned int), cudaMemcpyHostToDevice, stream);*/



	// srand(time(NULL));

	// curandState* randState;
	// cudaMalloc(&randState, sizeof(curandState)*GridDim*BlockDim);
	// H_ERR(cudaDeviceSynchronize());
   
 //    gpuErr(cudaPeekAtLastError());

	//initRandState << <GridDim, BlockDim, 0, stream >> >(randState);

	/*H_ERR(cudaDeviceSynchronize());*/

	// for (int i = 0; i < iterWT; i++) {

	//cudaMemcpy(deviceCounter, &blockCounter, sizeof(unsigned int), cudaMemcpyHostToDevice);

	LDAKernelTrainD << <GridDim, BlockDim, 0, stream >> > (alpha, beta, argDoc.deviceMapWord2Doc[argStreamId], argDoc.deviceTLTopic[argStreamId], argDT.deviceNZDTCount[argStreamId], argDoc.deviceTLDocCount[argStreamId], argDoc.deviceTLDocOffset[argStreamId], argDT.deviceDTCount[argStreamId], argDT.deviceDTOffset[argStreamId], argWTDen.deviceWTDense, argWTDen.deviceWTDenseCopy, argDoc.deviceTLWordCount[argStreamId], argDoc.deviceTLWordOffset[argStreamId], argWT.deviceWTCount, argWT.deviceWTOffset, argWT.deviceWTRowSum, argDoc.deviceCounterSampleKernelD[argStreamId], argDoc.deviceMapDoc2Word[argStreamId], argDoc.docLengthVec[argDT.chunkId], argWT.wordLength, argDoc.devicePerplexityMid[argStreamId], randState, argDoc.deviceWTHeadDense[argStreamId], argWTDen.numOfWordD, argDoc.deviceWordMaxTopic[argStreamId], argDoc.deviceWordSecondMaxTopic[argStreamId], argDoc.deviceMaxProb[argStreamId], argDoc.deviceThresProb[argStreamId], argDoc.deviceTimeRecord[argStreamId], argDoc.tokenSegment, argDoc.deviceRandomfloat[argStreamId],  argDoc.deviceEffectiveTokenIndex[argStreamId], argDoc.deviceNewTokenCount[argStreamId], argDT.deviceDTIndexValue[argStreamId],argDoc.deviceMaxSecTopic[argStreamId]);

	
	/*H_ERR(cudaDeviceSynchronize());*/

}
//(double alpha, double beta, int* d_Index, int* d_TopicIndex, int* d_SparseDTCount, int* d_SparseDTIndex, int* d_SparseDTValue, int* d_TokenCountDT, int* d_TokenOffsetDT, int* d_DocListCount, int* d_DocListOffset, int* d_WTDense, int* d_WTDenseCopy, int* d_TokenCount, int* d_TokenOffset, int* d_WordListCount, int* d_WordListOffset, int* d_WTRowSum, int* d_blockCounter, int*d_DocIndex, int D, int W, double* d_Perplexity, curandState *randState, double *WTHeadDense, int numOfWordD);


void SampleKernel(WTAll &argWT, DTChunk &argDT, Document &argDoc, curandState* randState, int argStreamId, cudaStream_t& stream) {

	int numOfWordD = argWT.wordLength - argWT.numOfWordS;
	/*unsigned int* deviceCounter;
	cudaMalloc(&deviceCounter, sizeof(unsigned int));
	cudaMemset(deviceCounter, 0, sizeof(unsigned int));*/
	cudaMemsetAsync(argDoc.deviceCounterSampleKernelS[argStreamId], 0, sizeof(unsigned int), stream);

	/*cudaMemcpyAsync(argDoc.deviceCounterSampleKernelS[argStreamId], &argDoc.counterSampleKernelS, sizeof(unsigned int), cudaMemcpyHostToDevice, stream);*/

	//initRandState << <GridDim, BlockDim, 0, stream>> >(randState);
	/*H_ERR(cudaDeviceSynchronize());*/

	LDAKernelTrain << <GridDim, BlockDim, 0, stream >> > (alpha, beta, argDoc.deviceMapWord2Doc[argStreamId], argDoc.deviceTLTopic[argStreamId], argDT.deviceNZDTCount[argStreamId], argDoc.deviceTLDocCount[argStreamId], argDoc.deviceTLDocOffset[argStreamId], argDT.deviceDTCount[argStreamId], argDT.deviceDTOffset[argStreamId], argWT.deviceNZWTCount, argWT.deviceWTIndex, argWT.deviceWTValue, argDoc.deviceTLWordCount[argStreamId], argDoc.deviceTLWordOffset[argStreamId], argWT.deviceWTCount, argWT.deviceWTOffset, argWT.deviceWTRowSum, argDoc.deviceCounterSampleKernelS[argStreamId], argDoc.deviceMapDoc2Word[argStreamId], argDoc.docLengthVec[argDT.chunkId], argWT.wordLength, argDoc.devicePerplexityMid[argStreamId], randState, argDoc.deviceWTHeadDense[argStreamId], numOfWordD, argWT.numOfWordS, argDoc.deviceWordMaxTopic[argStreamId], argDoc.deviceWordSecondMaxTopic[argStreamId], argDoc.deviceMaxSecTopic[argStreamId], argDT.deviceDTIndexValue[argStreamId]);

	//H_ERR(cudaDeviceSynchronize());

	


}

void MaxTopicKernel(WTAll &argWT, Document &argDoc, WTD &argWTDen, int argStreamId, cudaStream_t& stream) {

	int numOfWordD = argWT.wordLength - argWT.numOfWordS;
	/*unsigned int* deviceCounter;
	cudaMalloc(&deviceCounter, sizeof(unsigned int));*/
	cudaMemsetAsync(argDoc.deviceCounterMaxTopicKernel[argStreamId], 0, sizeof(unsigned int),stream);
	/*cudaMemcpyAsync(argDoc.deviceCounterMaxTopicKernel[argStreamId], &argDoc.counterMaxTopicKernel, sizeof(unsigned int), cudaMemcpyHostToDevice, stream);*/
	/*cudaMemcpyAsync(argDoc.deviceCounterMaxTopicKernel[argStreamId], &argDoc.counterMaxTopicKernel, sizeof(unsigned int), cudaMemcpyHostToDevice, stream);*/

	MaxTopicDense_Update_Kernel << <GridDim, BlockDim, 0, stream >> >(argDoc.deviceWordMaxTopic[argStreamId], argDoc.deviceWordSecondMaxTopic[argStreamId],  argWTDen.deviceWTDense, argDoc.deviceTLWordCount[argStreamId], argDoc.deviceTLWordOffset[argStreamId], argWT.deviceWTOffset, numOfWordD, argDoc.deviceCounterMaxTopicKernel[argStreamId], argWT.deviceWTRowSum, argWT.wordLength, beta, argDoc.deviceWordThirdMaxTopic[argStreamId],  argDoc.deviceMaxSecTopic[argStreamId], argDoc.deviceQArray[argStreamId], argDoc.deviceWordMaxProb[argStreamId], argDoc.deviceWordSecondMaxProb[argStreamId], argDoc.deviceWordThirdMaxProb[argStreamId]);
	//H_ERR(cudaDeviceSynchronize());
	/*cudaMemcpyAsync(argDoc.deviceCounterMaxTopicKernel[argStreamId], &argDoc.counterMaxTopicKernel, sizeof(unsigned int), cudaMemcpyHostToDevice, stream);*/

	cudaMemsetAsync(argDoc.deviceCounterMaxTopicKernel[argStreamId], 0, sizeof(unsigned int),stream);

	MaxTopicSparse_Update_Kernel << <GridDim, BlockDim, 0, stream>> >(argDoc.deviceWordMaxTopic[argStreamId], argDoc.deviceWordSecondMaxTopic[argStreamId],  argDoc.deviceTLWordCount[argStreamId], argDoc.deviceTLWordOffset[argStreamId], argWT.deviceWTOffset, numOfWordD, argDoc.deviceCounterMaxTopicKernel[argStreamId], argWT.deviceWTRowSum, argWT.wordLength, argWT.numOfWordS, argWT.deviceWTOffset, argWT.deviceNZWTCount, argWT.deviceWTIndex, argWT.deviceWTValue, beta, argDoc.deviceWordThirdMaxTopic[argStreamId], argDoc.deviceMaxSecTopic[argStreamId], argDoc.deviceQArray[argStreamId], argDoc.deviceWordMaxProb[argStreamId], argDoc.deviceWordSecondMaxProb[argStreamId], argDoc.deviceWordThirdMaxProb[argStreamId]);
	/*H_ERR(cudaDeviceSynchronize());*/

}


//void UpdateProbKernelD(WTD &argWTDen, WTAll &argWT, DTChunk &argDT, Document &argDoc, curandState* randState, int argStreamId, cudaStream_t& stream)
//{
//
//	//unsigned int blockCounter = 0;
//	//unsigned int* deviceCounter;
//	//cudaMalloc(&deviceCounter, sizeof(unsigned int));
//	cudaMemsetAsync(argDoc.deviceCounterUpdateProbKernel[argStreamId], 0, sizeof(unsigned int),stream);
//	/*cudaMemcpyAsync(argDoc.deviceCounterUpdateProbKernel[argStreamId], &argDoc.counterUpdateProbKernel, sizeof(unsigned int), cudaMemcpyHostToDevice, stream);*/
//
//	initRandState << <GridDim, BlockDim, 0, stream >> >(randState);
//	/*H_ERR(cudaDeviceSynchronize());*/
//
//	// for (int i = 0; i < iterWT; i++) {
//
//	//cudaMemcpy(deviceCounter, &blockCounter, sizeof(unsigned int), cudaMemcpyHostToDevice);
//
//	UpdateProbKernelTrainD << <256, 256, 0, stream >> > (alpha, beta, argDoc.deviceMapWord2Doc[argStreamId], argDoc.deviceTLTopic[argStreamId], argDT.deviceNZDTCount[argStreamId],  argDoc.deviceTLDocCount[argStreamId], argDoc.deviceTLDocOffset[argStreamId], argDT.deviceDTCount[argStreamId], argDT.deviceDTOffset[argStreamId], argWTDen.deviceWTDense, argWTDen.deviceWTDenseCopy, argDoc.deviceTLWordCount[argStreamId], argDoc.deviceTLWordOffset[argStreamId], argWT.deviceWTCount, argWT.deviceWTOffset, argWT.deviceWTRowSum, argDoc.deviceCounterUpdateProbKernel[argStreamId], argDoc.deviceMapDoc2Word[argStreamId], argDoc.docLengthVec[argDT.chunkId], argWT.wordLength, argDoc.devicePerplexityMid[argStreamId], randState, argDoc.deviceWTHeadDense[argStreamId], argWTDen.numOfWordD, argDoc.deviceWordMaxTopic[argStreamId], argDoc.deviceWordSecondMaxTopic[argStreamId], argDoc.deviceMaxProb[argStreamId], argDoc.deviceThresProb[argStreamId], argDoc.deviceWordThirdMaxTopic[argStreamId], argDoc.deviceRandomfloat[argStreamId], argDoc.deviceEffectiveTokenIndex[argStreamId], argDoc.deviceNewTokenCount[argStreamId], argDoc.deviceMaxSecTopic[argStreamId], argDoc.deviceQArray[argStreamId], argDoc.deviceWordMaxProb[argStreamId], argDoc.deviceWordSecondMaxProb[argStreamId], argDoc.deviceWordThirdMaxProb[argStreamId], argDoc.tokenSegment);
//
//	/*H_ERR(cudaDeviceSynchronize());
//*/
//}


void UpdateProbKernelD(WTD &argWTDen, WTAll &argWT, DTChunk &argDT, Document &argDoc, curandState* randState, int argStreamId, cudaStream_t& stream)
{
	//cudaMemsetAsync(argDoc.deviceCounterUpdateProbKernel[argStreamId], 0, sizeof(unsigned int),stream);
	/*cudaMemcpyAsync(argDoc.deviceCounterUpdateProbKernel[argStreamId], &argDoc.counterUpdateProbKernel, sizeof(unsigned int), cudaMemcpyHostToDevice, stream);*/


	/*H_ERR(cudaDeviceSynchronize());*/

	// for (int i = 0; i < iterWT; i++) {

	//cudaMemcpy(deviceCounter, &blockCounter, sizeof(unsigned int), cudaMemcpyHostToDevice);

	//UpdateProbKernelTrainD0 << <256, 256, 0, stream >> > (alpha, beta, argDoc.deviceMapWord2Doc[argStreamId], argDoc.deviceTLTopic[argStreamId], argDT.deviceNZDTCount[argStreamId],  argDoc.deviceTLDocCount[argStreamId], argDoc.deviceTLDocOffset[argStreamId], argDT.deviceDTCount[argStreamId], argDT.deviceDTOffset[argStreamId], argWTDen.deviceWTDense, argWTDen.deviceWTDenseCopy, argDoc.deviceTLWordCount[argStreamId], argDoc.deviceTLWordOffset[argStreamId], argWT.deviceWTCount, argWT.deviceWTOffset, argWT.deviceWTRowSum, argDoc.deviceCounterUpdateProbKernel[argStreamId], argDoc.deviceMapDoc2Word[argStreamId], argDoc.docLengthVec[argDT.chunkId], argWT.wordLength, argDoc.devicePerplexityMid[argStreamId], randState, argDoc.deviceWTHeadDense[argStreamId], argWTDen.numOfWordD, argDoc.deviceWordMaxTopic[argStreamId], argDoc.deviceWordSecondMaxTopic[argStreamId], argDoc.deviceMaxProb[argStreamId], argDoc.deviceThresProb[argStreamId], argDoc.deviceWordThirdMaxTopic[argStreamId], argDoc.deviceRandomfloat[argStreamId], argDoc.deviceEffectiveTokenIndex[argStreamId], argDoc.deviceNewTokenCount[argStreamId], argDoc.deviceMaxSecTopic[argStreamId], argDoc.deviceQArray[argStreamId], argDoc.deviceWordMaxProb[argStreamId], argDoc.deviceWordSecondMaxProb[argStreamId], argDoc.deviceWordThirdMaxProb[argStreamId], argDoc.tokenSegment, argDoc.deviceTotalTokenCount[argStreamId]);


	cudaMemsetAsync(argDoc.deviceCounterUpdateProbKernel[argStreamId], 0, sizeof(unsigned int),stream);
	/*cudaMemcpyAsync(argDoc.deviceCounterUpdateProbKernel[argStreamId], &argDoc.counterUpdateProbKernel, sizeof(unsigned int), cudaMemcpyHostToDevice, stream);*/

	initRandState << <GridDim, BlockDim, 0, stream >> >(randState);
	/*H_ERR(cudaDeviceSynchronize());*/

	// for (int i = 0; i < iterWT; i++) {

	//cudaMemcpy(deviceCounter, &blockCounter, sizeof(unsigned int), cudaMemcpyHostToDevice);

	UpdateProbKernelTrainD1 << <256, 256, 0, stream >> > (alpha, beta, argDoc.deviceMapWord2Doc[argStreamId], argDoc.deviceTLTopic[argStreamId], argDT.deviceNZDTCount[argStreamId],  argDoc.deviceTLDocCount[argStreamId], argDoc.deviceTLDocOffset[argStreamId], argDT.deviceDTCount[argStreamId], argDT.deviceDTOffset[argStreamId], argWTDen.deviceWTDense, argWTDen.deviceWTDenseCopy, argDoc.deviceTLWordCount[argStreamId], argDoc.deviceTLWordOffset[argStreamId], argWT.deviceWTCount, argWT.deviceWTOffset, argWT.deviceWTRowSum, argDoc.deviceCounterUpdateProbKernel[argStreamId], argDoc.deviceMapDoc2Word[argStreamId], argDoc.docLengthVec[argDT.chunkId], argWT.wordLength, argDoc.devicePerplexityMid[argStreamId], randState, argDoc.deviceWTHeadDense[argStreamId], argWTDen.numOfWordD, argDoc.deviceWordMaxTopic[argStreamId], argDoc.deviceWordSecondMaxTopic[argStreamId], argDoc.deviceMaxProb[argStreamId], argDoc.deviceThresProb[argStreamId], argDoc.deviceWordThirdMaxTopic[argStreamId], argDoc.deviceRandomfloat[argStreamId], argDoc.deviceEffectiveTokenIndex[argStreamId], argDoc.deviceNewTokenCount[argStreamId], argDoc.deviceMaxSecTopic[argStreamId], argDoc.deviceQArray[argStreamId], argDoc.deviceWordMaxProb[argStreamId], argDoc.deviceWordSecondMaxProb[argStreamId], argDoc.deviceWordThirdMaxProb[argStreamId], argDoc.tokenSegment);

	/*H_ERR(cudaDeviceSynchronize());
	 *
*/
	//unsigned int blockCounter = 0;
	//unsigned int* deviceCounter;
	//cudaMalloc(&deviceCounter, sizeof(unsigned int));
	//cudaMemsetAsync(argDoc.deviceCounterUpdateProbKernel[argStreamId], 0, sizeof(unsigned int),stream);
	/*cudaMemcpyAsync(argDoc.deviceCounterUpdateProbKernel[argStreamId], &argDoc.counterUpdateProbKernel, sizeof(unsigned int), cudaMemcpyHostToDevice, stream);*/

	//initRandState << <GridDim, BlockDim, 0, stream >> >(randState);
	/*H_ERR(cudaDeviceSynchronize());*/

	// for (int i = 0; i < iterWT; i++) {

	//cudaMemcpy(deviceCounter, &blockCounter, sizeof(unsigned int), cudaMemcpyHostToDevice);

	//UpdateProbKernelTrainD2 << <256, 256, 0, stream >> > (alpha, beta, argDoc.deviceMapWord2Doc[argStreamId], argDoc.deviceTLTopic[argStreamId], argDT.deviceNZDTCount[argStreamId],  argDoc.deviceTLDocCount[argStreamId], argDoc.deviceTLDocOffset[argStreamId], argDT.deviceDTCount[argStreamId], argDT.deviceDTOffset[argStreamId], argWTDen.deviceWTDense, argWTDen.deviceWTDenseCopy, argDoc.deviceTLWordCount[argStreamId], argDoc.deviceTLWordOffset[argStreamId], argWT.deviceWTCount, argWT.deviceWTOffset, argWT.deviceWTRowSum, argDoc.deviceCounterUpdateProbKernel[argStreamId], argDoc.deviceMapDoc2Word[argStreamId], argDoc.docLengthVec[argDT.chunkId], argWT.wordLength, argDoc.devicePerplexityMid[argStreamId], randState, argDoc.deviceWTHeadDense[argStreamId], argWTDen.numOfWordD, argDoc.deviceWordMaxTopic[argStreamId], argDoc.deviceWordSecondMaxTopic[argStreamId], argDoc.deviceMaxProb[argStreamId], argDoc.deviceThresProb[argStreamId], argDoc.deviceWordThirdMaxTopic[argStreamId], argDoc.deviceRandomfloat[argStreamId], argDoc.deviceEffectiveTokenIndex[argStreamId], argDoc.deviceNewTokenCount[argStreamId], argDoc.deviceMaxSecTopic[argStreamId], argDoc.deviceQArray[argStreamId], argDoc.deviceWordMaxProb[argStreamId], argDoc.deviceWordSecondMaxProb[argStreamId], argDoc.deviceWordThirdMaxProb[argStreamId], argDoc.tokenSegment);

	/*H_ERR(cudaDeviceSynchronize());*/
}



void PerplexityKernel(Document &argDoc, int argStreamId, cudaStream_t& stream) {


	LDATrainPerplexityReduce << <1, BlockDim, 0, stream >> > (argDoc.devicePerplexityMid[argStreamId], argDoc.totalNumOfTokens, argDoc.devicePerplexityAve[argStreamId]);

	cudaMemcpyAsync(argDoc.perplexityAve, argDoc.devicePerplexityAve[argStreamId],sizeof(float), cudaMemcpyDeviceToHost, stream);

	/*H_ERR(cudaDeviceSynchronize());*/
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
















