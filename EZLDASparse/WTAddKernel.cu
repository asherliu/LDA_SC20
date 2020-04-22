
#include "WTAddKernel.cuh"
//void WTAdditionKernel(WTAll &argWT, Document &argDoc, int argStreamId, cudaStream_t& stream) {
//
//
//	/*unsigned int* deviceCounter;
//	cudaMalloc(&deviceCounter, sizeof(unsigned int));*/
//	cudaMemsetAsync(argDoc.deviceCounterWTAdditionKernel[argStreamId], 0, sizeof(unsigned int), stream);
//	/*cudaMemcpyAsync(argDoc.deviceCounterWTAdditionKernel[argStreamId], &argDoc.counterWTAdditionKernel, sizeof(unsigned int), cudaMemcpyHostToDevice, stream);*/
//
//
//	int numOfWordS = argWT.blockCount + argWT.warpCount;
//	int numOfWordD = argWT.wordLength - argWT.numOfWordS;
//
//	/*int blockCounter = 0;
//	int iterBlock = (argWT.numOfWordS - 1) / GridDim + 1;
//	int* deviceWordLength;
//	int numOfWordD = argWT.wordLength-argWT.numOfWordS;*/
//	/*cudaMalloc((void**)&deviceWordLength, (1) * sizeof(int));
//
//	cudaMemcpy(deviceWordLength, &argWT.numOfWordS, sizeof(int),cudaMemcpyHostToDevice);*/
//	//for (int i = 0; i < iterBlock; i++) {
//	//	cudaMemcpy(argDoc.d_blockCounter, &blockCounter, (1) * sizeof(int), cudaMemcpyHostToDevice);
//	sparseMatrixAdd << <GridDim, BlockDim, 0, stream >> >(argWT.deviceWTCount, argWT.deviceWTOffset, argWT.deviceNZWTCount, argWT.deviceWTIndex, argWT.deviceWTValue, argWT.deviceChunkWTCount[argStreamId], argWT.deviceChunkWTOffset[argStreamId], argWT.deviceChunkNZWTCount[argStreamId], argWT.deviceChunkWTIndex[argStreamId], argWT.deviceChunkWTValue[argStreamId], argDoc.d_dense[argStreamId], argWT.numOfWordS, argDoc.deviceCounterWTAdditionKernel[argStreamId],argWT.deviceWTRowSum, numOfWordD);
//	/*H_ERR(cudaDeviceSynchronize());*/
//	/*	blockCounter++;
//	}*/
//
//
//}
