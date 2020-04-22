
#include "WTAddKernel.cuh"
void WTAdditionKernel(WTAll &argWT, Document &argDoc, cudaStream_t &stream) {


	int blockCounter = 0;
	int iterBlock = (argWT.numOfWordS - 1) / GridDim + 1;
	int* deviceWordLength;
	int numOfWordD = argWT.wordLength-argWT.numOfWordS;

	cudaSetDevice(0);
	/*cudaMalloc((void**)&deviceWordLength, (1) * sizeof(int));
	
	cudaMemcpy(deviceWordLength, &argWT.numOfWordS, sizeof(int),cudaMemcpyHostToDevice);*/
	for (int i = 0; i < iterBlock; i++) {
		/*cudaMemcpy(argDoc.d_blockCounter, &blockCounter, (1) * sizeof(int), cudaMemcpyHostToDevice);*/
		sparseMatrixAdd << <GridDim, BlockDim>> >(argWT.WTGPUChunkVec[0].deviceWTCount, argWT.WTGPUChunkVec[0].deviceWTOffset, argWT.WTGPUChunkVec[0].deviceNZWTCount, argWT.WTGPUChunkVec[0].deviceWTIndex, argWT.WTGPUChunkVec[0].deviceWTValue, argWT.deviceZeroChunkWTCount, argWT.deviceZeroChunkWTOffset, argWT.deviceZeroChunkNZWTCount, argWT.deviceZeroChunkWTIndex, argWT.deviceZeroChunkWTValue, argDoc.GPUChunkVec[0].d_dense, argWT.numOfWordS, blockCounter, argWT.WTGPUChunkVec[0].deviceWTRowSum, numOfWordD);
		H_ERR(cudaDeviceSynchronize());
		blockCounter++;
	}


}


void WTDenAdditionKernel(WTD &argWTDen, WTAll &argWT, Document &argDoc, cudaStream_t &stream) {

	cudaSetDevice(0);
	denseMatrixAddKernel << <GridDim, BlockDim >> > (argWTDen.WTDenseGPUChunkVec[0].deviceWTDense, argWTDen.deviceZeroWTDense, argWT.WTGPUChunkVec[0].deviceWTOffset, argWTDen.numOfWordD);
	H_ERR(cudaDeviceSynchronize());
}