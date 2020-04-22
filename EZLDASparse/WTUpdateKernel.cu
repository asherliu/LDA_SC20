#include "WTUpdateKernel.cuh"
void UpdateWTKernel(WTAll &argWT, Document &argDoc, int argChunkId, int argStreamId, cudaStream_t& stream) {

	
	//unsigned int* deviceCounter;
	//cudaMalloc(&deviceCounter, sizeof(unsigned int));
	cudaMemsetAsync(argDoc.deviceCounterWTUpdateKernel[argStreamId], 0, sizeof(unsigned int),stream);
	/*cudaMemcpyAsync(argDoc.deviceCounterWTUpdateKernel[argStreamId], &argDoc.counterWTUpdateKernel, sizeof(unsigned int), cudaMemcpyHostToDevice, stream);*/

	/*int numOfTokenD = argDoc.numOfTokenVecD[argChunkId];
	int numOfWordS = argWT.blockCount + argWT.warpCount;*/

	WT_Update_Kernel << <GridDim, BlockDim, 0, stream >> > (argDoc.deviceTLTopic[argStreamId], argDoc.deviceTLWordCount[argStreamId], argDoc.deviceTLWordOffset[argStreamId], argWT.deviceNZWTCount, argWT.deviceWTIndex, argWT.deviceWTValue, argWT.deviceWTCount, argWT.deviceWTOffset, argWT.deviceWTRowSum, argDoc.deviceCounterWTUpdateKernel[argStreamId], argWT.numOfWordS, argDoc.d_dense[argStreamId], argDoc.numOfTokenVecD[argChunkId]);

	//H_ERR(cudaDeviceSynchronize());
	
}

void UpdateWTRowSumKernel(WTAll &argWT, cudaStream_t& stream)

{
	WTRow_Sum_Update_Kernel<< <GridDim, BlockDim, 0, stream >> > (argWT.deviceNZWTCount, argWT.deviceWTOffset, argWT.deviceWTIndex, argWT.deviceWTValue, argWT.deviceWTRowSum, argWT.wordLength);

}

//void UpdateWTKernel(WTAll &argWT, Document &argDoc, int argChunkId) {
//
//	int iterBlock = (argWT.blockCount - 1) / GridDim + 1;// number of iterations for block.
//														 //int iterBlock = 9;// number of iterations for block.
//	int GridWarpDim = GridDim*BlockDim / 32;
//	int iterAll = (argWT.blockCount - 1) / GridDim + 1 + (argWT.warpCount - 1) / GridWarpDim + 1; // number of total iterations.
//
//	int blockCounter = 0;
//	int warpCounter = 0;
//	int numOfTokenD = argDoc.numOfTokenVecD[argChunkId];
//	for (int i = 0; i < iterAll; i++)
//	{
//		if (i < iterBlock)
//		{
//			H_ERR(cudaMemcpy(argDoc.d_blockCounter, &blockCounter, sizeof(int), cudaMemcpyHostToDevice));
//			tokenlist_to_matrix << <GridDim, BlockDim >> > (argDoc.deviceTLTopic, argWT.deviceChunkNZWTCount, argWT.deviceChunkWTIndex, argWT.deviceChunkWTValue, argWT.deviceChunkWTCount, argWT.deviceChunkWTOffset, argWT.deviceWTRowSum, argDoc.d_blockCounter, argWT.deviceBlockCount, argDoc.d_dense, numOfTokenD);
//			H_ERR(cudaDeviceSynchronize());
//			blockCounter++;
//
//
//		}
//		else
//		{
//			cudaMemcpy(argDoc.d_warpCounter, &warpCounter, sizeof(int), cudaMemcpyHostToDevice);
//			tokenlist_to_matrix_warp << <GridDim, BlockDim >> > (argDoc.deviceTLTopic, argWT.deviceChunkNZWTCount, argWT.deviceChunkWTIndex, argWT.deviceChunkWTValue, argWT.deviceChunkWTCount, argWT.deviceChunkWTOffset, argWT.deviceWTRowSum, argDoc.d_warpCounter, argWT.deviceBlockCount, argWT.deviceWarpCount, numOfTokenD);
//			/*printf("abc %d", warpCounter);*/
//			H_ERR(cudaDeviceSynchronize());
//			warpCounter++;
//		}
//		H_ERR(cudaDeviceSynchronize());
//
//
//	}
//
//}
//




