#include "WTUpdateKernel.cuh"
void UpdateWTKernel(WTAll &argWT, Document &argDoc, int argChunkId) {

	
	unsigned int* deviceCounter;
	cudaMalloc(&deviceCounter, sizeof(unsigned int));
	cudaMemset(deviceCounter, 0, sizeof(unsigned int));

	int numOfTokenD = argDoc.numOfTokenVecD[argChunkId];
	int numOfWordS = argWT.blockCount + argWT.warpCount;

	WT_Update_Kernel << <GridDim, BlockDim >> > (argDoc.deviceTLTopic, argWT.deviceChunkNZWTCount, argWT.deviceChunkWTIndex, argWT.deviceChunkWTValue, argWT.deviceChunkWTCount, argWT.deviceChunkWTOffset, argWT.deviceWTRowSum, deviceCounter, numOfWordS, argDoc.d_dense, numOfTokenD);

	H_ERR(cudaDeviceSynchronize());
	
}


//
//
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




