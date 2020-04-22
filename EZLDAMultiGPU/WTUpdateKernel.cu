#include "WTUpdateKernel.cuh"
void UpdateWTKernel(WTAll &argWT, Document &argDoc, int argChunkId, int argGPUId, cudaStream_t &stream) {

	int iterBlock = (argWT.blockCount - 1) / GridDim + 1;// number of iterations for block.
	//int iterBlock = 9;// number of iterations for block.
	int GridWarpDim = GridDim*BlockDim / 32;
	int iterAll = (argWT.blockCount - 1) / GridDim + 1 + (argWT.warpCount - 1) / GridWarpDim + 1; // number of total iterations.

	int blockCounter = 0;
	int warpCounter = 0;
	int GPUId = argGPUId;
	int numOfTokenD = argDoc.numOfTokenVecD[argChunkId];

	cudaSetDevice(GPUId);

	for (int i = 0; i < iterAll; i++)
	{
		if (i < iterBlock)
		{
			/*H_ERR(cudaMemcpy(argDoc.d_blockCounter, &blockCounter, sizeof(int), cudaMemcpyHostToDevice));*/
			tokenlist_to_matrix << <GridDim, BlockDim >> > (argDoc.GPUChunkVec[GPUId].deviceTLTopic, argWT.WTGPUChunkVec[GPUId].deviceChunkNZWTCount, argWT.WTGPUChunkVec[GPUId].deviceChunkWTIndex, argWT.WTGPUChunkVec[GPUId].deviceChunkWTValue,  argWT.WTGPUChunkVec[GPUId].deviceChunkWTCount, argWT.WTGPUChunkVec[GPUId].deviceChunkWTOffset, argWT.WTGPUChunkVec[GPUId].deviceWTRowSum, blockCounter, argWT.blockCount, argDoc.GPUChunkVec[GPUId].d_dense, numOfTokenD);
			 H_ERR(cudaDeviceSynchronize());
			 blockCounter++;

	
		}
		else
		{
			/*cudaMemcpy(argDoc.d_warpCounter, &warpCounter, sizeof(int), cudaMemcpyHostToDevice);*/
			tokenlist_to_matrix_warp << <GridDim, BlockDim >> > (argDoc.GPUChunkVec[GPUId].deviceTLTopic, argWT.WTGPUChunkVec[GPUId].deviceChunkNZWTCount, argWT.WTGPUChunkVec[GPUId].deviceChunkWTIndex, argWT.WTGPUChunkVec[GPUId].deviceChunkWTValue,  argWT.WTGPUChunkVec[GPUId].deviceChunkWTCount, argWT.WTGPUChunkVec[GPUId].deviceChunkWTOffset, argWT.WTGPUChunkVec[GPUId].deviceWTRowSum, warpCounter, argWT.blockCount, argWT.warpCount, numOfTokenD);
			/*printf("abc %d", warpCounter);*/
			H_ERR(cudaDeviceSynchronize());
			warpCounter++;
		}
		H_ERR(cudaDeviceSynchronize());
		
		
	}

}





