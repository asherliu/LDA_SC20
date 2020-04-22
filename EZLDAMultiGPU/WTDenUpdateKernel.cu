#include "WTDenUpdateKernel.cuh"
void UpdateWTDenKernel(WTD &argWTDen, WTAll &argWT, Document &argDoc, int argChunkId, int argGPUId, cudaStream_t &stream) {

	int iter= (argWTDen.numOfWordD - 1) / GridDim + 1;// number of iterations for block.
	
	int counter = 0;
	int GPUId = argGPUId;
	cudaSetDevice(GPUId);
	/*int numOfTokenD = argWTDen.numOfWordD;*/
	for (int i = 0; i < iter; i++)
	{

		
		WTDen_Update_Kernel << <GridDim, BlockDim >> >(argDoc.GPUChunkVec[GPUId].deviceTLTopic, argWTDen.WTDenseGPUChunkVec[GPUId].deviceWTDenseCopy, argDoc.GPUChunkVec[GPUId].deviceTLWordCount, argDoc.GPUChunkVec[GPUId].deviceTLWordOffset, argWT.WTGPUChunkVec[GPUId].deviceWTOffset, argWTDen.numOfWordD,  counter);

		H_ERR(cudaDeviceSynchronize());
		counter++;

	}

}

void UpdateWTDenRowSumKernel(WTD &argWTDen, WTAll &argWT, int argGPUId, cudaStream_t &stream)
{
	int GPUId = argGPUId;
	cudaSetDevice(GPUId);

	WTDen_Sum_Update_Kernel << <GridDim, BlockDim>> >(argWTDen.WTDenseGPUChunkVec[0].deviceWTDense, argWT.WTGPUChunkVec[GPUId].deviceWTRowSum, argWT.WTGPUChunkVec[GPUId].deviceWTOffset, argWTDen.numOfWordD);
	H_ERR(cudaDeviceSynchronize());
}



