#include "WTDenUpdateKernel.cuh"
void UpdateWTDenKernel(WTD &argWTDen, WTAll &argWT, Document &argDoc, int argChunkId) {



	/*int numOfTokenD = argWTDen.numOfWordD;*/

		unsigned int* deviceCounter;
		cudaMalloc(&deviceCounter, sizeof(unsigned int));
		cudaMemset(deviceCounter, 0, sizeof(unsigned int));

		WTDen_Update_Kernel << <GridDim, BlockDim >> >(argDoc.deviceTLTopic, argWTDen.deviceWTDense, argDoc.deviceTLWordCount, argDoc.deviceTLWordOffset, argWT.deviceWTOffset, argWTDen.numOfWordD, deviceCounter);

		H_ERR(cudaDeviceSynchronize());
		


}

void UpdateWTDenRowSumKernel(WTD &argWTDen, WTAll &argWT)
{
	WTDen_Sum_Update_Kernel << <GridDim, BlockDim >> >(argWTDen.deviceWTDense, argWT.deviceWTRowSum, argWT.deviceWTOffset, argWTDen.numOfWordD);
	H_ERR(cudaDeviceSynchronize());
}



