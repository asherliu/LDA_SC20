#include "WTDenUpdateKernel.cuh"
void UpdateWTDenKernel(WTD &argWTDen, WTAll &argWT, Document &argDoc, int argChunkId) {

	int iter= (argWTDen.numOfWordD - 1) / GridDim + 1;// number of iterations for block.
	
	int counter = 0;

	/*int numOfTokenD = argWTDen.numOfWordD;*/
	for (int i = 0; i < iter; i++)
	{


		WTDen_Update_Kernel << <GridDim, BlockDim >> >(argDoc.deviceTLTopic, argWTDen.deviceWTDense, argDoc.deviceTLWordCount, argDoc.deviceTLWordOffset, argWT.deviceWTOffset, argWTDen.numOfWordD,  counter);

		H_ERR(cudaDeviceSynchronize());
		counter++;

	}

}

void UpdateWTDenRowSumKernel(WTD &argWTDen, WTAll &argWT)
{
	WTDen_Sum_Update_Kernel << <GridDim, BlockDim >> >(argWTDen.deviceWTDense, argWT.deviceWTRowSum, argWT.deviceWTOffset, argWTDen.numOfWordD);
	H_ERR(cudaDeviceSynchronize());
}



