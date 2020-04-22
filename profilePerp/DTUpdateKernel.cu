
#include "DTUpdateKernel.cuh"
void UpdateDTKernel(DTChunk &argDT,Document &argDoc) {


	unsigned int* deviceCounter;
	cudaMalloc(&deviceCounter, sizeof(unsigned int));
	cudaMemset(deviceCounter, 0, sizeof(unsigned int));

	DT_Update_Kernel << <GridDim, BlockDim >> > (argDoc.deviceMapWord2Doc, argDoc.deviceTLTopic,  argDT.deviceNZDTCount, argDT.deviceDTIndex, argDT.deviceDTValue, argDoc.deviceTLDocCount, argDoc.deviceTLDocOffset, argDT.deviceDTCount, argDT.deviceDTOffset, deviceCounter, argDT.docLengthVec[argDT.chunkId], argDoc.d_dense);
	H_ERR(cudaDeviceSynchronize());
	
}

