
#include "DTUpdateKernel.cuh"
void UpdateDTKernel(DTChunk &argDT,Document &argDoc) {

	int blockCounter = 0;

	int iterDT = (argDT.docLengthVec[argDT.chunkId] - 1) / GridDim + 1;
	for (int i = 0; i < iterDT; i++) {
		cudaMemcpy(argDoc.d_blockCounter, &blockCounter, sizeof(int), cudaMemcpyHostToDevice);
		DT_Update_Kernel << <GridDim, BlockDim >> > (argDoc.deviceMapWord2Doc, argDoc.deviceTLTopic,  argDT.deviceNZDTCount, argDT.deviceDTIndex, argDT.deviceDTValue, argDoc.deviceTLDocCount, argDoc.deviceTLDocOffset, argDT.deviceDTCount, argDT.deviceDTOffset, argDoc.d_blockCounter, argDT.docLengthVec[argDT.chunkId], argDoc.d_dense);
		
		cudaDeviceSynchronize();
		blockCounter++;
	}
	
}

