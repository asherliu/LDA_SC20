
#include "DTUpdateKernel.cuh"
void UpdateDTKernel(DTChunk &argDT,Document &argDoc, int argStreamId, cudaStream_t& stream) {


	/*unsigned int* deviceCounter;
	cudaMalloc(&deviceCounter, sizeof(unsigned int));*/
	cudaMemsetAsync(argDoc.deviceCounterDTUpdateKernel[argStreamId], 0, sizeof(unsigned int), stream);

	/*cudaMemcpyAsync(argDoc.deviceCounterDTUpdateKernel[argStreamId], &argDoc.counterDTUpdateKernel, sizeof(unsigned int),cudaMemcpyHostToDevice, stream);*/

	DT_Update_Kernel << <GridDim, BlockDim, 0, stream >> > (argDoc.deviceMapWord2Doc[argStreamId], argDoc.deviceTLTopic[argStreamId],  argDT.deviceNZDTCount[argStreamId], argDT.deviceDTIndex[argStreamId], argDT.deviceDTValue[argStreamId], argDoc.deviceTLDocCount[argStreamId], argDoc.deviceTLDocOffset[argStreamId], argDT.deviceDTCount[argStreamId], argDT.deviceDTOffset[argStreamId], argDoc.deviceCounterDTUpdateKernel[argStreamId], argDT.docLengthVec[argDT.chunkId], argDoc.d_dense[argStreamId], argDoc.deviceMaxTokenCount[argStreamId], argDoc.deviceMaxTopic[argStreamId], argDoc.deviceSecondMaxTopic[argStreamId], argDoc.deviceSecondMaxTokenCount[argStreamId]);
	/*H_ERR(cudaDeviceSynchronize());*/
	
}

