#include "WTDenUpdateKernel.cuh"
//void UpdateWTDenKernel(WTD &argWTDen, WTAll &argWT, Document &argDoc, int argChunkId, int argStreamId, cudaStream_t& stream) {
//
//
//
//	/*int numOfTokenD = argWTDen.numOfWordD;*/
//
//		/*unsigned int* deviceCounter;
//		cudaMalloc(&deviceCounter, sizeof(unsigned int));*/
//		cudaMemsetAsync(argDoc.deviceCounterWTDenUpdateKernel[argStreamId], 0, sizeof(unsigned int), stream);
//		/*cudaMemcpyAsync(argDoc.deviceCounterWTDenUpdateKernel[argStreamId], &argDoc.counterWTDenUpdateKernel, sizeof(unsigned int), cudaMemcpyHostToDevice, stream);*/
//		WTDen_Update_Kernel << <GridDim, BlockDim, 0, stream >> >(argDoc.deviceTLTopic[argStreamId], argWTDen.deviceWTDense, argDoc.deviceTLWordCount[argStreamId], argDoc.deviceTLWordOffset[argStreamId], argWT.deviceWTOffset, argWTDen.numOfWordD, argDoc.deviceCounterWTDenUpdateKernel[argStreamId]);
//
//		/*H_ERR(cudaDeviceSynchronize());*/
//
//
//
//}

//void UpdateWTDenRowSumKernel(WTD &argWTDen, WTAll &argWT, cudaStream_t& stream)
//{
//	WTDen_Sum_Update_Kernel << <GridDim, BlockDim, 0, stream >> >(argWTDen.deviceWTDense, argWT.deviceWTRowSum, argWT.deviceWTOffset, argWTDen.numOfWordD);
//	/*H_ERR(cudaDeviceSynchronize());*/
//}
//
//
//void UpdateWTDenKernel1(WTD &argWTDen, WTAll &argWT, Document &argDoc, int argChunkId, int argStreamId, cudaStream_t& stream) {
//
//
//
//	/*int numOfTokenD = argWTDen.numOfWordD;*/
//
//		/*unsigned int* deviceCounter;
//		cudaMalloc(&deviceCounter, sizeof(unsigned int));*/
//		cudaMemsetAsync(argDoc.deviceCounterWTDenUpdateKernel[argStreamId], 0, sizeof(unsigned int), stream);
//		/*cudaMemcpyAsync(argDoc.deviceCounterWTDenUpdateKernel[argStreamId], &argDoc.counterWTDenUpdateKernel, sizeof(unsigned int), cudaMemcpyHostToDevice, stream);*/
//		WTDen_Update_Kernel << <GridDim, BlockDim, 0, stream >> >(argDoc.deviceTLTopic[argStreamId], argWTDen.deviceWTDenseCopy, argDoc.deviceTLWordCount[argStreamId], argDoc.deviceTLWordOffset[argStreamId], argWT.deviceWTOffset, argWTDen.numOfWordD, argDoc.deviceCounterWTDenUpdateKernel[argStreamId]);
//
//		/*H_ERR(cudaDeviceSynchronize());*/
//
//
//
//}
