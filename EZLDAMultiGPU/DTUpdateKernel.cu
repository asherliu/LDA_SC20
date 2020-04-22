
#include "DTUpdateKernel.cuh"
void UpdateDTKernel(DTChunk &argDT,Document &argDoc, int argGPUId, cudaStream_t &stream) {

	int blockCounter = 0;
	int GPUId = argGPUId;
	int chunkId = argGPUId;
	int iterDT = (argDT.docLengthVec[chunkId] - 1) / GridDim + 1;
	cudaSetDevice(GPUId);
	for (int i = 0; i < iterDT; i++) {
		/*cudaMemcpy(argDoc.d_blockCounter, &blockCounter, sizeof(int), cudaMemcpyHostToDevice);*/
		DT_Update_Kernel << <GridDim, BlockDim>> > (argDoc.GPUChunkVec[GPUId].deviceMapWord2Doc, argDoc.GPUChunkVec[GPUId].deviceTLTopic,  argDT.DTGPUChunkVec[GPUId].deviceNZDTCount, argDT.DTGPUChunkVec[GPUId].deviceDTIndex, argDT.DTGPUChunkVec[GPUId].deviceDTValue, argDoc.GPUChunkVec[GPUId].deviceTLDocCount, argDoc.GPUChunkVec[GPUId].deviceTLDocOffset, argDT.DTGPUChunkVec[GPUId].deviceDTCount, argDT.DTGPUChunkVec[GPUId].deviceDTOffset, blockCounter, argDT.docLengthVec[chunkId], argDoc.GPUChunkVec[GPUId].d_dense);
		
		cudaDeviceSynchronize();
		blockCounter++;
	}
	
}

