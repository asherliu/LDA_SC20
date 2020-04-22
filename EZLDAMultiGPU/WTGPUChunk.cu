
#include "WTGPUChunk.cuh"


WTGPUChunkData::WTGPUChunkData(int argGPUId, int argWordLength, int argMaxChunkWTLength, int argWTLength, int argNumOfWordS) {

	GPUId = argGPUId;
	wordLength = argWordLength;
	maxChunkWTLength = argMaxChunkWTLength;
	WTLength = argWTLength;
	numOfWordS = argNumOfWordS;


}


void WTGPUChunkData::GPUMemAllocate(int argGPUId) {

	GPUId = argGPUId;
	cudaSetDevice(argGPUId);
	cudaMalloc((void**)&deviceNZWTCount, (numOfWordS) * sizeof(int));
	cudaMalloc((void**)&deviceWTIndex, (maxWTLength) * sizeof(int));
	cudaMalloc((void**)&deviceWTValue, (maxWTLength) * sizeof(int));
	cudaMalloc((void**)&deviceWTCount, (wordLength) * sizeof(int));
	cudaMalloc((void**)&deviceWTOffset, (wordLength) * sizeof(int));

	cudaMalloc((void**)&deviceWTRowSum, (K) * sizeof(int));


	cudaMalloc((void**)&deviceChunkWTCount, (numOfWordS) * sizeof(int));
	cudaMalloc((void**)&deviceChunkWTOffset, (numOfWordS) * sizeof(int));
	cudaMalloc((void**)&deviceChunkNZWTCount, (numOfWordS) * sizeof(int));
	cudaMalloc((void**)&deviceChunkWTIndex, (maxChunkWTLength) * sizeof(int));
	cudaMalloc((void**)&deviceChunkWTValue, (maxChunkWTLength) * sizeof(int));



}
void WTGPUChunkData::GPUMemset(int argGPUId)
{
	cudaSetDevice(argGPUId);
	cudaMemset(deviceNZWTCount, 0, (numOfWordS) * sizeof(int));
	cudaMemset(deviceWTIndex, 0, (maxWTLength) * sizeof(int));
	cudaMemset(deviceWTValue, 0, (maxWTLength) * sizeof(int));
	cudaMemset(deviceWTRowSum, 0, (K) * sizeof(int));

}

void WTGPUChunkData::chunkGPUMemset(int argGPUId)
{

	cudaSetDevice(argGPUId);
	cudaMemset(deviceChunkNZWTCount, 0, (numOfWordS) * sizeof(int));
	cudaMemset(deviceChunkWTIndex, 0, (maxChunkWTLength) * sizeof(int));
	cudaMemset(deviceChunkWTValue, 0, (maxChunkWTLength) * sizeof(int));
	//cudaMemset(deviceWTRowSum, 0, (K) * sizeof(int));

}
