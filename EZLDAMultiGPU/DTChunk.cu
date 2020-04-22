
#include "DTChunk.cuh"
DTGPUChunk::DTGPUChunk(int argmaxDTLength, int argMaxDocLength, int argGPUId) {

	maxDTLength = argmaxDTLength;
	maxDocLength = argMaxDocLength;
	GPUId = argGPUId;
	//NZDTCount = new int[maxDocLength];
	//DTIndex = new int[maxDTLength];
	//DTValue = new int[maxDTLength];
	////DTCount = new int[maxDocLength];
	////DTOffset = new int[maxDocLength];
	//DTLengthVec = new int[numChunks];
	//docLengthVec = new int[numChunks];
}




void DTGPUChunk::GPUMemAllocate(int argGPUId) {

	GPUId = argGPUId;
	cudaSetDevice(argGPUId);
	cudaMalloc((void**)&deviceNZDTCount, (maxDocLength) * sizeof(int));
	cudaMalloc((void**)&deviceDTIndex, (maxDTLength) * sizeof(int));
	cudaMalloc((void**)&deviceDTValue, (maxDTLength) * sizeof(int));
	cudaMalloc((void**)&deviceDTCount, (maxDocLength) * sizeof(int));
	cudaMalloc((void**)&deviceDTOffset, (maxDocLength) * sizeof(int));

	//DTMemory = (3 * maxDocLength + 2 * maxDTLength) * sizeof(int) / 1000000000.0;
	//printf("DT memory usage:%f GB\n", DTMemory);

}

void DTGPUChunk::GPUMemSet(int argGPUId)
{
	GPUId = argGPUId;
	cudaSetDevice(argGPUId);
	cudaMemset(deviceNZDTCount, 0, (maxDocLength) * sizeof(int));
	cudaMemset(deviceDTIndex, 0, (maxDTLength) * sizeof(int));
	cudaMemset(deviceDTValue, 0, (maxDTLength) * sizeof(int));

}
