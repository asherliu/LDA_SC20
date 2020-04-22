#include "DataGPUChunk.cuh"

GPUChunk::GPUChunk(int argMaxTLLength, int argmaxDocLength, int argWordLength) {


	maxTLLength = argMaxTLLength;
	maxDocLength = argmaxDocLength;
	wordLength = argWordLength;


}






void GPUChunk::GPUMemAllocate(int argGPUId) {
	GPUId = argGPUId;
	cudaSetDevice(GPUId);
	cudaMalloc((void**)&deviceTLTopic, (maxTLLength) * sizeof(int));
	cudaMalloc((void**)&deviceTLDocCount, (maxDocLength) * sizeof(int));
	cudaMalloc((void**)&deviceTLDocOffset, (maxDocLength) * sizeof(int));
	cudaMalloc((void**)&deviceTLWordCount, (wordLength) * sizeof(int));
	cudaMalloc((void**)&deviceTLWordOffset, (wordLength) * sizeof(int));
	cudaMalloc((void**)&deviceMapWord2Doc, (maxTLLength) * sizeof(int));
	cudaMalloc((void**)&deviceMapDoc2Word, (maxTLLength) * sizeof(int));
	cudaMalloc((void**)&devicePerplexity, (maxTLLength) * sizeof(float));
	cudaMalloc((void**)&devicePerplexityMid, sizeof(float)*(GridDim*BlockDim / 32));

	cudaMalloc((void **)&d_dense, sizeof(int)*(GridDim*K));
	cudaMalloc((void **)&deviceWTHeadDense, sizeof(float)*(GridDim*K));

}
