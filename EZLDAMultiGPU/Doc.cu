#include "Doc.cuh"

Document::Document(string argFilePrefix, int argNumChunks, int argMaxTLLength, int argmaxDocLength, int argWordLength, int argNumGPUs) {

	filePrefix = argFilePrefix;
	numChunks = argNumChunks;
	maxTLLength = argMaxTLLength;
	maxDocLength = argmaxDocLength;
	wordLength = argWordLength;
	numGPUs = argNumGPUs;
	perplexityMid = new float[GridDim*BlockDim / 32];
	//perplexityMid2 = new float[GridDim*BlockDim / 32];
	perplexity = new float[maxTLLength];

	
}

void Document::loadDocument() {


	TLLengthVec = new int[numChunks];
	docLengthVec = new int[numChunks];
	numOfTokenVecD = new int[numChunks];
	numOfTokenVecS = new int[numChunks];

	ifstream docLength((filePrefix + string("/docLength.txt")).c_str(), ios::binary);//store max Doc and DT length
	ifstream TLLength((filePrefix + string("/TLLength.txt")).c_str(), ios::binary);
	ifstream TLSplit((filePrefix + string("/TLSplit.txt")).c_str(), ios::binary);

	for (int chunkId = 0; chunkId < numChunks; chunkId++) {

		TLLength >> TLLengthVec[chunkId];
		docLength >> docLengthVec[chunkId];
		TLSplit >> numOfTokenVecD[chunkId] >> numOfTokenVecS[chunkId];
		DocChunk tmpDocChunk(TLLengthVec[chunkId], docLengthVec[chunkId], wordLength);
		tmpDocChunk.CPUMemSet();
		tmpDocChunk.loadChunk(filePrefix, chunkId);
		docChunkVec.push_back(tmpDocChunk);

	}
	printf("All chunks loaded!");
	docLength.close();
	TLLength.close();

}


void Document::InitGPU()
{
	for (int GPUId = 0; GPUId < numGPUs; GPUId++) {

		GPUChunk ChunkGPU(maxTLLength, maxDocLength, wordLength);
		ChunkGPU.GPUMemAllocate(GPUId);
		GPUChunkVec.push_back(ChunkGPU);
	}

}








void Document::CPU2GPUPerplexity(int argGPUId) {

	cudaSetDevice(argGPUId);
	memset(perplexityMid, 0, GridDim*BlockDim / 32 * sizeof(float));

	//memset(perplexityMid2, 0, GridDim*BlockDim / 32 * sizeof(float));

	cudaMemcpy(GPUChunkVec[argGPUId].devicePerplexityMid, perplexityMid, (GridDim*BlockDim / 32) * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(GPUChunkVec[argGPUId].devicePerplexity,0,maxTLLength*sizeof(float));

}


void Document::GPU2CPUPerplexity(int argGPUId) {
	//cudaSetDevice(argGPUId);
	//if (argGPUId == 0) {
	//	cudaMemcpy(perplexityMid, GPUChunkVec[argGPUId].devicePerplexityMid, (GridDim*BlockDim / 32) * sizeof(float), cudaMemcpyDeviceToHost);
	//}
	//
	//else {
	//	cudaMemcpy(perplexityMid2, GPUChunkVec[argGPUId].devicePerplexityMid, (GridDim*BlockDim / 32) * sizeof(float), cudaMemcpyDeviceToHost);
	//}
	//

	///*cudaMemcpy(perplexity, GPUChunkVec[argGPUId].devicePerplexity, maxTLLength*sizeof(float), cudaMemcpyDeviceToHost);*/
	//sumPerplexity = 0.0;
	//if (argGPUId == 0) {
	//	for (int i = 0; i < GridDim*BlockDim / 32; i++) {
	//		// printf("Perplexity:%f \n", h_PerplexityMid[i]);
	//		sumPerplexity += perplexityMid[i] / 467723.0;
	//	}
	//}

	//else {
	//	for (int i = 0; i < GridDim*BlockDim / 32; i++) {
	//		// printf("Perplexity:%f \n", h_PerplexityMid[i]);
	//		sumPerplexity += perplexityMid2[i] / 467723.0;
	//	}
	//}
	//

	//printf("Parallel Perplexity:%f \n", sumPerplexity);




	cudaSetDevice(argGPUId);
	
	cudaMemcpy(perplexityMid, GPUChunkVec[argGPUId].devicePerplexityMid, (GridDim*BlockDim / 32) * sizeof(float), cudaMemcpyDeviceToHost);
	

	//else {
	//	cudaMemcpy(perplexityMid2, GPUChunkVec[argGPUId].devicePerplexityMid, (GridDim*BlockDim / 32) * sizeof(float), cudaMemcpyDeviceToHost);
	//}


	/*cudaMemcpy(perplexity, GPUChunkVec[argGPUId].devicePerplexity, maxTLLength*sizeof(float), cudaMemcpyDeviceToHost);*/
	sumPerplexity = 0.0;
	for (int i = 0; i < GridDim*BlockDim / 32; i++) {
			// printf("Perplexity:%f \n", h_PerplexityMid[i]);
			sumPerplexity += perplexityMid[i] / 467723.0;
	}





	printf("Parallel Perplexity:%f \n", sumPerplexity);




}

void Document::CPU2DiskPerplexity(string argFilePrefix) {

	ofstream OutPutPerplexity((argFilePrefix + string("/Perplexity.txt")).c_str(), ios::binary);
	for (int i = 0; i < maxTLLength; i++) {
		OutPutPerplexity << perplexity[i] << "\n";
	}
	OutPutPerplexity.close();
}

//void Document::GPUMemAllocate(int argGPUId) {
//	GPUId = argGPUId;
//	cudaSetDevice(GPUId);
//	cudaMalloc((void**)&deviceTLTopic, (maxTLLength) * sizeof(int));
//	cudaMalloc((void**)&deviceTLDocCount, (maxDocLength) * sizeof(int));
//	cudaMalloc((void**)&deviceTLDocOffset, (maxDocLength) * sizeof(int));
//	cudaMalloc((void**)&deviceTLWordCount, (wordLength) * sizeof(int));
//	cudaMalloc((void**)&deviceTLWordOffset, (wordLength) * sizeof(int));
//	cudaMalloc((void**)&deviceMapWord2Doc, (maxTLLength) * sizeof(int));
//	cudaMalloc((void**)&deviceMapDoc2Word, (maxTLLength) * sizeof(int));
//	cudaMalloc((void**)&devicePerplexity, (maxTLLength) * sizeof(float));
//	cudaMalloc((void**)&devicePerplexityMid, sizeof(float)*(GridDim*BlockDim / 32));
//	
//	cudaMalloc((void **)&d_blockCounter, sizeof(int)*(1));
//	cudaMalloc((void **)&d_warpCounter, sizeof(int)*(1));
//	cudaMalloc((void **)&d_dense, sizeof(int)*(GridDim*K));
//	cudaMalloc((void **)&deviceWTHeadDense, sizeof(float)*(GridDim*K));
//
//
//	TLMemory = ((3 * maxTLLength + 2 * maxDocLength + 2 * wordLength + GridDim*K) * sizeof(int) + (maxTLLength + GridDim*BlockDim / 32 + GridDim*K) * sizeof(float))/ 1000000000.0;
//
//	printf("Token list memory usage:%f GB\n", TLMemory);
//
//
//}


void Document::CPU2GPU(int argGPUId, int argChunkId) {

	cudaSetDevice(argGPUId);
	

	cudaMemcpy(GPUChunkVec[argGPUId].deviceTLTopic, docChunkVec[argChunkId].TLTopic, (TLLengthVec[argChunkId]) * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(GPUChunkVec[argGPUId].deviceTLDocCount, docChunkVec[argChunkId].TLDocCount, (docLengthVec[argChunkId]) * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(GPUChunkVec[argGPUId].deviceTLDocOffset, docChunkVec[argChunkId].TLDocOffset, (docLengthVec[argChunkId]) * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(GPUChunkVec[argGPUId].deviceTLWordCount, docChunkVec[argChunkId].TLWordCount, (wordLength) * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(GPUChunkVec[argGPUId].deviceTLWordOffset, docChunkVec[argChunkId].TLWordOffset, (wordLength) * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(GPUChunkVec[argGPUId].deviceMapWord2Doc, docChunkVec[argChunkId].mapWord2Doc, (TLLengthVec[argChunkId]) * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(GPUChunkVec[argGPUId].deviceMapDoc2Word, docChunkVec[argChunkId].mapDoc2Word, (TLLengthVec[argChunkId]) * sizeof(int), cudaMemcpyHostToDevice);


}

void Document::GPU2CPU(int argGPUId, int argChunkId) {
	cudaSetDevice(argGPUId);
	cudaMemcpy(docChunkVec[argChunkId].TLTopic, GPUChunkVec[argGPUId].deviceTLTopic, (TLLengthVec[argChunkId]) * sizeof(int), cudaMemcpyDeviceToHost);


}
