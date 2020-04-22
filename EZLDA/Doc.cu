#include "Doc.cuh"

Document::Document(string argFilePrefix, int argNumChunks, int argMaxTLLength, int argmaxDocLength, int argWordLength) {

	filePrefix = argFilePrefix;
	numChunks = argNumChunks;
	maxTLLength = argMaxTLLength;
	maxDocLength = argmaxDocLength;
	wordLength = argWordLength;

	chunksPerStream = numChunks / numStreams;




	//perplexityMid = new float[GridDim];
	cudaMallocHost((void**)&perplexityMid, GridDim * sizeof(float));

	/*perplexity = new float[maxTLLength];*/

	cudaMallocHost((void**)&perplexity, maxTLLength * sizeof(float));

	//perplexityAve = new float[1];

	cudaMallocHost((void**)&perplexityAve, 1 * sizeof(float));

	//effectiveTokenIndex = new int[maxTLLength];

	cudaMallocHost((void**)&effectiveTokenIndex, maxTLLength * sizeof(int));

	//newTokenCount = new int[wordLength];

	cudaMallocHost((void**)&newTokenCount, wordLength * sizeof(int));

	//maxTokenCount = new unsigned short int[maxTLLength];

	cudaMallocHost((void**)&maxTokenCount, maxTLLength * sizeof(unsigned short int));

	//Mflag = new unsigned short int[maxTLLength];

	cudaMallocHost((void**)&Mflag, maxTLLength * sizeof(unsigned short int));
	
}

void Document::loadDocument() {


	/*TLLengthVec = new int[numChunks];
	docLengthVec = new int[numChunks];
	numOfTokenVecD = new int[numChunks];
	numOfTokenVecS = new int[numChunks];
	timeRecord = new float[GridDim*BlockDim/32];*/
	cudaMallocHost((void**)&TLLengthVec, numChunks * sizeof(int));
	cudaMallocHost((void**)&docLengthVec, numChunks * sizeof(int));
	cudaMallocHost((void**)&numOfTokenVecD, numChunks * sizeof(int));
	cudaMallocHost((void**)&numOfTokenVecS, numChunks * sizeof(int));
	cudaMallocHost((void**)&timeRecord, GridDim*BlockDim / 32 * sizeof(float));
	

	ifstream docLength((filePrefix + string("/docLength.txt")).c_str(), ios::binary);//store max Doc and DT length
	ifstream TLLength((filePrefix + string("/TLLength.txt")).c_str(), ios::binary);
	ifstream TLSplit((filePrefix + string("/TLSplit.txt")).c_str(), ios::binary);

	for (int chunkId = 0; chunkId < numChunks; chunkId++) {

		TLLength >> TLLengthVec[chunkId];
		docLength >> docLengthVec[chunkId];
		TLSplit >> numOfTokenVecD[chunkId] >> numOfTokenVecS[chunkId];
		totalNumOfTokens += TLLengthVec[chunkId];


		DocChunk tmpDocChunk(TLLengthVec[chunkId], docLengthVec[chunkId], wordLength);
		tmpDocChunk.CPUMemSet();
		tmpDocChunk.loadChunk(filePrefix, chunkId);
		docChunkVec.push_back(tmpDocChunk);


		//float* tmpProbMaxChunk = new float[TLLengthVec[chunkId]];
		//memset(tmpProbMaxChunk, 0, TLLengthVec[chunkId] * sizeof(float));
		//unsigned short int* tmpProbMaxTopicChunk = new unsigned short int[TLLengthVec[chunkId]];
		//memset(tmpProbMaxTopicChunk, 0, TLLengthVec[chunkId] * sizeof(unsigned short int));

		//unsigned short int* tmpProbMaxFlagChunk = new unsigned short int[TLLengthVec[chunkId]];
		//memset(tmpProbMaxFlagChunk, 0, TLLengthVec[chunkId] * sizeof(unsigned short int));

		//unsigned short int* tmpProbMaxTopicFlagChunk = new unsigned short int[TLLengthVec[chunkId]];
		//memset(tmpProbMaxTopicFlagChunk, 0, TLLengthVec[chunkId] * sizeof(unsigned short int));

		unsigned short int* tmpMaxTokenCount = new unsigned short int[TLLengthVec[chunkId]];
		memset(tmpMaxTokenCount, 0, TLLengthVec[chunkId] * sizeof(unsigned short int));

		//probMaxChunkVec.push_back(tmpProbMaxChunk);
		//probMaxTopicChunkVec.push_back(tmpProbMaxTopicChunk);
		//probMaxFlagChunkVec.push_back(tmpProbMaxFlagChunk);
		//probMaxTopicFlagChunkVec.push_back(tmpProbMaxTopicFlagChunk);
		maxTokenCountVec.push_back(tmpMaxTokenCount);

	}
	memset(effectiveTokenIndex, 0, maxTLLength * sizeof(int));
	memset(newTokenCount, 0, wordLength * sizeof(int));
	memset(maxTokenCount, 0, maxTLLength * sizeof(unsigned short int));
	memset(Mflag, 0, maxTLLength * sizeof(unsigned short int));
	printf("total num of tokens:%f\n", totalNumOfTokens);
	printf("All chunks loaded!");
	docLength.close();
	TLLength.close();

}

void Document::CPU2GPUPerplexity(cudaStream_t& stream) {

	
	//memset(perplexityMid, 0, GridDim * sizeof(float));
	for (int i = 0; i < numStreams; i++) {
		cudaMemsetAsync(devicePerplexityMid[i], 0, GridDim * sizeof(float), stream);
		//cudaMemcpyAsync(devicePerplexityMid[i], perplexityMid, GridDim * sizeof(float), cudaMemcpyHostToDevice, stream);
	}
	
 //   /*cudaMemset(devicePerplexity,0,maxTLLength*sizeof(float));*/

	/*cudaMemsetAsync(devicePerplexityMid, 0, GridDim * sizeof(float), stream);*/

}


void Document::GPU2CPUPerplexity(cudaStream_t& stream) {

	cudaMemcpyAsync(perplexityMid, devicePerplexityMid, (GridDim) * sizeof(float), cudaMemcpyDeviceToHost, stream);

	//cudaMemcpy(perplexity, devicePerplexity, maxTLLength*sizeof(float), cudaMemcpyDeviceToHost);
	sumPerplexity = 0.0;

	for (int i = 0; i < GridDim; i++) {
		//printf("Perplexity:%f \n", perplexityMid[i]);
		sumPerplexity += perplexityMid[i]/ 467723.0;
	}

	//printf("Parallel Perplexity:%f \n", sumPerplexity);

}

void Document::CPU2DiskPerplexity(string argFilePrefix) {

	ofstream OutPutPerplexity((argFilePrefix + string("/Perplexity.txt")).c_str(), ios::binary);
	for (int i = 0; i < maxTLLength; i++) {
		OutPutPerplexity << perplexity[i] << "\n";
	}
	OutPutPerplexity.close();


}


void Document::GPU2CPUEffectiveTokenIndex() {

	cudaMemcpy(effectiveTokenIndex, deviceEffectiveTokenIndex, maxTLLength * sizeof(int), cudaMemcpyDeviceToHost);

	cudaMemcpy(newTokenCount, deviceNewTokenCount, wordLength * sizeof(int), cudaMemcpyDeviceToHost);
}


void Document::CPU2DiskEffectiveTokenIndex(string argFilePrefix) {

	ofstream OutPutEffectiveTokenIndex((argFilePrefix + string("/EffectiveTokenIndex.txt")).c_str(), ios::binary);
	for (int i = 0; i < maxTLLength; i++) {
		OutPutEffectiveTokenIndex << effectiveTokenIndex[i] << "\n";
	}
	OutPutEffectiveTokenIndex.close();

	ofstream OutPutNewTokenCount((argFilePrefix + string("/NewTokenCount.txt")).c_str(), ios::binary);
	for (int i = 0; i < wordLength; i++) {
		OutPutNewTokenCount << newTokenCount[i] << "\n";
	}
	OutPutNewTokenCount.close();


}

void Document::GPUMemAllocate() {

	for (int i = 0; i < numStreams; i++) {
		cudaMalloc((void**)&deviceTLTopic[i], (maxTLLength) * sizeof(unsigned short int));
		cudaMalloc((void**)&deviceTLDocCount[i], (maxDocLength) * sizeof(int));
		cudaMalloc((void**)&deviceTLDocOffset[i], (maxDocLength) * sizeof(int));
		cudaMalloc((void**)&deviceTLWordCount[i], (wordLength) * sizeof(int));
		cudaMalloc((void**)&deviceTLWordOffset[i], (wordLength) * sizeof(int));
		cudaMalloc((void**)&deviceMapWord2Doc[i], (maxTLLength) * sizeof(int));
		cudaMalloc((void**)&deviceMapDoc2Word[i], (maxTLLength) * sizeof(int));
		cudaMalloc((void**)&deviceRandomfloat[i], (maxTLLength) * sizeof(float));
		/*cudaMalloc((void**)&deviceMflag[i], (maxTLLength) * sizeof(unsigned short int));*/

		cudaMalloc((void**)&deviceEffectiveTokenIndex[i], (maxTLLength) * sizeof(int));
		cudaMalloc((void**)&deviceNewTokenCount[i], (wordLength) * sizeof(int));

		/*cudaMalloc((void**)&devicePerplexity[i], (maxTLLength) * sizeof(float));*/
		
	
		cudaMalloc((void **)&d_blockCounter[i], sizeof(int)*(1));
		cudaMalloc((void **)&d_warpCounter[i], sizeof(int)*(1));
		cudaMalloc((void **)&d_dense[i], sizeof(int)*(GridDim*BlockDim*K/32));
		cudaMalloc((void **)&deviceWTHeadDense[i], sizeof(float)*(GridDim*K));

	/*
		cudaMalloc((void**)&deviceProbMax, (maxTLLength) * sizeof(float));
		cudaMalloc((void**)&deviceProbMaxTopic, (maxTLLength) * sizeof(unsigned short int));
		cudaMalloc((void**)&deviceProbMaxFlag, (maxTLLength) * sizeof(unsigned short int));
		cudaMalloc((void**)&deviceProbMaxTopicFlag, (maxTLLength) * sizeof(unsigned short int));*/

		/*cudaMalloc((void**)&deviceMaxTokenCount[i], (maxTLLength) * sizeof(unsigned short int));
		cudaMalloc((void**)&deviceMaxTopic[i], (maxTLLength) * sizeof(unsigned short int));

		cudaMalloc((void**)&deviceSecondMaxTokenCount[i], (maxTLLength) * sizeof(unsigned short int));
		cudaMalloc((void**)&deviceSecondMaxTopic[i], (maxTLLength) * sizeof(unsigned short int));*/

		cudaMalloc((void**)&deviceMaxSecTopic[i], (maxTLLength) * sizeof(long long int));

		cudaMalloc((void**)&deviceWordMaxTopic[i], (wordLength) * sizeof(unsigned short int));
		cudaMalloc((void**)&deviceWordSecondMaxTopic[i], (wordLength) * sizeof(unsigned short int));
		cudaMalloc((void**)&deviceWordThirdMaxTopic[i], (wordLength) * sizeof(unsigned short int));

		cudaMalloc((void**)&deviceWordMaxProb[i], (wordLength) * sizeof(float));
		cudaMalloc((void**)&deviceWordSecondMaxProb[i], (wordLength) * sizeof(float));
		cudaMalloc((void**)&deviceWordThirdMaxProb[i], (wordLength) * sizeof(float));

		cudaMalloc((void**)&deviceQArray[i], (wordLength) * sizeof(float));

		cudaMalloc((void**)&deviceMaxProb[i], (maxTLLength) * sizeof(float));
		cudaMalloc((void**)&deviceThresProb[i], (maxTLLength) * sizeof(float));
		cudaMalloc((void**)&deviceTimeRecord[i], (GridDim*BlockDim/32) * sizeof(float));

		cudaMalloc((void**)&devicePerplexityAve[i], 1 * sizeof(float));
		cudaMalloc((void**)&devicePerplexityMid[i], sizeof(float)*GridDim);
		//cudaMalloc((void**)&deviceTotalTokenCount[i], (maxTLLength) * sizeof(unsigned short int));


	}
	
	


	TLMemory = (2*(6 * maxTLLength + 2 * maxDocLength + 7 * wordLength) * sizeof(int))/ 1000000000.0;

	printf("Token list memory usage:%f GB\n", TLMemory);


}

void Document::GPU2CPUTime() {

	cudaMemcpy(timeRecord, deviceTimeRecord, (GridDim*BlockDim / 32) * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemset(deviceTimeRecord, 0, (GridDim*BlockDim / 32) * sizeof(float));

}

//void Document::CPU2DiskTime(ofstream argOutPutTime) {
//
//	for (int i = 0; i < GridDim*BlockDim / 32; i++) {
//		argOutPutTime << timeRecord[i] << " ";
//	}
//	argOutPutTime << "\n";
//}




void Document::CPU2GPU(int argChunkId, int argStreamId, cudaStream_t& stream) {

	cudaMemcpyAsync(deviceTLTopic[argStreamId], docChunkVec[argChunkId].TLTopic, (TLLengthVec[argChunkId]) * sizeof(unsigned short int), cudaMemcpyHostToDevice, stream);
	cudaMemcpyAsync(deviceTLDocCount[argStreamId], docChunkVec[argChunkId].TLDocCount, (docLengthVec[argChunkId]) * sizeof(int), cudaMemcpyHostToDevice, stream);
	cudaMemcpyAsync(deviceTLDocOffset[argStreamId], docChunkVec[argChunkId].TLDocOffset, (docLengthVec[argChunkId]) * sizeof(int), cudaMemcpyHostToDevice, stream);
	cudaMemcpyAsync(deviceTLWordCount[argStreamId], docChunkVec[argChunkId].TLWordCount, (wordLength) * sizeof(int), cudaMemcpyHostToDevice, stream);
	cudaMemcpyAsync(deviceTLWordOffset[argStreamId], docChunkVec[argChunkId].TLWordOffset, (wordLength) * sizeof(int), cudaMemcpyHostToDevice, stream);
	cudaMemcpyAsync(deviceMapWord2Doc[argStreamId], docChunkVec[argChunkId].mapWord2Doc, (TLLengthVec[argChunkId]) * sizeof(int), cudaMemcpyHostToDevice, stream);
	//cudaMemcpyAsync(deviceMapDoc2Word[argStreamId], docChunkVec[argChunkId].mapDoc2Word, (TLLengthVec[argChunkId]) * sizeof(int), cudaMemcpyHostToDevice, stream);
	//cudaMemcpyAsync(deviceTotalTokenCount[argStreamId], docChunkVec[argChunkId].totalTokenCount, (TLLengthVec[argChunkId]) * sizeof(unsigned short int), cudaMemcpyHostToDevice, stream);

	//cudaMemcpy(deviceProbMax, probMaxChunkVec[argChunkId], (TLLengthVec[argChunkId]) * sizeof(float), cudaMemcpyHostToDevice);
	//cudaMemcpy(deviceProbMaxTopic, probMaxTopicChunkVec[argChunkId], (TLLengthVec[argChunkId]) * sizeof(unsigned short int), cudaMemcpyHostToDevice);

	//cudaMemcpy(deviceProbMaxTopicFlag, probMaxTopicFlagChunkVec[argChunkId], (TLLengthVec[argChunkId]) * sizeof(unsigned short int), cudaMemcpyHostToDevice);
	//cudaMemcpy(deviceProbMaxFlag, probMaxFlagChunkVec[argChunkId], (TLLengthVec[argChunkId]) * sizeof(unsigned short int), cudaMemcpyHostToDevice);

	/*cudaMemcpy(deviceMaxTopic, docChunkVec[argChunkId].TLMaxTopic, (TLLengthVec[argChunkId]) * sizeof(unsigned short int), cudaMemcpyHostToDevice);*/

	//cudaMemset(deviceProbMaxTopicFlag, 0, (maxTLLength) * sizeof(unsigned short int));
	//cudaMemset(deviceProbMaxFlag, 0, (maxTLLength) * sizeof(unsigned short int));
	/*cudaMemsetAsync(deviceMaxTokenCount[argStreamId], 0, (maxTLLength) * sizeof(unsigned short int), stream);*/
	cudaMemsetAsync(deviceMaxSecTopic[argStreamId], 0, (maxTLLength) * sizeof(long long int), stream);
	/*cudaMemsetAsync(deviceMflag[argStreamId], 0, (maxTLLength) * sizeof(unsigned short int), stream);*/


	cudaMemsetAsync(deviceEffectiveTokenIndex[argStreamId], 0, (maxTLLength) * sizeof(int), stream);
	cudaMemsetAsync(deviceNewTokenCount[argStreamId], 0, (wordLength) * sizeof(int), stream);
	//cudaMemsetAsync(deviceTotalTokenCount[argStreamId], 0, (maxTLLength) * sizeof(unsigned short int), stream);

	/*cudaMemcpyAsync(deviceMflag[argStreamId], Mflag, (maxTLLength) * sizeof(unsigned short int), cudaMemcpyHostToDevice, stream);*/

	/*cudaMemcpyAsync(deviceMaxTokenCount[argStreamId], maxTokenCount, (maxTLLength) * sizeof(unsigned short int), cudaMemcpyHostToDevice, stream);
	cudaMemcpyAsync(deviceEffectiveTokenIndex[argStreamId], effectiveTokenIndex, (maxTLLength) * sizeof(int), cudaMemcpyHostToDevice, stream);
	cudaMemcpyAsync(deviceNewTokenCount[argStreamId], newTokenCount, (wordLength) * sizeof(int), cudaMemcpyHostToDevice, stream);*/



}

void Document::GPU2CPU(int argChunkId, int argStreamId, cudaStream_t& stream) {

	cudaMemcpyAsync(docChunkVec[argChunkId].TLTopic, deviceTLTopic[argStreamId], (TLLengthVec[argChunkId]) * sizeof(unsigned short int), cudaMemcpyDeviceToHost, stream);
	/*cudaMemcpy(probMaxTopicFlagChunkVec[argChunkId],deviceProbMaxTopicFlag,  (TLLengthVec[argChunkId]) * sizeof(unsigned short int), cudaMemcpyDeviceToHost);
	cudaMemcpy( probMaxFlagChunkVec[argChunkId],deviceProbMaxFlag, (TLLengthVec[argChunkId]) * sizeof(unsigned short int), cudaMemcpyDeviceToHost);
	cudaMemcpy(probMaxChunkVec[argChunkId],deviceProbMax,  (TLLengthVec[argChunkId]) * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(probMaxTopicChunkVec[argChunkId], deviceProbMaxTopic, (TLLengthVec[argChunkId]) * sizeof(unsigned short int), cudaMemcpyDeviceToHost);*/
	//cudaMemcpyAsync(docChunkVec[argChunkId].TLMaxTopic, deviceMaxTopic[argStreamId], (TLLengthVec[argChunkId]) * sizeof(unsigned short int), cudaMemcpyDeviceToHost, stream);





}
//
//void Document::PercentageCalculate()
//{
//	increasePercent = 0.0;
//	topicUnchangedPercent = 0.0;
//	for (int chunkId = 0; chunkId < numChunks; chunkId++) {
//		for (int i = 0; i < TLLengthVec[chunkId]; i++) {
//			increasePercent += float(probMaxFlagChunkVec[chunkId][i]);
//			topicUnchangedPercent += float(probMaxTopicFlagChunkVec[chunkId][i]);
//		}		
//	}
//	printf("increasePercent:%f\n", increasePercent);
//	printf("topicUnchangedPercent:%f\n", topicUnchangedPercent);
//	printf("total num of tokens:%f\n", totalNumOfTokens);
//	increasePercent /= totalNumOfTokens;
//	topicUnchangedPercent /= totalNumOfTokens;
//	
//
//}
//
//
void Document::deviceCounterMemAllocate() {
	for (int i = 0; i < numStreams; i++) {
		cudaMalloc((void**)&deviceCounterWTUpdateKernel[i], sizeof(unsigned int));
		cudaMalloc((void**)&deviceCounterWTDenUpdateKernel[i], sizeof(unsigned int));
		cudaMalloc((void**)&deviceCounterWTAdditionKernel[i], sizeof(unsigned int));
		cudaMalloc((void**)&deviceCounterMaxTopicKernel[i], sizeof(unsigned int));
		cudaMalloc((void**)&deviceCounterDTUpdateKernel[i], sizeof(unsigned int));
		cudaMalloc((void**)&deviceCounterUpdateProbKernel[i], sizeof(unsigned int));
		cudaMalloc((void**)&deviceCounterSampleKernelD[i], sizeof(unsigned int));
		cudaMalloc((void**)&deviceCounterSampleKernelS[i], sizeof(unsigned int));
	}
	



}
