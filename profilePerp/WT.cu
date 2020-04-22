
#include "WT.cuh"
WTAll::WTAll(int argmaxWTLength, int argWordLength, int argNumChunks, int argMaxChunkWTLength, int argNumOfWordS) {
	maxWTLength = argmaxWTLength;
	wordLength = argWordLength;
	numChunks = argNumChunks;
	maxChunkWTLength = argMaxChunkWTLength;
	numOfWordS = argNumOfWordS;
	WTLengthVec = new int[numChunks];
	WTRowSum = new int[K];

	NZWTCount = new int[numOfWordS];
	WTIndex = new unsigned short int[maxWTLength];
	WTValue = new unsigned short int[maxWTLength];
	WTCount = new int[wordLength];
	WTOffset = new int[wordLength];
	


	////-----chunkWT-----for test--------
	//chunkNZWTCount = new int[wordLength];
	//chunkWTIndex = new int[maxChunkWTLength];
	//chunkWTValue = new int[maxChunkWTLength];
	////-----chunkWT-----for test--------


}




void WTAll::CPUMemSet() {

	memset(NZWTCount, 0, numOfWordS * sizeof(int));
	memset(WTIndex, 0, maxWTLength * sizeof(unsigned short int));
	memset(WTValue, 0, maxWTLength * sizeof(unsigned short int));
	memset(WTCount, 0, wordLength * sizeof(int));
	memset(WTOffset, 0, wordLength * sizeof(int));
	memset(WTRowSum, 0, K * sizeof(int));

}
void WTAll::GPUMemAllocate() {

	cudaMalloc((void**)&deviceNZWTCount, (numOfWordS) * sizeof(int));
	cudaMalloc((void**)&deviceWTIndex, (maxWTLength) * sizeof(unsigned short int));
	cudaMalloc((void**)&deviceWTValue, (maxWTLength) * sizeof(unsigned short int));
	cudaMalloc((void**)&deviceWTCount, (wordLength) * sizeof(int));
	cudaMalloc((void**)&deviceWTOffset, (wordLength) * sizeof(int));

	cudaMalloc((void**)&deviceWTRowSum, (K) * sizeof(int));
	cudaMalloc((void**)&deviceBlockCount, (1) * sizeof(int));
	cudaMalloc((void**)&deviceWarpCount, (1) * sizeof(int));

	cudaMalloc((void**)&deviceChunkWTCount, (numOfWordS) * sizeof(int));
	cudaMalloc((void**)&deviceChunkWTOffset, (numOfWordS) * sizeof(int));
	cudaMalloc((void**)&deviceChunkNZWTCount, (numOfWordS) * sizeof(int));
	cudaMalloc((void**)&deviceChunkWTIndex, (maxChunkWTLength) * sizeof(unsigned short int));
	cudaMalloc((void**)&deviceChunkWTValue, (maxChunkWTLength) * sizeof(unsigned short int));


	WTMemory = (6 * wordLength + 2 * maxWTLength + K + 2 * maxChunkWTLength) /1000000000.0 * sizeof(int);
	printf("WT memory usage(Sparse):%f GB\n", WTMemory);
	WTMemory = K /1000000000.0 * wordLength * sizeof(int);
	printf("WT memory usage(Dense):%f GB\n", WTMemory);

}
void WTAll::GPUMemset()
{
	cudaMemset(deviceNZWTCount, 0, (numOfWordS) * sizeof(int));
	cudaMemset(deviceWTIndex, 0, (maxWTLength) * sizeof(unsigned short int));
	cudaMemset(deviceWTValue, 0, (maxWTLength) * sizeof(unsigned short int));
	cudaMemset(deviceWTRowSum, 0, (K) * sizeof(int));

}

void WTAll::chunkGPUMemset()
{
	cudaMemset(deviceChunkNZWTCount, 0, (numOfWordS) * sizeof(int));
	cudaMemset(deviceChunkWTIndex, 0, (maxChunkWTLength) * sizeof(unsigned short int));
	cudaMemset(deviceChunkWTValue, 0, (maxChunkWTLength) * sizeof(unsigned short int));
	//cudaMemset(deviceWTRowSum, 0, (K) * sizeof(int));

}


void WTAll::loadWTLength(string argFilePrefix) {

	ifstream WTLength((argFilePrefix + string("/WTLength.txt")).c_str(), ios::binary);//store max Doc and DT length	
	for (int chunkId = 0; chunkId < numChunks; chunkId++) {
		WTLength >> WTLengthVec[chunkId];

	}
	WTLength.close();
}

void WTAll::loadWTCountOffset(string argFilePrefix) {

	//--------load chunkWTCountOffset--------------
	for (int chunkId = 0; chunkId < numChunks; chunkId++) {

		WTChunkData chunkWTData(chunkId, wordLength, maxChunkWTLength, WTLengthVec[chunkId], numOfWordS);
		chunkWTData.CPUMemSet();
		chunkWTData.loadWTCountOffset(argFilePrefix);
		WTChunkVec.push_back(chunkWTData);
	}
	//--------load chunkWTCountOffset--------------



	//--------load WTCountOffset--------------

	ifstream WTCountOffset((argFilePrefix + string("/WTCountOffset.txt")).c_str(), ios::binary);//store Word offset of TL
	blockCount = 0;
	for (int i = 0; i < wordLength; i++)
	{
		WTCountOffset >> WTCount[i] >> WTOffset[i];

		if (i >= wordLength - numOfWordS) {
			if (WTCount[i] > 32) {
				blockCount++;
			}
		}
		
	}
	WTCountOffset.close();
	warpCount = numOfWordS - blockCount;
	printf("WT Count and Offset loaded!...\n");

	//--------load WTCountOffset--------------

}


void WTAll::blockWarpCountCPU2GPU() {

	cudaMemcpy(deviceBlockCount, &blockCount, (1) * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(deviceWarpCount, &warpCount, (1) * sizeof(int), cudaMemcpyHostToDevice);

}

void WTAll::CPU2GPUCountOffset() {

	cudaMemcpy(deviceWTCount, WTCount, (wordLength) * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(deviceWTOffset, WTOffset, (wordLength) * sizeof(int), cudaMemcpyHostToDevice);

}

void WTAll::WTCPU2GPU() {

	cudaMemcpy(deviceNZWTCount, NZWTCount, (numOfWordS) * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(deviceWTIndex, WTIndex, (maxWTLength) * sizeof(unsigned short int), cudaMemcpyHostToDevice);
	cudaMemcpy(deviceWTValue, WTValue, (maxWTLength) * sizeof(unsigned short int), cudaMemcpyHostToDevice);
	
}

void WTAll::WTGPU2CPU() {

	cudaMemcpy(NZWTCount, deviceNZWTCount, (numOfWordS) * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(WTIndex, deviceWTIndex, (maxWTLength) * sizeof(unsigned short int), cudaMemcpyDeviceToHost);
	cudaMemcpy(WTValue, deviceWTValue, (maxWTLength) * sizeof(unsigned short int), cudaMemcpyDeviceToHost);
	cudaMemcpy(WTRowSum, deviceWTRowSum, (K) * sizeof(int), cudaMemcpyDeviceToHost);
}


void WTAll::chunkCPU2GPUCountOffset(int argChunkId) {

	int chunkId = argChunkId;
	
	cudaMemcpy(deviceChunkWTCount, WTChunkVec[chunkId].WTCount, (numOfWordS) * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(deviceChunkWTOffset, WTChunkVec[chunkId].WTOffset, (numOfWordS) * sizeof(int), cudaMemcpyHostToDevice);

}

void WTAll::chunkWTCPU2GPU(int argChunkId) {

	int chunkId = argChunkId;

	cudaMemcpy(deviceChunkNZWTCount, WTChunkVec[chunkId].NZWTCount, (numOfWordS) * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(deviceChunkWTIndex, WTChunkVec[chunkId].WTIndex, (WTLengthVec[chunkId]) * sizeof(unsigned short int), cudaMemcpyHostToDevice);
	cudaMemcpy(deviceChunkWTValue, WTChunkVec[chunkId].WTValue, (WTLengthVec[chunkId]) * sizeof(unsigned short int), cudaMemcpyHostToDevice);

}

void WTAll::chunkWTGPU2CPU(int argChunkId) {

	int chunkId = argChunkId;

	cudaMemcpy(WTChunkVec[chunkId].NZWTCount, deviceChunkNZWTCount, (numOfWordS) * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(WTChunkVec[chunkId].WTIndex, deviceChunkWTIndex, (WTLengthVec[chunkId]) * sizeof(unsigned short int), cudaMemcpyDeviceToHost);
	cudaMemcpy(WTChunkVec[chunkId].WTValue, deviceChunkWTValue, (WTLengthVec[chunkId]) * sizeof(unsigned short int), cudaMemcpyDeviceToHost);

}


void WTAll::CPU2Disk(string argFilePrefix) {

	ofstream OutputNZWTCount((argFilePrefix + string("/NZWTCount.txt")).c_str(), ios::binary);
	for (int i = 0; i < numOfWordS; i++) {
		OutputNZWTCount << NZWTCount[i] << "\n";
	}
	OutputNZWTCount.close();
	ofstream OutputWTIndexValue((argFilePrefix + string("/WTIndexValue.txt")).c_str(), ios::binary);
	for (int i = 0; i < maxWTLength; i++) {
		OutputWTIndexValue << WTIndex[i] << " " << WTValue[i] << "\n";
	}
	OutputWTIndexValue.close();

	ofstream OutputWTRowSum((argFilePrefix + string("/WTRowSum.txt")).c_str(), ios::binary);

	for (int i = 0; i < K; i++) {
		OutputWTRowSum << WTRowSum[i]<< "\n";
	}
	OutputWTRowSum.close();

}


void WTAll::CPU2DiskChunk(string argFilePrefix, int argChunkId) {

	int chunkId = argChunkId;
	string chunkFolderName = argFilePrefix + "/chunk" + to_string(chunkId);

	ofstream OutputNZWTCount((chunkFolderName + string("/NZWTCount.txt")).c_str(), ios::binary);
	for (int i = 0; i < numOfWordS; i++) {
		OutputNZWTCount << WTChunkVec[chunkId].NZWTCount[i] << "\n";
	}
	OutputNZWTCount.close();
	ofstream OutputWTIndexValue((chunkFolderName + string("/WTIndexValue.txt")).c_str(), ios::binary);
	for (int i = 0; i < WTLengthVec[chunkId]; i++) {
		OutputWTIndexValue << WTChunkVec[chunkId].WTIndex[i] << " " << WTChunkVec[chunkId].WTValue[i] << "\n";
	}
	OutputWTIndexValue.close();
}







