
#include "WT.cuh"
WTAll::WTAll(int argmaxWTLength, int argWordLength, int argNumChunks, int argMaxChunkWTLength, int argNumOfWordS, int argNumGPUS) {
	maxWTLength = argmaxWTLength;
	wordLength = argWordLength;
	numChunks = argNumChunks;
	maxChunkWTLength = argMaxChunkWTLength;
	numOfWordS = argNumOfWordS;
	numGPUs = argNumGPUS;
	WTLengthVec = new int[numChunks];
	WTRowSum = new int[K];

	NZWTCount = new int[numOfWordS];
	WTIndex = new int[maxWTLength];
	WTValue = new int[maxWTLength];
	WTCount = new int[wordLength];
	WTOffset = new int[wordLength];
	
	


	tmpChunkNZWTCount = new int[numOfWordS];
	tmpChunkWTIndex = new int[maxChunkWTLength];
	tmpChunkWTValue = new int[maxChunkWTLength];
	tmpChunkWTCount = new int[numOfWordS];
	tmpChunkWTOffset = new int[numOfWordS];










	////-----chunkWT-----for test--------
	//chunkNZWTCount = new int[wordLength];
	//chunkWTIndex = new int[maxChunkWTLength];
	//chunkWTValue = new int[maxChunkWTLength];
	////-----chunkWT-----for test--------


}




void WTAll::CPUMemSet() {

	memset(NZWTCount, 0, numOfWordS * sizeof(int));
	memset(WTIndex, 0, maxWTLength * sizeof(int));
	memset(WTValue, 0, maxWTLength * sizeof(int));
	memset(WTCount, 0, wordLength * sizeof(int));
	memset(WTOffset, 0, wordLength * sizeof(int));
	memset(WTRowSum, 0, K * sizeof(int));

}
//void WTAll::GPUMemAllocate(int argGPUId) {
//	
//	GPUId = argGPUId;
//	cudaSetDevice(GPUId);
//	cudaMalloc((void**)&deviceNZWTCount, (numOfWordS) * sizeof(int));
//	cudaMalloc((void**)&deviceWTIndex, (maxWTLength) * sizeof(int));
//	cudaMalloc((void**)&deviceWTValue, (maxWTLength) * sizeof(int));
//	cudaMalloc((void**)&deviceWTCount, (wordLength) * sizeof(int));
//	cudaMalloc((void**)&deviceWTOffset, (wordLength) * sizeof(int));
//
//	cudaMalloc((void**)&deviceWTRowSum, (K) * sizeof(int));
//	cudaMalloc((void**)&deviceBlockCount, (1) * sizeof(int));
//	cudaMalloc((void**)&deviceWarpCount, (1) * sizeof(int));
//
//	cudaMalloc((void**)&deviceChunkWTCount, (numOfWordS) * sizeof(int));
//	cudaMalloc((void**)&deviceChunkWTOffset, (numOfWordS) * sizeof(int));
//	cudaMalloc((void**)&deviceChunkNZWTCount, (numOfWordS) * sizeof(int));
//	cudaMalloc((void**)&deviceChunkWTIndex, (maxChunkWTLength) * sizeof(int));
//	cudaMalloc((void**)&deviceChunkWTValue, (maxChunkWTLength) * sizeof(int));
//
//
//	WTMemory = (6 * wordLength + 2 * maxWTLength + K + 2 * maxChunkWTLength) /1000000000.0 * sizeof(int);
//	printf("WT memory usage(Sparse):%f GB\n", WTMemory);
//	WTMemory = K /1000000000.0 * wordLength * sizeof(int);
//	printf("WT memory usage(Dense):%f GB\n", WTMemory);
//
//}
//void WTAll::GPUMemset()
//{
//	cudaSetDevice(GPUId);
//	cudaMemset(deviceNZWTCount, 0, (numOfWordS) * sizeof(int));
//	cudaMemset(deviceWTIndex, 0, (maxWTLength) * sizeof(int));
//	cudaMemset(deviceWTValue, 0, (maxWTLength) * sizeof(int));
//	cudaMemset(deviceWTRowSum, 0, (K) * sizeof(int));
//
//}
//
//void WTAll::chunkGPUMemset()
//{
//	/*cudaSetDevice(GPUId);*/
//	cudaMemset(deviceChunkNZWTCount, 0, (numOfWordS) * sizeof(int));
//	cudaMemset(deviceChunkWTIndex, 0, (maxChunkWTLength) * sizeof(int));
//	cudaMemset(deviceChunkWTValue, 0, (maxChunkWTLength) * sizeof(int));
//	//cudaMemset(deviceWTRowSum, 0, (K) * sizeof(int));
//
//}


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


void WTAll::InitWTGPU()
{
	for (int GPUId = 0; GPUId < numGPUs; GPUId++) {

		WTGPUChunkData GPUChunkWTData(GPUId, wordLength, maxChunkWTLength, WTLengthVec[GPUId], numOfWordS);
		GPUChunkWTData.GPUMemAllocate(GPUId);
		GPUChunkWTData.GPUMemset(GPUId);
		GPUChunkWTData.chunkGPUMemset(GPUId);
		WTGPUChunkVec.push_back(GPUChunkWTData);
	}

}

void WTAll::GPUMemAllocate() {

	cudaSetDevice(0);
	cudaMalloc((void**)&deviceZeroWTRowSum, (K) * sizeof(int));
	cudaMalloc((void**)&deviceZeroChunkWTCount, (numOfWordS) * sizeof(int));
	cudaMalloc((void**)&deviceZeroChunkWTOffset, (numOfWordS) * sizeof(int));
	cudaMalloc((void**)&deviceZeroChunkNZWTCount, (numOfWordS) * sizeof(int));
	cudaMalloc((void**)&deviceZeroChunkWTIndex, (maxChunkWTLength) * sizeof(int));
	cudaMalloc((void**)&deviceZeroChunkWTValue, (maxChunkWTLength) * sizeof(int));

}





void WTAll::GPUDataTransfer(int argGPUId, cudaStream_t &stream) {
	cudaSetDevice(0);

	cudaMemcpy(deviceZeroChunkWTCount, WTGPUChunkVec[argGPUId].deviceChunkWTCount, (numOfWordS) * sizeof(int), cudaMemcpyDeviceToDevice);
	cudaMemcpy(deviceZeroChunkWTOffset, WTGPUChunkVec[argGPUId].deviceChunkWTOffset, (numOfWordS) * sizeof(int), cudaMemcpyDeviceToDevice);
	cudaMemcpy(deviceZeroChunkNZWTCount, WTGPUChunkVec[argGPUId].deviceChunkNZWTCount, (numOfWordS) * sizeof(int), cudaMemcpyDeviceToDevice);
	cudaMemcpy(deviceZeroChunkWTIndex, WTGPUChunkVec[argGPUId].deviceChunkWTIndex, (maxChunkWTLength) * sizeof(int), cudaMemcpyDeviceToDevice);
	cudaMemcpy(deviceZeroChunkWTValue, WTGPUChunkVec[argGPUId].deviceChunkWTValue, (maxChunkWTLength) * sizeof(int), cudaMemcpyDeviceToDevice);

	/*cudaMemcpyAsync(deviceZeroChunkWTCount, WTGPUChunkVec[argGPUId].deviceChunkWTCount, (numOfWordS) * sizeof(int), cudaMemcpyDeviceToDevice, stream);
	cudaMemcpyAsync(deviceZeroChunkWTOffset, WTGPUChunkVec[argGPUId].deviceChunkWTOffset, (numOfWordS) * sizeof(int), cudaMemcpyDeviceToDevice, stream);
	cudaMemcpyAsync(deviceZeroChunkNZWTCount, WTGPUChunkVec[argGPUId].deviceChunkNZWTCount, (numOfWordS) * sizeof(int), cudaMemcpyDeviceToDevice, stream);
	cudaMemcpyAsync(deviceZeroChunkWTIndex, WTGPUChunkVec[argGPUId].deviceChunkWTIndex, (maxChunkWTLength) * sizeof(int), cudaMemcpyDeviceToDevice, stream);
	cudaMemcpyAsync(deviceZeroChunkWTValue, WTGPUChunkVec[argGPUId].deviceChunkWTValue, (maxChunkWTLength) * sizeof(int), cudaMemcpyDeviceToDevice, stream);*/


}

void WTAll::GPUDataDistribute(int argGPUId, cudaStream_t &stream)
{
	cudaSetDevice(0);
	/*cudaMemcpyAsync(WTGPUChunkVec[argGPUId].deviceWTRowSum, WTGPUChunkVec[0].deviceWTRowSum,(K) * sizeof(int), cudaMemcpyDeviceToDevice, stream);
	cudaMemcpyAsync(WTGPUChunkVec[argGPUId].deviceNZWTCount, WTGPUChunkVec[0].deviceNZWTCount, (numOfWordS) * sizeof(int), cudaMemcpyDeviceToDevice, stream);
	cudaMemcpyAsync(WTGPUChunkVec[argGPUId].deviceWTIndex, WTGPUChunkVec[0].deviceWTIndex, (maxWTLength) * sizeof(int), cudaMemcpyDeviceToDevice, stream);
	cudaMemcpyAsync(WTGPUChunkVec[argGPUId].deviceWTValue, WTGPUChunkVec[0].deviceWTValue, (maxWTLength) * sizeof(int), cudaMemcpyDeviceToDevice, stream);*/

	cudaMemcpy(WTGPUChunkVec[argGPUId].deviceWTRowSum, WTGPUChunkVec[0].deviceWTRowSum, (K) * sizeof(int), cudaMemcpyDeviceToDevice);
	cudaMemcpy(WTGPUChunkVec[argGPUId].deviceNZWTCount, WTGPUChunkVec[0].deviceNZWTCount, (numOfWordS) * sizeof(int), cudaMemcpyDeviceToDevice);
	cudaMemcpy(WTGPUChunkVec[argGPUId].deviceWTIndex, WTGPUChunkVec[0].deviceWTIndex, (maxWTLength) * sizeof(int), cudaMemcpyDeviceToDevice);
	cudaMemcpy(WTGPUChunkVec[argGPUId].deviceWTValue, WTGPUChunkVec[0].deviceWTValue, (maxWTLength) * sizeof(int), cudaMemcpyDeviceToDevice);

}






//void WTAll::blockWarpCountCPU2GPU() {
//	cudaSetDevice(GPUId);
//	cudaMemcpy(deviceBlockCount, &blockCount, (1) * sizeof(int), cudaMemcpyHostToDevice);
//	cudaMemcpy(deviceWarpCount, &warpCount, (1) * sizeof(int), cudaMemcpyHostToDevice);
//
//}

void WTAll::CPU2GPUCountOffset(int argGPUId) {

	cudaSetDevice(argGPUId);
	cudaMemcpy(WTGPUChunkVec[argGPUId].deviceWTCount, WTCount, (wordLength) * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(WTGPUChunkVec[argGPUId].deviceWTOffset, WTOffset, (wordLength) * sizeof(int), cudaMemcpyHostToDevice);

}

void WTAll::WTCPU2GPU(int argGPUId) {
	cudaSetDevice(argGPUId);
	cudaMemcpy(WTGPUChunkVec[argGPUId].deviceNZWTCount, NZWTCount, (numOfWordS) * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(WTGPUChunkVec[argGPUId].deviceWTIndex, WTIndex, (maxWTLength) * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(WTGPUChunkVec[argGPUId].deviceWTValue, WTValue, (maxWTLength) * sizeof(int), cudaMemcpyHostToDevice);
	
}

void WTAll::WTGPU2CPU(int argGPUId) {
	cudaSetDevice(argGPUId);
	cudaMemcpy(NZWTCount, WTGPUChunkVec[argGPUId].deviceNZWTCount, (numOfWordS) * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(WTIndex, WTGPUChunkVec[argGPUId].deviceWTIndex, (maxWTLength) * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(WTValue, WTGPUChunkVec[argGPUId].deviceWTValue, (maxWTLength) * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(WTRowSum, WTGPUChunkVec[argGPUId].deviceWTRowSum, (K) * sizeof(int), cudaMemcpyDeviceToHost);
}


void WTAll::chunkCPU2GPUCountOffset(int argGPUId) {

	/*int chunkId = argChunkId;*/
	cudaSetDevice(argGPUId);
	cudaMemcpy(WTGPUChunkVec[argGPUId].deviceChunkWTCount, WTChunkVec[argGPUId].WTCount, (numOfWordS) * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(WTGPUChunkVec[argGPUId].deviceChunkWTOffset, WTChunkVec[argGPUId].WTOffset, (numOfWordS) * sizeof(int), cudaMemcpyHostToDevice);

}

void WTAll::GPUDataTransferBackCPU(int argGPUId) {

	/*int chunkId = argChunkId;*/
	cudaSetDevice(argGPUId);
	
	cudaMemcpy(tmpChunkWTCount, WTGPUChunkVec[argGPUId].deviceChunkWTCount, (numOfWordS) * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(tmpChunkWTOffset, WTGPUChunkVec[argGPUId].deviceChunkWTOffset, (numOfWordS) * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(tmpChunkNZWTCount, WTGPUChunkVec[argGPUId].deviceChunkNZWTCount, (numOfWordS) * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(tmpChunkWTIndex, WTGPUChunkVec[argGPUId].deviceChunkWTIndex, (maxChunkWTLength) * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(tmpChunkWTValue, WTGPUChunkVec[argGPUId].deviceChunkWTValue, (maxChunkWTLength) * sizeof(int), cudaMemcpyDeviceToHost);

}


void WTAll::GPUDataTransferToGPU(int argGPUId) {

	cudaSetDevice(0);

	cudaMemcpy(deviceZeroChunkWTCount, tmpChunkWTCount,(numOfWordS) * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(deviceZeroChunkWTOffset, tmpChunkWTOffset, (numOfWordS) * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(deviceZeroChunkNZWTCount, tmpChunkNZWTCount,  (numOfWordS) * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(deviceZeroChunkWTIndex, tmpChunkWTIndex, (maxChunkWTLength) * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(deviceZeroChunkWTValue, tmpChunkWTValue, (maxChunkWTLength) * sizeof(int), cudaMemcpyHostToDevice);

}


void WTAll::GPUDataDistributeBackCPU(int argGPUId, cudaStream_t &stream)
{
	cudaSetDevice(0);
	/*cudaMemcpyAsync(WTGPUChunkVec[argGPUId].deviceWTRowSum, WTGPUChunkVec[0].deviceWTRowSum,(K) * sizeof(int), cudaMemcpyDeviceToDevice, stream);
	cudaMemcpyAsync(WTGPUChunkVec[argGPUId].deviceNZWTCount, WTGPUChunkVec[0].deviceNZWTCount, (numOfWordS) * sizeof(int), cudaMemcpyDeviceToDevice, stream);
	cudaMemcpyAsync(WTGPUChunkVec[argGPUId].deviceWTIndex, WTGPUChunkVec[0].deviceWTIndex, (maxWTLength) * sizeof(int), cudaMemcpyDeviceToDevice, stream);
	cudaMemcpyAsync(WTGPUChunkVec[argGPUId].deviceWTValue, WTGPUChunkVec[0].deviceWTValue, (maxWTLength) * sizeof(int), cudaMemcpyDeviceToDevice, stream);*/

	cudaMemcpy(WTRowSum, WTGPUChunkVec[0].deviceWTRowSum, (K) * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(NZWTCount, WTGPUChunkVec[0].deviceNZWTCount, (numOfWordS) * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(WTIndex, WTGPUChunkVec[0].deviceWTIndex, (maxWTLength) * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(WTValue, WTGPUChunkVec[0].deviceWTValue, (maxWTLength) * sizeof(int), cudaMemcpyDeviceToHost);

}


void WTAll::GPUDataDistributeToGPU(int argGPUId, cudaStream_t &stream) {


	cudaSetDevice(argGPUId);


	cudaMemcpy(WTGPUChunkVec[argGPUId].deviceWTRowSum, WTRowSum,(K) * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(WTGPUChunkVec[argGPUId].deviceNZWTCount, NZWTCount, (numOfWordS) * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(WTGPUChunkVec[argGPUId].deviceWTIndex, WTIndex, (maxWTLength) * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(WTGPUChunkVec[argGPUId].deviceWTValue, WTValue,(maxWTLength) * sizeof(int), cudaMemcpyHostToDevice);



}



void WTAll::chunkWTCPU2GPU(int argGPUId) {

	/*int chunkId = argChunkId;*/
	cudaSetDevice(argGPUId);
	cudaMemcpy(WTGPUChunkVec[argGPUId].deviceChunkNZWTCount, WTChunkVec[argGPUId].NZWTCount, (numOfWordS) * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(WTGPUChunkVec[argGPUId].deviceChunkWTIndex, WTChunkVec[argGPUId].WTIndex, (WTLengthVec[argGPUId]) * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(WTGPUChunkVec[argGPUId].deviceChunkWTValue, WTChunkVec[argGPUId].WTValue, (WTLengthVec[argGPUId]) * sizeof(int), cudaMemcpyHostToDevice);

}

void WTAll::verifyWTSum() {

	cudaSetDevice(0);

	cudaMemcpy(WTRowSum, WTGPUChunkVec[0].deviceWTRowSum, (K) * sizeof(int), cudaMemcpyDeviceToHost);
	int sum = 0;
	for (int i = 0; i < K; i++) {
		sum += WTRowSum[i];

	}
	printf("\nRow sum:%d\n", sum);



}







void WTAll::chunkWTGPU2CPU(int argGPUId) {

	/*int chunkId = argChunkId;*/
	cudaSetDevice(argGPUId);
	cudaMemcpy(WTChunkVec[argGPUId].NZWTCount, WTGPUChunkVec[argGPUId].deviceChunkNZWTCount, (numOfWordS) * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(WTChunkVec[argGPUId].WTIndex, WTGPUChunkVec[argGPUId].deviceChunkWTIndex, (WTLengthVec[argGPUId]) * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(WTChunkVec[argGPUId].WTValue, WTGPUChunkVec[argGPUId].deviceChunkWTValue, (WTLengthVec[argGPUId]) * sizeof(int), cudaMemcpyDeviceToHost);

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







