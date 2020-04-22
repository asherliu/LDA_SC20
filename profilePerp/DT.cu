#include "DT.cuh"


DTChunk::DTChunk(int argmaxDTLength, int argMaxDocLength, int argNumChunks) {

	maxDTLength = argmaxDTLength;
	maxDocLength = argMaxDocLength;
	numChunks = argNumChunks;
	NZDTCount = new int[maxDocLength];
	DTIndex = new unsigned short int[maxDTLength];
	DTValue = new int[maxDTLength];
	//DTCount = new int[maxDocLength];
	//DTOffset = new int[maxDocLength];
	DTLengthVec = new int[numChunks];
	docLengthVec = new int[numChunks];
}

void DTChunk::loadDocDTLength(string argFilePrefix) {
	ifstream DTLength((argFilePrefix + string("/DTLength.txt")).c_str(), ios::binary);//store max Doc and DT length
	ifstream docLength((argFilePrefix + string("/docLength.txt")).c_str(), ios::binary);//store max Doc and DT length
	for (int chunkId = 0; chunkId < numChunks; chunkId++) {
		DTLength >> DTLengthVec[chunkId];
		docLength >> docLengthVec[chunkId];
	}
	DTLength.close();
	docLength.close();
}



void DTChunk::CPUMemSet() {

	memset(NZDTCount, 0, maxDocLength * sizeof(int));
	memset(DTIndex, 0, maxDTLength * sizeof(unsigned short int));
	memset(DTValue, 0, maxDTLength * sizeof(int));
	//memset(DTCount, 0, maxDocLength * sizeof(int));
	//memset(DTOffset, 0, maxDocLength * sizeof(int));

}

void DTChunk::GPUMemAllocate() {
	cudaMalloc((void**)&deviceNZDTCount, (maxDocLength) * sizeof(int));
	cudaMalloc((void**)&deviceDTIndex, (maxDTLength) * sizeof(unsigned short int));
	cudaMalloc((void**)&deviceDTValue, (maxDTLength) * sizeof(int));
	cudaMalloc((void**)&deviceDTCount, (maxDocLength) * sizeof(int));
	cudaMalloc((void**)&deviceDTOffset, (maxDocLength) * sizeof(int));

	DTMemory = (3 * maxDocLength + 2 * maxDTLength) * sizeof(int) / 1000000000.0;
	printf("DT memory usage:%f GB\n", DTMemory);

}

void DTChunk::loadDTCountOffset(string argFilePrefix) {

	/*chunkId = argChunkId;*/
	
	for (int chunkId = 0; chunkId < numChunks; chunkId++) {
		string chunkFolderName = argFilePrefix + "/chunk" + to_string(chunkId);
		ifstream DTCountOffset((chunkFolderName + string("/DTCountOffset.txt")).c_str(), ios::binary);//store Word offset of TL
		int* DTCount = new int[docLengthVec[chunkId]];
		int* DTOffset = new int[docLengthVec[chunkId]];
		memset(DTCount, 0, docLengthVec[chunkId] * sizeof(int));
		memset(DTOffset, 0, docLengthVec[chunkId] * sizeof(int));

		for (int i = 0; i < docLengthVec[chunkId]; i++)
		{
			DTCountOffset >> DTCount[i] >> DTOffset[i];
		}
		DTCountOffset.close();
		DTCountVec.push_back(DTCount);
		DTOffsetVec.push_back(DTOffset);


	}
	


}




void DTChunk::CPU2GPU(int argChunkId) {
	chunkId = argChunkId;
	//docLength = argDocLength;
	cudaMemcpy(deviceNZDTCount, NZDTCount, (docLengthVec[chunkId]) * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(deviceDTIndex, DTIndex, (DTLengthVec[chunkId]) * sizeof(unsigned short int), cudaMemcpyHostToDevice);
	cudaMemcpy(deviceDTValue, DTValue, (DTLengthVec[chunkId]) * sizeof(int), cudaMemcpyHostToDevice);
	

}


void DTChunk::GPUMemSet(int argChunkId)
{
	chunkId = argChunkId;
	cudaMemset(deviceNZDTCount, 0, (maxDocLength) * sizeof(int));
	cudaMemset(deviceDTIndex, 0, (maxDTLength) * sizeof(unsigned short int));
	cudaMemset(deviceDTValue, 0, (maxDTLength) * sizeof(int));

}






void DTChunk::CPU2GPUDTCountOffset(int argChunkId) {
	chunkId = argChunkId;
	//docLength = argDocLength;

	cudaMemcpy(deviceDTCount, DTCountVec[chunkId], (docLengthVec[chunkId]) * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(deviceDTOffset, DTOffsetVec[chunkId], (docLengthVec[chunkId]) * sizeof(int), cudaMemcpyHostToDevice);

}


void DTChunk::GPU2CPU(int argChunkId) {
	chunkId = argChunkId;
	//docLength = argDocLength;
	cudaMemcpy(NZDTCount, deviceNZDTCount, (docLengthVec[chunkId]) * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(DTIndex, deviceDTIndex, (DTLengthVec[chunkId]) * sizeof(unsigned short int), cudaMemcpyDeviceToHost);
	cudaMemcpy(DTValue, deviceDTValue, (DTLengthVec[chunkId]) * sizeof(int), cudaMemcpyDeviceToHost);

}

void DTChunk::CPU2Disk(string argFilePrefix,int argChunkId) {
	chunkId = argChunkId;
	//docLength = argDocLength;
	string chunkFolderName = argFilePrefix + "/chunk" + to_string(chunkId);
	ofstream OutputNZDTCount((chunkFolderName + string("/NZDTCount.txt")).c_str(), ios::binary);
	for (int i = 0; i < docLengthVec[chunkId]; i++) {
		OutputNZDTCount << NZDTCount[i] << "\n";
	}
	OutputNZDTCount.close();
	ofstream OutputDTIndexValue((chunkFolderName + string("/DTIndexValue.txt")).c_str(), ios::binary);
	for (int i = 0; i < DTLengthVec[chunkId]; i++) {
		OutputDTIndexValue << DTIndex[i] <<" "<<DTValue[i]<< "\n";
	}
	OutputDTIndexValue.close();
}