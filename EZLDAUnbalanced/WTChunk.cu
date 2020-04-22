#include "WTChunk.cuh"
WTChunkData::WTChunkData(int argChunkId, int argWordLength, int argMaxChunkWTLength,int argWTLength, int argNumOfWordS) {

	chunkId=argChunkId;
	wordLength=argWordLength;
	maxChunkWTLength = argMaxChunkWTLength;
	WTLength = argWTLength;
	numOfWordS = argNumOfWordS;

	//NZWTCount=new int[numOfWordS];

	//WTIndex=new unsigned short int[WTLength];
	//WTValue=new unsigned short int[WTLength];

	//WTCount=new int[numOfWordS];
	//WTOffset= new int[numOfWordS];

	cudaMallocHost((void**)&NZWTCount, numOfWordS * sizeof(int));
	cudaMallocHost((void**)&WTIndex, WTLength * sizeof(unsigned short int));
	cudaMallocHost((void**)&WTValue, WTLength * sizeof(unsigned short int));
	cudaMallocHost((void**)&WTCount, numOfWordS * sizeof(int));
	cudaMallocHost((void**)&WTOffset, numOfWordS * sizeof(int));


}

void WTChunkData::CPUMemSet() {

	memset(NZWTCount, 0, numOfWordS * sizeof(int));
	memset(WTIndex, 0, WTLength * sizeof(unsigned short int));
	memset(WTValue, 0, WTLength * sizeof(unsigned short int));
	memset(WTCount, 0, numOfWordS * sizeof(int));
	memset(WTOffset, 0, numOfWordS * sizeof(int));

}


void WTChunkData::loadWTCountOffset(string argFilePrefix) {

	string chunkFolderName = argFilePrefix + "/chunk" + to_string(chunkId);

	ifstream WTCountOffset((chunkFolderName + string("/WTCountOffset.txt")).c_str(), ios::binary);//store Word offset of TL

	for (int i = 0; i < numOfWordS; i++)
	{
		WTCountOffset >> WTCount[i] >> WTOffset[i];
	}
	WTCountOffset.close();

	printf("WT Count and Offset loaded!...: chunkId=%d\n",chunkId);

}