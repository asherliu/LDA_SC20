#ifndef _DT_H_
#define _DT_H_

#include <iostream>
#include <string>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

#include <fstream>
#include <cuda_runtime_api.h>
using namespace std;

//struct DTCountOffsetStruct{
//
//	int chunkId;
//	int* DTCount;// size is docStep
//	int* DTOffset;//size is docStep
//
//}DTCountOffset;
//
//






class DTChunk
{
public:
	int chunkId;
	
	int maxDTLength;
	int maxDocLength;

	//int docLength;
	int numChunks;
	float DTMemory;
	int* DTLengthVec;
	int* docLengthVec;
	int* NZDTCount;//none-zero count of each column of DT
	unsigned short int* DTIndex;// size is maxDTlength
	int* DTValue;// size is maxDTlength


	vector<int*> DTCountVec;// size is docStep
	vector<int*> DTOffsetVec;//size is docStep

	int* deviceNZDTCount;
	unsigned short int* deviceDTIndex;
	int* deviceDTValue;
	int* deviceDTCount;
	int* deviceDTOffset;

	DTChunk(int argmaxDTLength, int argMaxDocLength,int argNumChunks);

	~DTChunk()
	{

	};
	void loadDocDTLength(string argFilePrefix);
	void CPUMemSet();
	void GPUMemAllocate();
	void loadDTCountOffset(string argFilePrefix);
	void GPUMemSet(int argChunkId);
	void CPU2GPU(int argChunkId);
	void CPU2GPUDTCountOffset(int argChunkId);
	void GPU2CPU(int argChunkId);
	void CPU2Disk(string argFilePrefix, int argChunkId);
};


#endif