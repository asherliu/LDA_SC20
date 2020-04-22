#ifndef _DT_H_
#define _DT_H_

#include <iostream>
#include <string>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

#include <fstream>
#include <cuda_runtime_api.h>
#include "Argument.cuh"
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

	int* deviceNZDTCount[numStreams];
	/*unsigned short int* deviceDTIndex[numStreams];
	int* deviceDTValue[numStreams];*/
	int* deviceDTIndexValue[numStreams];
	int* deviceDTCount[numStreams];
	int* deviceDTOffset[numStreams];

	DTChunk(int argmaxDTLength, int argMaxDocLength,int argNumChunks);

	~DTChunk()
	{

	};
	void loadDocDTLength(string argFilePrefix);
	void CPUMemSet();
	void GPUMemAllocate();
	void loadDTCountOffset(string argFilePrefix);
	void GPUMemSet(int argChunkId, int argStreamId, cudaStream_t& stream);
	void CPU2GPU(int argChunkId);
	void CPU2GPUDTCountOffset(int argChunkId, int argStreamId, cudaStream_t& stream);
	void GPU2CPU(int argChunkId);
	void CPU2Disk(string argFilePrefix, int argChunkId);
};


#endif