#ifndef _DTCHUNK_H_
#define _DTCHUNK_H_

#include <iostream>
#include <string>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

#include <fstream>
#include <cuda_runtime_api.h>
using namespace std;


class DTGPUChunk
{
public:

	/*int chunkId;*/
	int GPUId;
	/*int numGPUs;*/
	int maxDTLength;
	int maxDocLength;

	//int docLength;
	//int numChunks;
	//float DTMemory;
	//int* DTLengthVec;
	//int* docLengthVec;
	//int* NZDTCount;//none-zero count of each column of DT
	//int* DTIndex;// size is maxDTlength
	//int* DTValue;// size is maxDTlength


	//vector<int*> DTCountVec;// size is docStep
	//vector<int*> DTOffsetVec;//size is docStep

	int* deviceNZDTCount;
	int* deviceDTIndex;
	int* deviceDTValue;
	int* deviceDTCount;
	int* deviceDTOffset;

	DTGPUChunk(int argmaxDTLength, int argMaxDocLength, int argGPUId);

	~DTGPUChunk()
	{

	};
	//void loadDocDTLength(string argFilePrefix);
	//void CPUMemSet();
	void GPUMemAllocate(int);
	void GPUMemSet(int argGPUId);
	//void loadDTCountOffset(string argFilePrefix);
	//void GPUMemSet(int argChunkId);
	//void CPU2GPU(int argChunkId);
	//void CPU2GPUDTCountOffset(int argChunkId);
	//void GPU2CPU(int argChunkId);
	//void CPU2Disk(string argFilePrefix, int argChunkId);
};


#endif