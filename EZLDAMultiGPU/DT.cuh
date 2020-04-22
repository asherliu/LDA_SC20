#ifndef _DT_H_
#define _DT_H_

#include <iostream>
#include <string>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

#include <fstream>
#include <cuda_runtime_api.h>
#include "DTChunk.cuh"
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
	int GPUId;
	
	int maxDTLength;
	int maxDocLength;

	//int docLength;
	int numChunks;
	int numGPUs;
	float DTMemory;
	int* DTLengthVec;
	int* docLengthVec;
	int* NZDTCount;//none-zero count of each column of DT
	int* DTIndex;// size is maxDTlength
	int* DTValue;// size is maxDTlength


	vector<int*> DTCountVec;// size is docStep
	vector<int*> DTOffsetVec;//size is docStep
	vector<DTGPUChunk> DTGPUChunkVec;

	//int* deviceNZDTCount;
	//int* deviceDTIndex;
	//int* deviceDTValue;
	//int* deviceDTCount;
	//int* deviceDTOffset;

	DTChunk(int argmaxDTLength, int argMaxDocLength,int argNumChunks, int argNumGPUs);

	~DTChunk()
	{

	};
	void loadDocDTLength(string argFilePrefix);
	void CPUMemSet();
	//void GPUMemAllocate(int);
	void loadDTCountOffset(string argFilePrefix);
	void InitDTGPU();
	//void GPUMemSet(int argChunkId);
	void CPU2GPU(int argGPUId);
	void CPU2GPUDTCountOffset(int argGPUId);
	void GPU2CPU(int argGPUId);
	void CPU2Disk(string argFilePrefix, int argChunkId);
};


#endif