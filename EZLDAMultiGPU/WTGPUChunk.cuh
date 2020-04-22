#ifndef _WTGPUCHUNK_H_
#define _WTGPUCHUNK_H_

#include <iostream>
#include <string>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

#include <fstream>
#include "Argument.cuh"
#include <cuda_runtime_api.h>
using namespace std;

class WTGPUChunkData
{
public:
	int GPUId;
	int wordLength;
	int maxWTLength;
	int maxChunkWTLength;

	int WTLength;
	int numOfWordS;

	int* deviceNZWTCount;
	int* deviceWTIndex;
	int* deviceWTValue;
	int* deviceWTCount;
	int* deviceWTOffset;
	int* deviceWTRowSum;

	int* deviceChunkWTCount;
	int* deviceChunkWTOffset;
	int* deviceChunkNZWTCount;
	int* deviceChunkWTIndex;
	int* deviceChunkWTValue;

	WTGPUChunkData(int argGPUId, int argWordLength, int argMaxChunkWTLength, int argWTLength, int argNumOfWordS);

	~WTGPUChunkData()
	{

	};


	void GPUMemAllocate(int argGPUId);
	void GPUMemset(int argGPUId);
	void chunkGPUMemset(int argGPUId);

	//void CPUMemSet();
	//void loadWTCountOffset(string argFilePrefix);

};


#endif