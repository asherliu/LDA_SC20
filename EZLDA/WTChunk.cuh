#ifndef _WTCHUNK_H_
#define _WTCHUNK_H_

#include <iostream>
#include <string>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

#include <fstream>
#include "Argument.cuh"
#include <cuda_runtime_api.h>
using namespace std;

class WTChunkData
{
public:
	int chunkId;
	int wordLength;
	int maxChunkWTLength;
	
	int WTLength;
	int numOfWordS;
	int* NZWTCount;
	unsigned short int* WTIndex;
	unsigned short int* WTValue;
	
	int* WTCount;
	int* WTOffset;

	WTChunkData(int argChunkId, int argWordLength, int argMaxChunkWTLength,int argWTLength, int argNumOfWordS);

	~WTChunkData()
	{

	};

	void CPUMemSet();
	void loadWTCountOffset(string argFilePrefix);

};


#endif