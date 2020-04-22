#ifndef _WT_H_
#define _WT_H_

#include <iostream>
#include <string>
#include <stdio.h>
#include <stdlib.h>
#include <vector>


#include <fstream>
#include "Argument.cuh"
#include "WTChunk.cuh"
#include <cuda_runtime_api.h>
using namespace std;

class WTAll
{
public:
	/*int chunkId;*/
	int blockCount;// number of words processed by block
	int warpCount;// number of words processed by warp

	int maxWTLength;
	int wordLength;
	int numOfWordS;

	int maxChunkWTLength;
	int numChunks;
	float WTMemory;

	int* WTLengthVec;
	int* WTRowSum;


	int* NZWTCount;//none-zero count of each column of WT
	unsigned short int* WTIndex;// size is maxDTlength
	unsigned short int* WTValue;// size is maxDTlength
	int* WTCount;
	int* WTOffset;
	

	vector<WTChunkData> WTChunkVec;

	////-----chunkWT-----for test--------
	int* chunkNZWTCount;//none-zero count of each column of WT
	int* chunkWTIndex;// size is maxDTlength
	int* chunkWTValue;// size is maxDTlength
	//int* chunkWTRowSum;
	////------chunkWT-----for test--------




	//vector<int*> WTCountVec;// size is docStep
	//vector<int*> WTOffsetVec;//size is docStep

	int* deviceNZWTCount;
	unsigned short int* deviceWTIndex;
	unsigned short int* deviceWTValue;
	int* deviceWTCount;
	int* deviceWTOffset;
	int* deviceWTRowSum;


	
	int* deviceChunkNZWTCount[numStreams];//none-zero count of each column of WT
	unsigned short int* deviceChunkWTIndex[numStreams];// size is maxDTlength
	unsigned short int* deviceChunkWTValue[numStreams];// size is maxDTlength
	int* deviceChunkWTCount[numStreams];
	int* deviceChunkWTOffset[numStreams];


	//int* deviceChunkWTCount;
	//int* deviceChunkWTOffset;

	////-----chunkWT-----for test--------
	//int* deviceChunkNZWTCount;//none-zero count of each column of WT
	//int* deviceChunkWTIndex;// size is maxDTlength
	//int* deviceChunkWTValue;// size is maxDTlength
	//int* deviceChunkWTRowSum;
	////------chunkWT-----for test--------


	int* deviceBlockCount;
	int* deviceWarpCount;
	WTAll(int argmaxWTLength, int argWordLength, int argNumChunks, int argMaxChunkWTLength, int argNumOfWordS);

	~WTAll()
	{

	};
	//void loadWTLength(string argFilePrefix);
	void CPUMemSet();
	void GPUMemAllocate();
	void GPUMemset(cudaStream_t& stream);
	void chunkGPUMemset(int argStreamId, cudaStream_t& stream);
	void loadWTLength(string argFilePrefix);
	void loadWTCountOffset(string argFilePrefix);
	//void CPU2GPU();
	void blockWarpCountCPU2GPU();
	void CPU2GPUCountOffset(cudaStream_t& stream);
	void WTCPU2GPU(cudaStream_t& stream);
	void WTGPU2CPU();


	void chunkCPU2GPUCountOffset(int argChunkId, int argStreamId, cudaStream_t& stream);
	void chunkWTCPU2GPU(int argChunkId, int argStreamId, cudaStream_t& stream);
	void chunkWTGPU2CPU(int argChunkId, int argStreamId, cudaStream_t& stream);
	//void GPU2CPU();
	//void GPU2CPUChunk(int argChunkId);
	void CPU2Disk(string argFilePrefix);
	void CPU2DiskChunk(string argFilePrefix, int argChunkId);

};


#endif