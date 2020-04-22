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
#include "WTGPUChunk.cuh"
#include <cuda_runtime_api.h>
using namespace std;

class WTAll
{
public:
	/*int chunkId;*/
	int numGPUs;
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




	int* tmpChunkNZWTCount;
	int* tmpChunkWTIndex;
	int* tmpChunkWTValue;
	int* tmpChunkWTCount;
	int* tmpChunkWTOffset;





	int* NZWTCount;//none-zero count of each column of WT
	int* WTIndex;// size is maxDTlength
	int* WTValue;// size is maxDTlength
	int* WTCount;
	int* WTOffset;
	
	int* deviceZeroChunkWTCount;
	int* deviceZeroChunkWTOffset;
	int* deviceZeroChunkNZWTCount;
	int* deviceZeroChunkWTIndex;
	int* deviceZeroChunkWTValue;
	int* deviceZeroWTRowSum;

	vector<WTChunkData> WTChunkVec;

	vector<WTGPUChunkData> WTGPUChunkVec;


	//////-----chunkWT-----for test--------
	////int* chunkNZWTCount;//none-zero count of each column of WT
	////int* chunkWTIndex;// size is maxDTlength
	////int* chunkWTValue;// size is maxDTlength
	////int* chunkWTRowSum;
	//////------chunkWT-----for test--------




	////vector<int*> WTCountVec;// size is docStep
	////vector<int*> WTOffsetVec;//size is docStep

	//int* deviceNZWTCount;
	//int* deviceWTIndex;
	//int* deviceWTValue;
	//int* deviceWTCount;
	//int* deviceWTOffset;
	//int* deviceWTRowSum;


	//
	//int* deviceChunkNZWTCount;//none-zero count of each column of WT
	//int* deviceChunkWTIndex;// size is maxDTlength
	//int* deviceChunkWTValue;// size is maxDTlength
	//int* deviceChunkWTCount;
	//int* deviceChunkWTOffset;


	////int* deviceChunkWTCount;
	////int* deviceChunkWTOffset;

	//////-----chunkWT-----for test--------
	////int* deviceChunkNZWTCount;//none-zero count of each column of WT
	////int* deviceChunkWTIndex;// size is maxDTlength
	////int* deviceChunkWTValue;// size is maxDTlength
	////int* deviceChunkWTRowSum;
	//////------chunkWT-----for test--------


	//int* deviceBlockCount;
	//int* deviceWarpCount;
	WTAll(int argmaxWTLength, int argWordLength, int argNumChunks, int argMaxChunkWTLength, int argNumOfWordS, int argNumGPUS);

	~WTAll()
	{

	};
	//void loadWTLength(string argFilePrefix);
	void CPUMemSet();
	//void GPUMemAllocate(int);
	//void GPUMemset();
	//void chunkGPUMemset();
	void loadWTLength(string argFilePrefix);
	void loadWTCountOffset(string argFilePrefix);
	void InitWTGPU();
	void GPUMemAllocate();
	void GPUDataTransfer(int argGPUId, cudaStream_t &stream);
	void GPUDataDistribute(int argGPUId, cudaStream_t &stream);
	//void CPU2GPU();
	/*void blockWarpCountCPU2GPU();*/
	void CPU2GPUCountOffset(int argGPUId);
	void WTCPU2GPU(int argGPUId);
	void WTGPU2CPU(int argGPUId);


	void chunkCPU2GPUCountOffset(int argChunkId);

	void GPUDataTransferBackCPU(int argGPUId);

	void GPUDataTransferToGPU(int argGPUId);
	void GPUDataDistributeBackCPU(int argGPUId, cudaStream_t &stream);
	void GPUDataDistributeToGPU(int argGPUId, cudaStream_t &stream);


	void chunkWTCPU2GPU(int argChunkId);
	void chunkWTGPU2CPU(int argChunkId);


	void verifyWTSum();
	//void GPU2CPU();
	//void GPU2CPUChunk(int argChunkId);
	void CPU2Disk(string argFilePrefix);
	void CPU2DiskChunk(string argFilePrefix, int argChunkId);

};


#endif