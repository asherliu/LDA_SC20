#ifndef _WTDense_H_
#define _WTDense_H_

#include <iostream>
#include <string>
#include <stdio.h>
#include <stdlib.h>
#include <vector>


#include <fstream>
#include "Argument.cuh"
#include "WTChunk.cuh"
#include "WTDenseChunk.cuh"
#include <cuda_runtime_api.h>
using namespace std;

class WTD
{
public:
	int GPUId;
	int numGPUs;
	int numOfWordD;
	int wordLength;
	int WTDenseLength;
	float WTMemory;
	/*int* WTRowSumDense;*/
	int* WTDense;
	int* WTDenseCopy;
	//int* deviceWTRowSumDense;


	int* deviceZeroWTDense;
	/*int* deviceWTDenseCopy;*/

	vector<WTDChunk> WTDenseGPUChunkVec;

	WTD(int argNumOfWordD, int argWordLength, int argNumGPUs);
	~WTD()
	{

	};

	void CPUMemSet();
	void InitWTGPU();
	/*void GPUMemAllocate(int);*/
	void GPUMemset(int argGPUId);
	/*void GPUMemInit();*/

	void WTD::GPUMemAllocate();


	void GPUMemCopy(int argGPUId);
	void WTDenGPU2CPU(int argGPUId);
	void WTDenCPU2Disk(string argFilePrefix);
	void GPUDataTransfer(int argGPUId, cudaStream_t &stream);
	void GPUDataDistribute(int argGPUId, cudaStream_t &stream);
	void GPUDataTransferBackCPU(int argGPUId);

	void GPUDataTransferToGPU(int argGPUId);

	void GPUDataDistributeBackCPU(int argGPUId);

	void GPUDataDistributeToGPU(int argGPUId);





};



#endif