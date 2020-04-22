#ifndef _WTDENSECHUNK_H_
#define _WTDENSECHUNK_H_

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

class WTDChunk
{
public:
	int GPUId;
	int numOfWordD;
	int wordLength;
	int WTDenseLength;
	//float WTMemory;
	/*int* WTRowSumDense;*/
	//int* WTDense;
	//int* WTDenseCopy;
	/*int* deviceWTRowSumDense;*/
	int* deviceWTDense;
	int* deviceWTDenseCopy;
	WTDChunk(int argNumOfWordD, int argWordLength, int argGPUId);
	~WTDChunk()
	{

	};

	/*void CPUMemSet();*/
	void GPUMemAllocate(int);
	void GPUMemsetWTDenseCopy(int argGPUId);
	void GPUMemsetWTDense(int argGPUId);
	void GPUMemInit(int argGPUId);
	//void GPUMemCopy();
	//void WTDenGPU2CPU();
	//void WTDenCPU2Disk(string argFilePrefix);


};



#endif