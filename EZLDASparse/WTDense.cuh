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
#include <cuda_runtime_api.h>
using namespace std;

class WTD
{
public:

	int numOfWordD;
	int wordLength;
	int WTDenseLength;
	float WTMemory;
	/*int* WTRowSumDense;*/
	int* WTDense;
	int* WTDenseCopy;
	/*int* deviceWTRowSumDense;*/
	int* deviceWTDense;
	int* deviceWTDenseCopy;
	WTD(int argNumOfWordD, int argWordLength);
	~WTD()
	{

	};

//	void CPUMemSet();
//	void GPUMemAllocate();
//	void GPUMemset(cudaStream_t& stream);
//	void GPUMemInit();
//	void GPUMemCopy(cudaStream_t& stream);
//	void WTDenGPU2CPU();
//	void WTDenCPU2Disk(string argFilePrefix);


};



#endif
