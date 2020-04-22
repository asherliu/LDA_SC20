#ifndef _DOC_H_

#define _DOC_H_

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>

//#include "doc_chunk.h"
#include "DataChunk.cuh"
#include "DataGPUChunk.cuh"
#include <cuda_runtime_api.h>
#include "Argument.cuh"

using namespace std;





class Document
{

public:
	string		filePrefix;
	int			GPUId;
	int			numChunks;
	int			numGPUs;
	int			maxTLLength;// max length of tokenlist
	int			maxDocLength;
	int			wordLength;
	float		TLMemory;
	float      sumPerplexity;
	int*		docLengthVec;
	int*		TLLengthVec;
	int*		numOfTokenVecD;
	int*		numOfTokenVecS;
	float*		perplexityMid;
	//float*		perplexityMid2;

	float*		perplexity;

	//int*		deviceTLTopic;
	//int*		deviceTLDocCount;
	//int*		deviceTLDocOffset;
	//int*		deviceTLWordCount;
	//int*		deviceTLWordOffset;
	//int*		deviceMapWord2Doc;
	//int*		deviceMapDoc2Word;
	//float*		devicePerplexity;
	//float*		devicePerplexityMid;

	int*		d_blockCounter;
	int*		d_warpCounter;
	int*		d_dense;
	float*     deviceWTHeadDense;


	vector<DocChunk> docChunkVec;
	vector<GPUChunk> GPUChunkVec;
	//Document();
	Document(string argFilePrefix, int argNumChunks, int argMaxTLLength, int argmaxDocLength, int argWordLength, int argNumGPUs);

	void loadDocument();
	void InitGPU();
	/*void GPUMemAllocate(int);*/
	void CPU2GPUPerplexity(int argGPUId);
	void GPU2CPUPerplexity(int argGPUId);
	void CPU2GPU(int argGPUId, int argChunkId);
	void GPU2CPU(int argGPUId, int argChunkId);
	void CPU2DiskPerplexity(string argFilePrefix);

};


#endif