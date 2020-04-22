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
#include <cuda_runtime_api.h>
#include "Argument.cuh"

using namespace std;





class Document
{

public:
	string		filePrefix;
	int			numChunks;
	int			maxTLLength;// max length of tokenlist
	int			maxDocLength;
	int			wordLength;
	int			tokenSegment = 2000;
	float		totalNumOfTokens=0.0;
	float		TLMemory;
	float       sumPerplexity;
	int*		docLengthVec;
	int*		TLLengthVec;
	int*		numOfTokenVecD;
	int*		numOfTokenVecS;
	float*		perplexityMid;
	float*		perplexity;

	
	unsigned short int*		deviceTLTopic;
	int*					deviceTLDocCount;
	int*					deviceTLDocOffset;
	int*					deviceTLWordCount;
	int*					deviceTLWordOffset;
	int*					deviceMapWord2Doc;
	int*					deviceMapDoc2Word;
	float*					devicePerplexity;
	float*					devicePerplexityMid;

	int*		d_blockCounter;
	int*		d_warpCounter;
	int*		d_dense;
	float*     deviceWTHeadDense;


	vector<DocChunk> docChunkVec;
	//Document();
	Document(string argFilePrefix, int argNumChunks, int argMaxTLLength, int argmaxDocLength, int argWordLength);

	void loadDocument();
	void GPUMemAllocate();
	void CPU2GPUPerplexity();
	void GPU2CPUPerplexity();
	void CPU2GPU(int);
	void GPU2CPU(int argChunkId);
	void CPU2DiskPerplexity(string argFilePrefix);

};


#endif
