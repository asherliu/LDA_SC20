#ifndef _DATAGPUCHUNK_H_

#define _DATAGPUCHUNK_H_

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>

//#include "doc_chunk.h"

#include <cuda_runtime_api.h>
#include "Argument.cuh"

using namespace std;





class GPUChunk
{

public:
	int			GPUId;
	int			maxTLLength;// max length of tokenlist
	int			maxDocLength;
	int			wordLength;

	int*		deviceTLTopic;
	int*		deviceTLDocCount;
	int*		deviceTLDocOffset;
	int*		deviceTLWordCount;
	int*		deviceTLWordOffset;
	int*		deviceMapWord2Doc;
	int*		deviceMapDoc2Word;
	float*		devicePerplexity;
	float*		devicePerplexityMid;

	int*		d_dense;
	float*     deviceWTHeadDense;



	GPUChunk(int argMaxTLLength, int argmaxDocLength, int argWordLength);
	void GPUMemAllocate(int);

};


#endif