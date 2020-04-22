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
	int			tokenSegment = 1000;
	int			chunksPerStream;
	float		totalNumOfTokens=0.0;
	float		TLMemory;
	float       sumPerplexity;
	float		increasePercent = 0.0;
	float		topicUnchangedPercent = 0.0;
	float*		perplexityAve;

	int*		docLengthVec;
	int*		TLLengthVec;
	int*		numOfTokenVecD;
	int*		numOfTokenVecS;
	int*        effectiveTokenIndex;
	int*		newTokenCount;

	float*		perplexityMid;
	float*		perplexity;
	float*		timeRecord;
	
	unsigned short int* maxTokenCount;
	unsigned short int* Mflag;
	

	
	unsigned short int*		deviceTLTopic[numStreams];
	int*					deviceTLDocCount[numStreams];
	int*					deviceTLDocOffset[numStreams];
	int*					deviceTLWordCount[numStreams];
	int*					deviceTLWordOffset[numStreams];
	int*					deviceMapWord2Doc[numStreams];
	int*					deviceMapDoc2Word[numStreams];
	int*					deviceEffectiveTokenIndex[numStreams];
	int*					deviceNewTokenCount[numStreams];
	/*float*					devicePerplexity[numStreams];*/
	
	float*					deviceTimeRecord[numStreams];
	float*					deviceRandomfloat[numStreams];
	float*					devicePerplexityAve[numStreams];
	unsigned short int*		deviceMflag[numStreams];

	/*float*					deviceProbMax;
	unsigned short int*		deviceProbMaxTopic;
	unsigned short int*		deviceProbMaxFlag;
	unsigned short int*		deviceProbMaxTopicFlag;*/

	unsigned short int*		deviceMaxTokenCount[numStreams];
	unsigned short int*		deviceSecondMaxTokenCount[numStreams];
	unsigned short int*		deviceMaxTopic[numStreams];
	unsigned short int*		deviceSecondMaxTopic[numStreams];
	float*					deviceMaxProb[numStreams];
	float*					deviceThresProb[numStreams];

	unsigned short int*		deviceWordMaxTopic[numStreams];
	unsigned short int*		deviceWordSecondMaxTopic[numStreams];
	unsigned short int*		deviceWordThirdMaxTopic[numStreams];

	int*		d_blockCounter[numStreams];
	int*		d_warpCounter[numStreams];
	int*		d_dense[numStreams];
	float*		deviceWTHeadDense[numStreams];

	unsigned int			counterWTUpdateKernel = 0;
	unsigned int			counterWTDenUpdateKernel = 0;
	unsigned int			counterWTAdditionKernel = 0;
	unsigned int			counterMaxTopicKernel = 0;
	unsigned int			counterDTUpdateKernel = 0;
	unsigned int			counterUpdateProbKernel = 0;
	unsigned int			counterSampleKernelD = 0;
	unsigned int			counterSampleKernelS = 0;

	unsigned int*			deviceCounterWTUpdateKernel[numStreams];
	unsigned int*			deviceCounterWTDenUpdateKernel[numStreams];
	unsigned int*			deviceCounterWTAdditionKernel[numStreams];
	unsigned int*			deviceCounterMaxTopicKernel[numStreams];
	unsigned int*			deviceCounterDTUpdateKernel[numStreams];
	unsigned int*			deviceCounterUpdateProbKernel[numStreams];
	unsigned int*			deviceCounterSampleKernelD[numStreams];
	unsigned int*			deviceCounterSampleKernelS[numStreams];
	float*					devicePerplexityMid[numStreams];
	vector<DocChunk> docChunkVec;

	





	//vector<float*> probMaxChunkVec;
	//vector<unsigned short int*> probMaxTopicChunkVec;
	//vector<unsigned short int*> probMaxFlagChunkVec;
	//vector<unsigned short int*> probMaxTopicFlagChunkVec;
	vector<unsigned short int*> maxTokenCountVec;


	//Document();
	Document(string argFilePrefix, int argNumChunks, int argMaxTLLength, int argmaxDocLength, int argWordLength);

	void loadDocument();
	void GPUMemAllocate();
	void CPU2GPUPerplexity(cudaStream_t& stream);
	void GPU2CPUPerplexity(cudaStream_t& stream);
	void CPU2GPU(int argChunkId, int argStreamId, cudaStream_t& stream);
	void GPU2CPU(int argChunkId, int argStreamId, cudaStream_t& stream);
	void CPU2DiskPerplexity(string argFilePrefix);
	void GPU2CPUTime();
	void GPU2CPUEffectiveTokenIndex();
	void CPU2DiskEffectiveTokenIndex(string argFilePrefix);
	void deviceCounterMemAllocate();
	/*void CPU2DiskTime(ofstream argOutPutTime);*/
	//void PercentageCalculate();

};


#endif