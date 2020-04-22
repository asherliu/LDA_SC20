#ifndef _DATACHUNK_H_

#define _DATACHUNK_H_




#include <iostream>
#include <string>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

#include <fstream>




using namespace std;


class DocChunk
{
public:
	int chunkId;
	int docLength;
	int TLLength;//store the length of token list
	int wordLength;

	int* TLTopic;// size is maxTLlength
	int* TLDocCount;// size is docStep
	int* TLDocOffset;//size is docStep
	int* TLWordCount;// size is W
	int* TLWordOffset;// size is W
	int* mapWord2Doc; //length of token list
	int* mapDoc2Word;

	

	DocChunk(int argTLLength, int argDocLength, int argWordLength);

	//DocChunk();
	//DocChunk(int argChunkId,
	//	int argDocIdStart,
	//	int argDocIdEnd,
	//	int argNumDocs,
	//	int argNumChunks);
	~DocChunk()
	{

	};
	void CPUMemSet();
	void loadChunk(string argFilePrefix, int argChunkId);
	void GPUMemAllocate();
	void CPU2GPU();
	void GPU2CPU();

};


#endif