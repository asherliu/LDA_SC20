
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <time.h>
#include <sstream>
//#include <windows.h>
#include<algorithm>
#include<unistd.h>
#include<sys/stat.h>
//#include <io.h>
//#include <direct.h>
using namespace std;


//void computeMaxVec(int* argMaxVec, int* argCompareVec, int argVecLength);
void computeOffset(int* argOffset, int* argCount, int argVecLength);
void creatFolder(string argFolderName);

typedef struct indexSort
{
	int wordId;
	int tokenCount;
	int wordRank;

}indexSortStruct;


typedef struct chunkSort
{
	int mapDoc2Word;
	int docId;
	int wordId;
	int wordRank;
	int topic;
}chunkSortStruct;

typedef struct chunkMapSort
{
	int mapDoc2Word;
	int mapWord2Doc;
}chunkMapSortStruct;


typedef struct maxLengthStruct
{
	int maxTLLength;// length of tokenlist
	int maxDTLength;//length of DT matrix
	int maxWTLength;//length of WT matrix
	int docStep;//length of doc count and offset
	int W;//length of word count and offset
	int maxChunkWTLength;

}maxLength; //store the max length of vector


bool compareTokenCount(const indexSortStruct& a, const indexSortStruct& b)
{
	return a.tokenCount > b.tokenCount;
}

bool getWordRank(const indexSortStruct& a, const indexSortStruct& b)
{
	return a.wordId < b.wordId;
}

bool sortWordRank(const chunkSortStruct& a, const chunkSortStruct& b)
{
	return a.wordRank < b.wordRank;
}

bool sortChunkMap(const chunkMapSortStruct& a, const chunkMapSortStruct& b)
{
	return a.mapDoc2Word < b.mapDoc2Word;
}


const int K = 1024;


int main(int argc, char *argv[]) {
	//string docIdxFileName = filePrefix + ".doc.idx";
	string docIdxFileName = "/gpfs/alpine/proj-shared/csc289/lda/datasets/docword.nytimes.txt";
	ifstream docIdxStream(docIdxFileName.c_str(), ios::binary);
	if (!docIdxStream.is_open())
	{
		cout << "File " << docIdxFileName << " open failed" << endl;
		exit(0);
	}
	int D;		//number of Document in the whole tokenlist
	int W;		//number of Word int the whole tokenlist
	int NNZ;	//number of triple in the corpus
	int threshold=K; 
	int numOfWordS = 0;
	int numOfWordD = 0;

	docIdxStream >> D >> W >> NNZ;
	int numChunks = 4;
	int* chunkSizeVec= new int[numChunks];	//store length of tokenlist of chunks
	int chunkSizeMax = 0; //max of element in chunkSizeVec
	int* docIdStartVec = new int[numChunks];
	int* docIdEndVec= new int[numChunks];
	int* wordCountVec = new int[W];	//store the token count of each word	
	int* wordRankVec = new int[W];		// store the rank sorted by token count

	memset(wordCountVec, 0, W * sizeof(int));
	memset(wordRankVec, 0, W * sizeof(int));
	memset(chunkSizeVec, 0, numChunks * sizeof(int));

	int docStep = D / numChunks;
	for (int i = 0; i < numChunks; i++)
	{
		docIdStartVec[i] = i*docStep+1;
		docIdEndVec[i] =(D-(i + 1)*docStep>0)? (i + 1)*docStep:D;
	}

	maxLength chunkLength;
	chunkLength.docStep = docStep;
	chunkLength.W = W;

	string filePrefix = "/gpfs/alpine/proj-shared/csc289/lda/datasets/nytimes";
	printf("chunk partitioning ...\n");
	int j = 0;
	int tmpD=0, tmpW=0, tmpNNZ=0;
	for (int chunkId = 0; chunkId < numChunks; chunkId++)
	{
		ofstream docWordChunk((filePrefix + string(".chunk") + to_string(chunkId) + string(".txt")).c_str(), ios::binary);
		if (chunkId)
		{
			for (int i = 0; i < tmpNNZ; i++)
			{
				docWordChunk << tmpD << " " << tmpW << "\n";
			}
			wordCountVec[tmpW-1] += tmpNNZ;
			chunkSizeVec[chunkId] += tmpNNZ;
		}

		while (j < NNZ) {

			docIdxStream >> tmpD >> tmpW >> tmpNNZ;
		
			if (tmpD > docIdEndVec[chunkId])
			{
				docWordChunk.close();
				break;
			}
			wordCountVec[tmpW - 1] += tmpNNZ;
			chunkSizeVec[chunkId] += tmpNNZ;
			for (int i = 0; i < tmpNNZ; i++)
			{
				docWordChunk << tmpD << " " << tmpW << "\n";
			}
			//docWordChunk << tmpD << tmpW << tmpNNZ;
			j++;
		}
		
	}
	docIdxStream.close();

 

	for (int chunkId = 0; chunkId < numChunks; chunkId++) {
		chunkSizeMax = (chunkSizeMax > chunkSizeVec[chunkId]) ? chunkSizeMax : chunkSizeVec[chunkId];
	}
	for (int chunkId = 0; chunkId < numChunks; chunkId++) {
		printf("chunkId chunkSize %d %d\n",chunkId,chunkSizeVec[chunkId]);
	}	
	printf("chunkmax :%d\n",chunkSizeMax);
	
	chunkLength.maxTLLength = chunkSizeMax;


	//----generate wordId-wordRank pair--------//
	
	printf("generate wordId-wordRank pair ...\n");

	vector<indexSortStruct> wordId2Rank(W);

	for (int i = 0; i < W; i++) {
		wordId2Rank[i].tokenCount = wordCountVec[i];
		wordId2Rank[i].wordId = i+1;
		wordId2Rank[i].wordRank = 0;
	}
	stable_sort(wordId2Rank.begin(), wordId2Rank.end(), compareTokenCount);
	for (int i = 0; i < W; i++) {
		wordCountVec[i] = wordId2Rank[i].tokenCount;
		if (wordCountVec[i] >= threshold) numOfWordD +=1;
		wordId2Rank[i].wordRank = i+1;
	}
	numOfWordS = W - numOfWordD;


	stable_sort(wordId2Rank.begin(), wordId2Rank.end(), getWordRank);
	
	for (int i = 0; i < W; i++) {
		wordRankVec[i]=wordId2Rank[i].wordRank;
	}

	//---------------------------------------------


	//-------generate word-to-doc map---------

	printf("generate word-to-doc map and output data...\n");


	//********Tokenlist offset
	int* chunkWordCountVec = new int[W];
	int* chunkWordOffsetVec = new int[W];
	int* chunkDocCountVec = new int[docStep];
	int* chunkDocOffsetVec = new int[docStep];
	int* numOfTokenVecD = new int[numChunks];
	int* numOfTokenVecS = new int[numChunks];
	memset(numOfTokenVecD, 0, numChunks * sizeof(int));
	memset(numOfTokenVecS, 0, numChunks * sizeof(int));
	//*********Tokenlist offset






	//************DT matrix offset
	int* chunkDTCountVec = new int[docStep];
	int* chunkDTOffsetVec = new int[docStep];
	int* maxDTLengthVec = new int[numChunks];
	memset(maxDTLengthVec, 0, numChunks * sizeof(int));
	//*************DT matrix offset


	//***********WT matrix offset
	int* WTCountVec = new int[W];
	int* WTOffsetVec = new int[W];

	int* chunkWTCountVec = new int[numOfWordS];
	int* chunkWTOffsetVec = new int[numOfWordS];



	memset(WTCountVec, 0, W * sizeof(int));
	memset(WTOffsetVec, 0, W * sizeof(int));

	int* maxWTLengthVec = new int[numChunks];
	memset(maxWTLengthVec, 0, numChunks * sizeof(int));

	//***********WT matrix offset


	vector<chunkSortStruct> chunkVec(chunkSizeMax); //store the information of chunk
	vector<chunkMapSortStruct>chunkMapVec(chunkSizeMax);//store the information of chunk map

	string chunkFilePrefix = "/gpfs/alpine/proj-shared/csc289/lda/datasets/nytimesLargeK";// folder that store preprocessed chunks
	creatFolder(chunkFilePrefix);


	for (int chunkId = 0; chunkId < numChunks; chunkId++)
	{
		string chunkFolderName = chunkFilePrefix + "/chunk" + to_string(chunkId);
		
		creatFolder(chunkFolderName);

		ifstream docWordChunk((filePrefix + string(".chunk") + to_string(chunkId) + string(".txt")).c_str(), ios::binary);

		ofstream TL((chunkFolderName + string("/TL.txt")).c_str(), ios::binary);//Store TL and word2doc map

		ofstream word2DocMap((chunkFolderName + string("/word2DocMap.txt")).c_str(), ios::binary);//Store TL and word2doc map

		ofstream doc2WordMap((chunkFolderName + string("/doc2WordMap.txt")).c_str(), ios::binary);//Store TL and word2doc map



		ofstream wordCountOffset((chunkFolderName + string("/wordCountOffset.txt")).c_str(), ios::binary);//store Word offset of TL
		ofstream docCountOffset((chunkFolderName + string("/docCountOffset.txt")).c_str(), ios::binary);//store Doc offset of TL
		ofstream DTCountOffset((chunkFolderName + string("/DTCountOffset.txt")).c_str(), ios::binary);//store DT offset
	
		ofstream chunkWTCountOffset((chunkFolderName + string("/WTCountOffset.txt")).c_str(), ios::binary);//store DT offset


		memset(chunkDTCountVec, 0, docStep * sizeof(int));
		memset(chunkDTOffsetVec, 0, docStep * sizeof(int));

		memset(chunkWordCountVec, 0, W * sizeof(int));
		memset(chunkWordOffsetVec, 0, W * sizeof(int));

		memset(chunkDocCountVec, 0, docStep * sizeof(int));
		memset(chunkDocOffsetVec, 0, docStep * sizeof(int));

		memset(chunkWTCountVec, 0, numOfWordS * sizeof(int));
		memset(chunkWTOffsetVec, 0, numOfWordS * sizeof(int));


		for (int i = 0; i < chunkSizeVec[chunkId]; i++)
		{
			docWordChunk >> tmpD >> tmpW ;
			chunkVec[i].docId = tmpD;
			chunkVec[i].wordId = tmpW;
			chunkVec[i].wordRank = wordRankVec[tmpW - 1];
			chunkVec[i].mapDoc2Word = i;
			chunkVec[i].topic = rand() % (K)+1;
			chunkDocCountVec[tmpD - docIdStartVec[chunkId]] += 1;
		}
		docWordChunk.close();


		//**************compute docCountOffset and DTCountOffset
		
		for (int i = 0; i < docStep; i++)
		{
			chunkDTCountVec[i] = ((chunkDocCountVec[i] - 1) / 32 + 1) * 32;
			chunkDTCountVec[i] = (chunkDTCountVec[i] > K) ? K : chunkDTCountVec[i];
			maxDTLengthVec[chunkId] += chunkDTCountVec[i];		

		}

		computeOffset(chunkDocOffsetVec, chunkDocCountVec, docStep);
		computeOffset(chunkDTOffsetVec, chunkDTCountVec, docStep);

		for (int i = 0; i < docStep; i++)
		{

			docCountOffset << chunkDocCountVec[i] << " " << chunkDocOffsetVec[i] << "\n";
			DTCountOffset << chunkDTCountVec[i] << " " << chunkDTOffsetVec[i] << "\n";

		}

		//**************compute docCountOffset and DTCountOffset


		stable_sort(chunkVec.begin(), chunkVec.begin()+ chunkSizeVec[chunkId], sortWordRank);// sort by word rank


		//**************sort to get word2doc map and write TLMap and wordCountOffset
		for (int i = 0; i < chunkSizeVec[chunkId]; i++)
		{

			doc2WordMap << chunkVec[i].docId-chunkId*docStep << "\n";
			chunkMapVec[i].mapDoc2Word = chunkVec[i].mapDoc2Word;
			chunkMapVec[i].mapWord2Doc = i;
			chunkWordCountVec[chunkVec[i].wordRank - 1] += 1;
		}

		for (int i = 0; i < W; i++)
		{
			if (i < numOfWordD)
			{
				numOfTokenVecD[chunkId] += chunkWordCountVec[i];
			}
			else
			{
				numOfTokenVecS[chunkId] += chunkWordCountVec[i];
				chunkWTCountVec[i - numOfWordD] = (chunkWordCountVec[i] > K) ? K : chunkWordCountVec[i];
				maxWTLengthVec[chunkId] += chunkWTCountVec[i - numOfWordD];
			}

		}

		computeOffset(chunkWTOffsetVec, chunkWTCountVec, numOfWordS);

		for (int i = 0; i < numOfWordS; i++) {

			chunkWTCountOffset << chunkWTCountVec[i] << " " << chunkWTOffsetVec[i] << "\n";

		}


		stable_sort(chunkMapVec.begin(), chunkMapVec.begin() + chunkSizeVec[chunkId], sortChunkMap);

		for (int i = 0; i < chunkSizeVec[chunkId]; i++) {
			TL << chunkVec[i].topic << "\n";
			word2DocMap	<<chunkMapVec[i].mapWord2Doc << "\n";
		}

		computeOffset(chunkWordOffsetVec, chunkWordCountVec, W);
		for (int i = 0; i < W; i++)
		{
			WTCountVec[i] += chunkWordCountVec[i];
			wordCountOffset << chunkWordCountVec[i] << " " << chunkWordOffsetVec[i] << "\n";
		}

		//**************sort to get word2doc map and write TLMap and wordCountOffset

		TL.close();
		word2DocMap.close();
		doc2WordMap.close();
		wordCountOffset.close();
		docCountOffset.close();
		DTCountOffset.close();
		chunkWTCountOffset.close();

	}


	//***********Compute WT Count and Offset
	ofstream WTCountOffset((chunkFilePrefix + string("/WTCountOffset.txt")).c_str(), ios::binary);//store WT offset
	for (int i = 0; i < W; i++) {
		WTCountVec[i] = (WTCountVec[i] > K) ? K : WTCountVec[i];


	}
	computeOffset(WTOffsetVec, WTCountVec, W);
	chunkLength.maxWTLength = WTOffsetVec[W - 1] + WTCountVec[W - 1]- WTOffsetVec[numOfWordD];
	for (int i = 0; i < W; i++) {
		WTCountOffset << WTCountVec[i] << " " << WTOffsetVec[i] << "\n";
	}
	WTCountOffset.close();
	//***********Compute WT Count and Offset


	//***********Compute max Doc and DT length
	ofstream TLLength((chunkFilePrefix + string("/TLLength.txt")).c_str(), ios::binary);//store max Doc and DT length

	ofstream DTLength((chunkFilePrefix + string("/DTLength.txt")).c_str(), ios::binary);//store max Doc and DT length

	ofstream docLength((chunkFilePrefix + string("/docLength.txt")).c_str(), ios::binary);//store max Doc and DT length

	ofstream WTLength((chunkFilePrefix + string("/WTLength.txt")).c_str(), ios::binary);//store max Doc and DT length


	ofstream TLSplit((chunkFilePrefix + string("/TLSplit.txt")).c_str(), ios::binary);


	chunkLength.maxDTLength = 0;
	chunkLength.maxChunkWTLength = 0;
	for (int chunkId = 0; chunkId < numChunks; chunkId++)
	{
		docLength << (docIdEndVec[chunkId]-docIdStartVec[chunkId]+1) << "\n";
		TLLength << chunkSizeVec[chunkId] << "\n";
		DTLength << maxDTLengthVec[chunkId] << "\n";
		WTLength << maxWTLengthVec[chunkId] << "\n";
		TLSplit << numOfTokenVecD[chunkId] << " " << numOfTokenVecS[chunkId] << "\n";

		chunkLength.maxDTLength = (chunkLength.maxDTLength > maxDTLengthVec[chunkId]) ? chunkLength.maxDTLength : maxDTLengthVec[chunkId];
		chunkLength.maxChunkWTLength = (chunkLength.maxChunkWTLength > maxWTLengthVec[chunkId]) ? chunkLength.maxChunkWTLength : maxWTLengthVec[chunkId];

	}
	docLength.close();
	TLLength.close();
	DTLength.close();
	WTLength.close();
	TLSplit.close();
	//***********Compute max Doc and DT length

	//********** Output length struct
	ofstream lengthVec((chunkFilePrefix + string("/lengthVec.txt")).c_str(), ios::binary);//store max Doc and DT length

	lengthVec << chunkLength.maxTLLength << " " << chunkLength.maxDTLength << " " << chunkLength.maxWTLength << " " << chunkLength.docStep << " " << chunkLength.W<< " " << chunkLength.maxChunkWTLength << " " << numOfWordD << " " << numOfWordS;
	lengthVec.close();
	//********** Output length struct

	printf("done! \n");
	//-----------------------------------------------

	
}




void creatFolder(string argFolderName) {
	if (access(argFolderName.c_str(), 0) == -1)
	{
		cout << argFolderName << " not exist!" << endl;
		cout << "now creat it!" << endl;
		int flag = mkdir(argFolderName.c_str(),0744);
		if (flag == 0)
		{
			cout << "make successfully" << endl;
		}
		else {
			cout << "make errorly" << endl;
		}
	}
}


void computeOffset(int* argOffset, int* argCount, int argVecLength)

{
	int temp = 0;
	for (int i = 0; i < argVecLength; i++) {
		argOffset[i] = temp;
		temp += argCount[i];
	}
}


