#include"DataChunk.cuh"

DocChunk::DocChunk(int argTLLength, int argDocLength, int argWordLength) {

	TLLength = argTLLength;
	docLength = argDocLength;
	wordLength = argWordLength;


}

void DocChunk::CPUMemSet() {

	cudaMallocHost((void**)&TLTopic, TLLength* sizeof(unsigned short int));
	cudaMallocHost((void**)&TLMaxTopic, TLLength * sizeof(unsigned short int));

	cudaMallocHost((void**)&TLDocCount, docLength * sizeof(int));
	cudaMallocHost((void**)&TLDocOffset, docLength * sizeof(int));

	cudaMallocHost((void**)&TLWordCount, wordLength * sizeof(int));
	cudaMallocHost((void**)&TLWordOffset, wordLength * sizeof(int));

	cudaMallocHost((void**)&mapWord2Doc, TLLength * sizeof(int));
	cudaMallocHost((void**)&mapDoc2Word, TLLength * sizeof(int));

	/*TLTopic = new unsigned short int[TLLength];
	TLMaxTopic = new unsigned short int[TLLength];

	TLDocCount = new int[docLength];
	TLDocOffset = new int[docLength];
	TLWordCount = new int[wordLength];
	TLWordOffset = new int[wordLength];
	mapWord2Doc = new int[TLLength];
	mapDoc2Word = new int[TLLength];*/

	memset(TLTopic, 0, TLLength * sizeof(unsigned short int));
	memset(TLMaxTopic, 0, TLLength * sizeof(unsigned short int));

	memset(TLDocCount, 0, docLength * sizeof(int));
	memset(TLDocOffset, 0, docLength * sizeof(int));
	memset(TLWordCount, 0, wordLength * sizeof(int));
	memset(TLWordOffset, 0, wordLength * sizeof(int));
	memset(mapWord2Doc, 0, TLLength * sizeof(int));
	memset(mapDoc2Word, 0, TLLength * sizeof(int));
}


void DocChunk::loadChunk(string argFilePrefix, int argChunkId) {


	chunkId = argChunkId;
	string chunkFolderName = argFilePrefix + "/chunk" + to_string(chunkId);
	CPUMemSet();
	printf("loading chunk %d ...\n", chunkId);

	ifstream TL((chunkFolderName + string("/TL.txt")).c_str(), ios::binary);//Store TL and word2doc map

	ifstream word2DocMap((chunkFolderName + string("/word2DocMap.txt")).c_str(), ios::binary);//Store TL and word2doc map

	ifstream doc2WordMap((chunkFolderName + string("/doc2WordMap.txt")).c_str(), ios::binary);//Store TL and word2doc map



	for (int i = 0; i < TLLength; i++) {

		TL >> TLTopic[i];
		word2DocMap >> mapWord2Doc[i];
		doc2WordMap >> mapDoc2Word[i];
		TLMaxTopic[i] = TLTopic[i];
	}
	TL.close();
	word2DocMap.close();
	doc2WordMap.close();

	ifstream wordCountOffset((chunkFolderName + string("/wordCountOffset.txt")).c_str(), ios::binary);//store Word offset of TL

	for (int i = 0; i < wordLength; i++)
	{
		wordCountOffset >> TLWordCount[i] >> TLWordOffset[i];
	}
	wordCountOffset.close();

	ifstream docCountOffset((chunkFolderName + string("/docCountOffset.txt")).c_str(), ios::binary);//store Doc offset of TL and DT offset

	for (int i = 0; i < docLength; i++)
	{
		docCountOffset >> TLDocCount[i] >> TLDocOffset[i];
	}
	docCountOffset.close();
	printf("chunk %d loaded!...\n", chunkId);

}
