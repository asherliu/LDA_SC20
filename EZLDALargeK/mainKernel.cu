
#ifndef _MAINKERNEL_H_
#define _MAINKERNEL_H_

#include "DTUpdateKernel.cuh"
#include "WTUpdateKernel.cuh"
#include "WTDenUpdateKernel.cuh"
#include "WTAddKernel.cuh"
#include "SamplingKernel.cuh"
#include "WTDense.cuh"
using namespace std;

void fileErrorCheck(ifstream& argFileStream, string folderName)
{
	if (!argFileStream.is_open())
	{
		cout << "File " << folderName << " open failed" << endl;
		exit(0);
	}
}

int main(int argc, char *argv[]) {

	clock_t startTime, startTime1,endTime;
	double WTTime=0.0;
	double samplingTime=0.0;
	double DTTime=0.0;
	double totalTime=0.0;

	int maxTLLength;
	int maxDTLength;
	int maxWTLength;
	int maxDocLength;
	int wordLength;
	int maxChunkWTLength;
	int numOfWordD;
	int numOfWordS;
	int numChunks = 4;
	int numIters = 300;

	string chunkFilePrefix = "/gpfs/alpine/proj-shared/csc289/lda/datasets/nytimesLargeK";// folder that store preprocessed chunks

	ifstream lengthVec((chunkFilePrefix + string("/lengthVec.txt")).c_str(), ios::binary);//store max Doc and DT length

	ofstream timeRecord((chunkFilePrefix + string("/timeRecord.txt")).c_str(), ios::binary);

	fileErrorCheck(lengthVec, "/lengthVec.txt");

	lengthVec >> maxTLLength >> maxDTLength >> maxWTLength >> maxDocLength >> wordLength>>maxChunkWTLength>> numOfWordD>> numOfWordS;
	lengthVec.close();

	Document document(chunkFilePrefix,numChunks,maxTLLength,maxDocLength,wordLength);

	document.loadDocument();
	document.GPUMemAllocate();
	

	DTChunk chunkDT(maxDTLength,maxDocLength,numChunks);
	chunkDT.loadDocDTLength(chunkFilePrefix);
	chunkDT.CPUMemSet();
	chunkDT.GPUMemAllocate();	
	chunkDT.loadDTCountOffset(chunkFilePrefix);
	WTD WTDen(numOfWordD, wordLength);
	WTDen.GPUMemAllocate();
	WTDen.GPUMemInit();


	
	WTAll WT(maxWTLength, wordLength, numChunks, maxChunkWTLength,numOfWordS);




	WT.CPUMemSet();
	WT.GPUMemAllocate();
	WT.GPUMemset();
	WT.loadWTLength(chunkFilePrefix);
	WT.loadWTCountOffset(chunkFilePrefix);
	WT.blockWarpCountCPU2GPU();
	WT.CPU2GPUCountOffset();

	printf("Total memory usage : %f GB\n", document.TLMemory + WT.WTMemory + chunkDT.DTMemory);

	for (int chunkId = 0; chunkId < numChunks; chunkId++) {
		document.CPU2GPU(chunkId);
		WT.chunkCPU2GPUCountOffset(chunkId);
		WT.chunkGPUMemset();
		//--------------update WTDen matrix ---------
		UpdateWTDenKernel(WTDen, WT, document, chunkId);
		//--------------update WTDen matrix-----------

		//--------------update WT matrix--------
		
		//WT.chunkCPU2GPUCountOffset(chunkId);
		//WT.chunkGPUMemset();
		UpdateWTKernel(WT, document,chunkId);
		//WT.chunkWTGPU2CPU(chunkId);// marker
		//
		//WT.CPU2DiskChunk(chunkFilePrefix, chunkId);// marker
		/*printf("\n what's this %d\n", chunkId);*/
		//--------------update WT matrix-----------

	}
	/*WTDen.WTDenGPU2CPU();
	WTDen.WTDenCPU2Disk(chunkFilePrefix);*/
	printf("WT ended!\n");

	//WT.CPU2GPUCountOffset();
	startTime = clock();
	for (int iter = 0; iter < numIters; iter++) {


		startTime1=clock();
		//printf("chunk WT updated!\n");
		WT.GPUMemset();
		//--------------update WTDenSum -----------
		UpdateWTDenRowSumKernel(WTDen,WT);
		//--------------update WTDenSum -----------

		//--------------update WTSum -----------
		for (int chunkId = 0; chunkId < numChunks; chunkId++) {
			WT.chunkCPU2GPUCountOffset(chunkId);
			WT.chunkGPUMemset();
			WT.chunkWTCPU2GPU(chunkId);
			WTAdditionKernel(WT, document);
		}
		//--------------update WTSum -----------
		//WT.WTGPU2CPU();// marker
		//WT.CPU2Disk(chunkFilePrefix);// marker






		printf("WT updated!\n");
		endTime = clock();
		WTTime+=(double)(endTime-startTime1)/CLOCKS_PER_SEC;


		document.CPU2GPUPerplexity();
		
		for (int chunkId = 0; chunkId < numChunks; chunkId++) {
			
			startTime1=clock();
			/*printf("step: %d\n",chunkId);*/
			//--------------update DT matrix-----------
			document.CPU2GPU(chunkId);
			/*printf("%d\n", 1);*/
			chunkDT.GPUMemSet(chunkId);
			/*printf("%d\n", 2);*/
			chunkDT.CPU2GPUDTCountOffset(chunkId);
			/*printf("%d\n", 3);*/
			//chunkDT.CPU2GPU(chunkId, document.docLengthVec[chunkId]);
			UpdateDTKernel(chunkDT, document);
			/*printf("%d\n", 4);*/
			//chunkDT.GPU2CPU(chunkId);
			//chunkDT.CPU2Disk(chunkFilePrefix, chunkId);// marker
			
			//--------------update DT matrix-----------
			endTime = clock();
			/*printf("%d\n", 5);*/
			DTTime+=(double)(endTime-startTime1)/CLOCKS_PER_SEC;

			

			startTime1=clock();
			//--------------sampling-----------
			/*printf("%d\n", 6);*/
			SampleKernelD(WTDen, WT, chunkDT, document);
			/*printf("%d\n", 7);*/
			//WTDen.WTDenGPU2CPU();// marker
			//WTDen.WTDenCPU2Disk(chunkFilePrefix);// marker
			SampleKernel(WT, chunkDT, document);
			/*printf("%d\n", 8);*/
			//WT.WTGPU2CPU();// marker
			//WT.CPU2Disk(chunkFilePrefix);// marker
			document.GPU2CPU(chunkId);
			//--------------sampling-----------

			endTime = clock();
			samplingTime+=(double)(endTime-startTime1)/CLOCKS_PER_SEC;

			startTime1=clock();
			//--------------update chunkWT matrix-----------
			WT.chunkCPU2GPUCountOffset(chunkId);
			WT.chunkGPUMemset();
			UpdateWTKernel(WT, document, chunkId);
			WT.chunkWTGPU2CPU(chunkId);
			//WT.CPU2DiskChunk(chunkFilePrefix, chunkId);
			//--------------update chunkWT matrix-----------
			endTime = clock();
			WTTime+=(double)(endTime-startTime1)/CLOCKS_PER_SEC;
		}
		WTDen.GPUMemCopy();
		WTDen.GPUMemset();

		printf("done!!!!!");
		document.GPU2CPUPerplexity();



		endTime = clock();
		totalTime=(double)(endTime-startTime)/CLOCKS_PER_SEC;
		timeRecord << WTTime << " " << DTTime << " " << samplingTime << " " << totalTime << " " << document.sumPerplexity<< "\n";
		printf("WTTime: %f, DTTime: %f, samplingTime:%f, totalTime:%f, perplexity:%f\n",WTTime,DTTime,samplingTime,totalTime,document.sumPerplexity);

	}
	
	timeRecord.close();
}
#endif