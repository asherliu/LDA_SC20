
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
	int numChunks = 3;
	int numIters = 200;
	const int numGPUs = 3;
	//int numChunksPerGPU = numChunks / numGPUs;

	string chunkFilePrefix = "C:/shilong/LDA/DSLDACode/datasetDS/docword_kos";// folder that store preprocessed chunks

	ifstream lengthVec((chunkFilePrefix + string("/lengthVec.txt")).c_str(), ios::binary);//store max Doc and DT length

	ofstream timeRecord((chunkFilePrefix + string("/timeRecord.txt")).c_str(), ios::binary);

	fileErrorCheck(lengthVec, "/lengthVec.txt");

	lengthVec >> maxTLLength >> maxDTLength >> maxWTLength >> maxDocLength >> wordLength>>maxChunkWTLength>> numOfWordD>> numOfWordS;
	lengthVec.close();

	Document document(chunkFilePrefix,numChunks,maxTLLength,maxDocLength,wordLength,numGPUs);

	document.loadDocument();
	document.InitGPU();

	cudaStream_t stream[numGPUs];
	for (int i = 0; i < numGPUs; i++) {
		cudaSetDevice(i);
		cudaStreamCreate(&stream[i]);
	}



	DTChunk chunkDT(maxDTLength, maxDocLength, numChunks,numGPUs);
	chunkDT.loadDocDTLength(chunkFilePrefix);// load DT and Doc length from disk to cpu
	chunkDT.loadDTCountOffset(chunkFilePrefix);// load DT count and offset from disk to cpu
	chunkDT.InitDTGPU();// allocate DT and Count and offset ; Init DT 




	//chunkDT.CPUMemSet();
	/*for (int i = 0; i < numGPUs; i++) {
		chunkDT.GPUMemAllocate(i);
	}*/
	
	
	WTD WTDen(numOfWordD, wordLength,numGPUs);
	WTDen.InitWTGPU();
	WTDen.GPUMemAllocate();
	//for (int i = 0; i < numGPUs; i++) {
	//	WTDen.GPUMemAllocate(i);
	//	WTDen.GPUMemInit();//may occur bug
	//}
	


	
	WTAll WT(maxWTLength, wordLength, numChunks, maxChunkWTLength,numOfWordS,numGPUs);
	WT.loadWTLength(chunkFilePrefix);
	WT.loadWTCountOffset(chunkFilePrefix);
	WT.InitWTGPU();
	WT.GPUMemAllocate();
	/*for (int i = 0; i < numGPUs; i++) {
		WT.GPUMemAllocate(i);
		WT.GPUMemset();
		WT.blockWarpCountCPU2GPU();
		WT.CPU2GPUCountOffset();
	}	*/

	/*WT.CPUMemSet();*/


	curandState* randState[numGPUs];
	srand(time(NULL));
	for (int i = 0; i < numGPUs; i++) {
		cudaSetDevice(i);
		cudaMalloc(&randState[i], sizeof(curandState)*GridDim*BlockDim);//may have bugs
	}



	H_ERR(cudaDeviceSynchronize());

	printf("Total memory usage : %f GB\n", document.TLMemory + WT.WTMemory + chunkDT.DTMemory);


	
	for (int chunkId = 0; chunkId < numChunks; chunkId++) {
		document.CPU2GPU(chunkId,chunkId);
		WT.chunkCPU2GPUCountOffset(chunkId);
		WT.CPU2GPUCountOffset(chunkId);
		H_ERR(cudaDeviceSynchronize());
		//WT.WTGPUChunkVec[chunkId].chunkGPUMemset(chunkId);
		//--------------update WTDen matrix ---------
		printf("1");
		UpdateWTDenKernel(WTDen, WT, document, chunkId, chunkId, stream[chunkId]);
		H_ERR(cudaDeviceSynchronize());
		printf("2");
		//--------------update WTDen matrix-----------

		//--------------update WT matrix--------
		
		//WT.chunkCPU2GPUCountOffset(chunkId);
		//WT.chunkGPUMemset();
		UpdateWTKernel(WT, document,chunkId, chunkId, stream[chunkId]);
		H_ERR(cudaDeviceSynchronize());
		//WT.chunkWTGPU2CPU(chunkId);// marker
		//
		//WT.CPU2DiskChunk(chunkFilePrefix, chunkId);// marker
		/*printf("\n what's this %d\n", chunkId);*/
		//--------------update WT matrix-----------

	}
	printf("3");
	H_ERR(cudaDeviceSynchronize());



	for (int chunkId = 0; chunkId < numChunks; chunkId++) {
		printf("4");
		chunkDT.CPU2GPUDTCountOffset(chunkId);
	}

	H_ERR(cudaDeviceSynchronize());

	/*printf("3");*/


	///*WTDen.WTDenGPU2CPU();
	//WTDen.WTDenCPU2Disk(chunkFilePrefix);*/
	//printf("WT ended!\n");

	////WT.CPU2GPUCountOffset();
	//startTime = clock();
	for (int iter = 0; iter < numIters; iter++) {


			
		/*startTime1=clock();*/
		
		// MemorySet WTDenCopy
	/*	for (int chunkId = 0; chunkId < numChunks; chunkId++) {
			WTDen.WTDenseGPUChunkVec[chunkId].GPUMemset(chunkId);
			
		}*/

		//--------------MemSet WT and WTRowSum -----------
		for (int chunkId = 0; chunkId < numChunks; chunkId++) {
			WT.WTGPUChunkVec[chunkId].GPUMemset(chunkId);
			H_ERR(cudaDeviceSynchronize());
			WTDen.WTDenseGPUChunkVec[chunkId].GPUMemsetWTDense(chunkId);
			H_ERR(cudaDeviceSynchronize());
		}
		//--------------MemSet WT and WTRowSum -----------

		H_ERR(cudaDeviceSynchronize());



		for (int chunkId = 0; chunkId < numChunks; chunkId++) {
			/*WT.GPUDataTransfer(chunkId, stream[0]);*/
			WT.GPUDataTransferBackCPU(chunkId);
			H_ERR(cudaDeviceSynchronize());
			WT.GPUDataTransferToGPU(chunkId);
			H_ERR(cudaDeviceSynchronize());
			/*WTDen.GPUDataTransfer(chunkId, stream[0]);*/
			WTDen.GPUDataTransferBackCPU(chunkId);
			H_ERR(cudaDeviceSynchronize());
			WTDen.GPUDataTransferToGPU(chunkId);
			H_ERR(cudaDeviceSynchronize());
			printf("\nchunkID:%d\n", chunkId);
			WTAdditionKernel(WT, document, stream[0]);
			H_ERR(cudaDeviceSynchronize());
			printf("\nchunkID:%d\n", chunkId);
			WTDenAdditionKernel(WTDen, WT, document, stream[0]);
			H_ERR(cudaDeviceSynchronize());
		}

		
		



		//for (int chunkId = 0; chunkId < 3; chunkId++) {
		//	WT.GPUDataTransfer(chunkId, stream[0]);
		//	WTDen.GPUDataTransfer(chunkId, stream[0]);
		//	
		//	H_ERR(cudaDeviceSynchronize());
		//	printf("\nchunkID:%d\n", chunkId);
		//	WTAdditionKernel(WT, document, stream[0]);
		//	H_ERR(cudaDeviceSynchronize());
		//	printf("\nchunkID:%d\n", chunkId);
		//	WTDenAdditionKernel(WTDen, WT, document, stream[0]);

		//}
		H_ERR(cudaDeviceSynchronize());
		printf("5");


		UpdateWTDenRowSumKernel(WTDen, WT, 0, stream[0]);
		H_ERR(cudaDeviceSynchronize());
		WT.verifyWTSum();
		H_ERR(cudaDeviceSynchronize());

		for (int chunkId = 0; chunkId < numChunks; chunkId++) {
			WT.GPUDataDistributeBackCPU(chunkId, stream[0]);
			H_ERR(cudaDeviceSynchronize());
			WT.GPUDataDistributeToGPU(chunkId, stream[0]);
			H_ERR(cudaDeviceSynchronize());
			WTDen.GPUDataDistributeBackCPU(chunkId);
			H_ERR(cudaDeviceSynchronize());
			WTDen.GPUDataDistributeToGPU(chunkId);
			H_ERR(cudaDeviceSynchronize());


			/*WTDen.GPUDataDistribute(chunkId, stream[0]);*/


			//WT.GPUDataDistribute(chunkId, stream[0]);
		}
		printf("6");
		H_ERR(cudaDeviceSynchronize());


		//--------------MemSet WTDenseCopy and chunkWT -----------
		for (int chunkId = 0; chunkId < numChunks; chunkId++) {
			WTDen.WTDenseGPUChunkVec[chunkId].GPUMemsetWTDenseCopy(chunkId);
			H_ERR(cudaDeviceSynchronize());
			WT.WTGPUChunkVec[chunkId].chunkGPUMemset(chunkId);
			H_ERR(cudaDeviceSynchronize());
		}
		//--------------MemSet WTDenseCopy and chunkWT---------- -
		

		printf("7");
		H_ERR(cudaDeviceSynchronize());


		for (int chunkId = 0; chunkId < numChunks; chunkId++)
		{
			document.CPU2GPUPerplexity(chunkId);
			H_ERR(cudaDeviceSynchronize());
			chunkDT.DTGPUChunkVec[chunkId].GPUMemSet(chunkId);
			H_ERR(cudaDeviceSynchronize());
		}
		printf("8");
		H_ERR(cudaDeviceSynchronize());





		/*for (int chunkId = 0; chunkId < numChunks; chunkId++) {

			chunkDT.DTGPUChunkVec[chunkId].GPUMemSet(chunkId);
		}
		printf("9");
		H_ERR(cudaDeviceSynchronize());*/


		

		for (int chunkId = 0; chunkId < numChunks; chunkId++) {


			UpdateDTKernel(chunkDT, document, chunkId, stream[chunkId]);
			H_ERR(cudaDeviceSynchronize());
		
			SampleKernelD(WTDen, WT, chunkDT, document, randState[chunkId], chunkId, chunkId, stream[chunkId]);
			H_ERR(cudaDeviceSynchronize());
			SampleKernel(WT, chunkDT, document, randState[chunkId], chunkId, chunkId, stream[chunkId]);
			H_ERR(cudaDeviceSynchronize());
			UpdateWTKernel(WT, document, chunkId, chunkId, stream[chunkId]);
			H_ERR(cudaDeviceSynchronize());

		}
		H_ERR(cudaDeviceSynchronize());

		for (int chunkId = 0; chunkId < numChunks; chunkId++)
		{
			document.GPU2CPUPerplexity(chunkId);
			
		}
		H_ERR(cudaDeviceSynchronize());


	


		}
		
		
	
	
}
#endif
