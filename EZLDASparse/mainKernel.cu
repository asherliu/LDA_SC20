
#ifndef _MAINKERNEL_H_
#define _MAINKERNEL_H_
#define HAVE_STRUCT_TIMESPEC

#include "DTUpdateKernel.cuh"
#include "WTUpdateKernel.cuh"
#include "WTDenUpdateKernel.cuh"
#include "WTAddKernel.cuh"
#include "SamplingKernel.cuh"
#include "WTDense.cuh"
#include "Argument.cuh"
#include <stdio.h>
#include <stdlib.h>
//#include <pthread.h>
#include <thread>

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
	double transferTimeCPU2GPU=0.0;
	double transferTimeGPU2CPU=0.0;
	double WTTime=0.0;
	double samplingTimeD=0.0;
	double samplingTimeS=0.0;
	double DTTime=0.0;
	double totalTime=0.0;
	double UpdateMTime=0.0;

	int maxTLLength;
	int maxDTLength;
	int maxWTLength;
	int maxDocLength;
	int wordLength;
	int maxChunkWTLength;
	int numOfWordD;
	int numOfWordS;
	int numChunks = 16;
	int numIters = 300;

	int chunksPerStream = numChunks / numStreams;

	string chunkFilePrefix = "/gpfs/alpine/proj-shared/csc289/lda/datasets/data200k";

	
	ofstream SamplingDRecord((chunkFilePrefix + string("/SamplingDRecord.txt")).c_str(), ios::binary);

	ifstream lengthVec((chunkFilePrefix + string("/lengthVec.txt")).c_str(), ios::binary);//store max Doc and DT length

	ofstream timeRecord((chunkFilePrefix + string("/timeRecord.txt")).c_str(), ios::binary);
	ofstream maxPercentRecord((chunkFilePrefix + string("/maxPercentRecord.txt")).c_str(), ios::binary);

	ofstream warpTimeRecord((chunkFilePrefix + string("/warpTimeRecord.txt")).c_str(), ios::binary);

	fileErrorCheck(lengthVec, "/lengthVec.txt");

	lengthVec >> maxTLLength >> maxDTLength >> maxWTLength >> maxDocLength >> wordLength>>maxChunkWTLength>> numOfWordD>> numOfWordS;
	lengthVec.close();

	Document document(chunkFilePrefix,numChunks,maxTLLength,maxDocLength,wordLength);


	//Document doc[2] = {Document(chunkFilePrefix,numChunks,maxTLLength,maxDocLength,wordLength), Document(chunkFilePrefix,numChunks,maxTLLength,maxDocLength,wordLength)};
	//doc[1].loadDocument();
	//doc[1].GPUMemAllocate();
	//doc[1].deviceCounterMemAllocate();
	//doc[2].loadDocument();
	//doc[2].GPUMemAllocate();
	//doc[2].deviceCounterMemAllocate();

	document.loadDocument();
	document.GPUMemAllocate();
	document.deviceCounterMemAllocate();
	H_ERR(cudaDeviceSynchronize());

	DTChunk chunkDT(maxDTLength,maxDocLength,numChunks);
	chunkDT.loadDocDTLength(chunkFilePrefix);
	chunkDT.CPUMemSet();
	chunkDT.GPUMemAllocate();	
	chunkDT.loadDTCountOffset(chunkFilePrefix);
//	WTD WTDen(numOfWordD, wordLength);
//	WTDen.GPUMemAllocate();
//	WTDen.GPUMemInit();

	curandState* randState[numStreams];
	cudaStream_t streams[numStreams];
	cudaStream_t syncStream;
	cudaStreamCreateWithPriority(&syncStream, cudaStreamDefault,0);
	cudaEvent_t stopEvents[numStreams];
	cudaEvent_t startEvent;
	cudaStreamCreate(&syncStream);
	cudaEventCreate(&startEvent);
	for (int i = 0; i < numStreams; i++)
	{
		cudaStreamCreateWithPriority(&streams[i], cudaStreamDefault, i);
		cudaEventCreate(&stopEvents[i]);
		cudaMalloc(&randState[i], sizeof(curandState)*GridDim*BlockDim);
	}

	int clockRate;
	cudaDeviceGetAttribute(&clockRate, cudaDevAttrClockRate, 0);

	printf("clockRate:%d\n", clockRate);
	WTAll WT(maxWTLength+ numOfWordD*K, wordLength, numChunks, maxChunkWTLength,wordLength);

	WT.CPUMemSet();
	WT.GPUMemAllocate();
	WT.GPUMemset(streams[0]);
	//WT.loadWTLength(chunkFilePrefix);
	WT.loadWTCountOffset(chunkFilePrefix);
	/*WT.blockWarpCountCPU2GPU();*/
	WT.CPU2GPUCountOffset(streams[0]);
	srand(time(NULL));
	float iterTime=0.0;
	cudaEvent_t iterStart, iterStop;
	cudaEventCreate(&iterStart);
	cudaEventCreate(&iterStop);

	H_ERR(cudaDeviceSynchronize());

	printf("Total memory usage : %f GB\n", document.TLMemory + WT.WTMemory + chunkDT.DTMemory);

	/*for (int chunkId = 0; chunkId < numChunks; chunkId++)*/
//	for (int batchId = 0; batchId < chunksPerStream; batchId++)
//	{
//		for (int streamId = 0; streamId < numStreams; streamId++)
//		{
//			int chunkId = batchId*numStreams + streamId;
//			document.CPU2GPU(chunkId, streamId, streams[0]);
////			WT.chunkCPU2GPUCountOffset(chunkId, streamId, streams[0]);
////			WT.chunkGPUMemset(streamId, streams[0]);
////			//--------------update WTDen matrix ---------
////			UpdateWTDenKernel(WTDen, WT, document, chunkId, streamId, streams[0]);
//			//--------------update WTDen matrix-----------
//
//			//--------------update WT matrix--------
//			UpdateWTKernel(WT, document, chunkId, streamId, streams[0]);
////			WT.chunkWTGPU2CPU(chunkId, streamId, streams[0]);// marker
//			//--------------update WT matrix-----------
//		}
//	}
//	H_ERR(cudaDeviceSynchronize());
	printf("WT ended!\n");

	//WT.CPU2GPUCountOffset();
	startTime = clock();
	/*pthread_t thread[numStreams];*/
	thread threadBlock[numStreams];
	for (int iter = 0; iter < numIters; iter++) {

		cudaEventRecord(iterStart,streams[1]);
		startTime1=clock();
		//printf("chunk WT updated!\n");
		WT.GPUMemset(streams[1]);
		for (int batchId = 0; batchId < chunksPerStream; batchId++) {
			for (int streamId = 0; streamId < numStreams; streamId++)
			{
				int chunkId = batchId*numStreams + streamId;
				document.CPU2GPU(chunkId, streamId, streams[1]);
	//			WT.chunkCPU2GPUCountOffset(chunkId, streamId, streams[0]);
	//			WT.chunkGPUMemset(streamId, streams[0]);
	//			//--------------update WTDen matrix ---------
	//			UpdateWTDenKernel(WTDen, WT, document, chunkId, streamId, streams[0]);
				//--------------update WTDen matrix-----------

				//--------------update WT matrix--------
				UpdateWTKernel(WT, document, chunkId, streamId, streams[1]);
	//			WT.chunkWTGPU2CPU(chunkId, streamId, streams[0]);// marker
				//--------------update WT matrix-----------
			}
		}






		//--------------update WTDenSum -----------
		//UpdateWTDenRowSumKernel(WTDen,WT, streams[1]);
		UpdateWTRowSumKernel(WT, streams[1]);
		//--------------update WTDenSum -----------
		/*H_ERR(cudaDeviceSynchronize());*/
		//--------------update WTSum -----------

		//for (int chunkId = 0; chunkId < numChunks; chunkId++) 
//		for (int batchId = 0; batchId < chunksPerStream; batchId++) {
//			for (int streamId = 0; streamId < numStreams; streamId++)
//			{
//				int chunkId = batchId*numStreams + streamId;
//				WT.chunkCPU2GPUCountOffset(chunkId, streamId, streams[1]);
//				WT.chunkGPUMemset(streamId, streams[1]);
//				WT.chunkWTCPU2GPU(chunkId, streamId, streams[1]);
//				WTAdditionKernel(WT, document, streamId, streams[1]);
//			}
//		}
		//--------------update WTSum -----------
		//WT.WTGPU2CPU();// marker
		//WT.CPU2Disk(chunkFilePrefix);// marker

		printf("WT updated!\n");
		endTime = clock();
		WTTime+=(double)(endTime-startTime1)/CLOCKS_PER_SEC;

		document.CPU2GPUPerplexity(streams[1]);
		cudaEventRecord(startEvent, streams[1]);

		/*samplingTimeD=0;*/
		/*cudaDeviceSynchronize();*/
		
		

		
		for (int batchId = 0; batchId < chunksPerStream; batchId++) {

			for (int streamId = 0; streamId < numStreams; streamId++) {
				int chunkId = batchId*numStreams + streamId;
				printf("step: %d\n", chunkId);
				//--------------update DT matrix-----------

				/*int startTime1 = clock();*/

				//int endTime = clock();
				//int transferTimeCPU2GPU += (double)(endTime - startTime1) / CLOCKS_PER_SEC;
				

				printf("%d\n", 1);
				chunkDT.GPUMemSet(chunkId, streamId, streams[streamId]);
				printf("%d\n", 2);
				chunkDT.CPU2GPUDTCountOffset(chunkId, streamId, streams[streamId]);
				printf("%d\n", 3);
				
				if ((chunkId<numStreams)&&(streamId!=1))cudaStreamWaitEvent(streams[streamId], startEvent, 0);

				document.CPU2GPU(chunkId, streamId, streams[streamId]);

				MaxTopicKernel(WT, document, streamId, streams[streamId]);

				UpdateDTKernel(chunkDT, document, streamId, streams[streamId]);
				//chunkDT.CPU2GPU(chunkId, document.docLengthVec[chunkId]);
				/*startTime1 = clock();*/
				
				/*endTime = clock();*/
				printf("%d\n", 4);
				//chunkDT.GPU2CPU(chunkId);
				//chunkDT.CPU2Disk(chunkFilePrefix, chunkId);// marker

				//--------------update DT matrix-----------
				// endTime = clock();
				printf("%d\n", 5);
				//DTTime += (double)(endTime - startTime1) / CLOCKS_PER_SEC;

				/*startTime1 = clock();*/
				

				UpdateProbKernelD(WT, chunkDT, document, randState[streamId], streamId, streams[streamId]);
				/*endTime = clock();
				UpdateMTime += (double)(endTime - startTime1) / CLOCKS_PER_SEC;*/
				printf("%d\n", 6);
				/*startTime1 = clock();*/

				//--------------sampling-----------

				/*if (chunkId == 0) {
				document.GPU2CPUEffectiveTokenIndex();
				document.CPU2DiskEffectiveTokenIndex(chunkFilePrefix);
				}
				*/

				//--------------sampling-----------



				//SampleKernelD(WTDen, WT, chunkDT, document, randState[streamId], streamId, streams[streamId]);

				/*endTime = clock();
				samplingTimeD += (double)(endTime - startTime1) / CLOCKS_PER_SEC;*/
				// if (chunkId == 0) {
				// 	document.GPU2CPUTime();
				// 	for (int i = 0; i < GridDim*BlockDim / 32; i++) {
				// 		warpTimeRecord << document.timeRecord[i] << " ";
				// 	}
				// 	warpTimeRecord << "\n";
				// }

				printf("%d\n", 7);
				//WTDen.WTDenGPU2CPU();// marker
				//WTDen.WTDenCPU2Disk(chunkFilePrefix);// marker
				/*startTime1 = clock();*/

				SampleKernelD(WT, chunkDT, document, randState[streamId], streamId, streams[streamId]);
				printf("%d\n", 8);
				/*endTime = clock();*/
				//WT.WTGPU2CPU();// marker
				//WT.CPU2Disk(chunkFilePrefix);// marker

				//--------------sampling-----------

				//	endTime = clock();
				/*samplingTimeS += (double)(endTime - startTime1) / CLOCKS_PER_SEC;

				startTime1 = clock();

				endTime = clock();
				transferTimeGPU2CPU += (double)(endTime - startTime1) / CLOCKS_PER_SEC;*/

				/*startTime1 = clock();*/
				//--------------update chunkWT matrix-----------
//				WT.chunkCPU2GPUCountOffset(chunkId, streamId, streams[streamId]);
//				WT.chunkGPUMemset(streamId, streams[streamId]);
//				UpdateWTDenKernel1(WTDen, WT, document, chunkId, streamId, streams[streamId]);
//				UpdateWTKernel(WT, document, chunkId, streamId, streams[streamId]);
//
//				WT.chunkWTGPU2CPU(chunkId, streamId, streams[streamId]);

				document.GPU2CPU(chunkId, streamId, streams[streamId]);
				if (streamId != 1) cudaEventRecord(stopEvents[streamId], streams[streamId]);
				//WT.CPU2DiskChunk(chunkFilePrefix, chunkId);
				//--------------update chunkWT matrix-----------
				//endTime = clock();
				//WTTime += (double)(endTime - startTime1) / CLOCKS_PER_SEC;
				/*return NULL;*/
			}



		}




		//for (int batchId = 0; batchId < chunksPerStream; batchId++) {

		//	for (int streamId = 0; streamId < numStreams; streamId++) {
		//		int chunkId = batchId*numStreams + streamId;
		//		threadBlock[streamId] = thread(thrd_func, ref(document), ref(chunkDT), ref(WT), ref(WTDen), randState, streams, chunkId, streamId);
		//		/*if (pthread_create(&thread[streamId], NULL, thrd_func, ())) {
		//		fprintf(stderr, "Error creating threadn");
		//		return 1;
		//		}*/
		//	}

		//	for (int streamId = 0; streamId < numStreams; streamId++) {
		//		threadBlock[streamId].join();
		//	}

		//}


		for (int streamId = 0; streamId < numStreams; streamId++) {
			if (streamId != 1) {
				//cudaEventRecord(stopEvents[streamId], streams[streamId]);
				cudaStreamWaitEvent(streams[1], stopEvents[streamId], 0);
			}
			
		}
		cudaEventRecord(iterStop,streams[1]);
		cudaEventSynchronize(iterStop);
		cudaEventElapsedTime(&iterTime,iterStart,iterStop);


		




		/*cudaDeviceSynchronize();*/
//		WT.GPUMemCopy(streams[1]);
//		WT.GPUMemset(streams[1]);
		for (int streamId = 0; streamId < numStreams; streamId++) {
			PerplexityKernel(document, streamId, streams[1]);
		}
		

	/*	document.PercentageCalculate();*/

		printf("done!!!!!");
		/*document.GPU2CPUPerplexity();*/

		// document.CPU2DiskPerplexity(chunkFilePrefix);

		//endTime = clock();
		//totalTime=(double)(endTime-startTime)/CLOCKS_PER_SEC;

		//maxPercentRecord << document.increasePercent << " " << document.topicUnchangedPercent << " " << document.perplexityAve[0] << "\n";
		//timeRecord << WTTime << " " << DTTime << " " <<UpdateMTime<< " " <<samplingTimeD << " " << samplingTimeS << " " << transferTimeCPU2GPU << " " << transferTimeGPU2CPU << " " <<totalTime << " " << document.sumPerplexity<< "\n";
		//
		SamplingDRecord << iterTime << "\n";

		//printf("WTTime: %f, DTTime: %f, samplingTimeD:%f, samplingTimeS:%f,transferTimeCPU2GPU:%f,transferTimeGPU2CPU:%f,totalTime:%f\n",WTTime,DTTime,samplingTimeD,samplingTimeS,transferTimeCPU2GPU,transferTimeGPU2CPU,totalTime);

	}

	timeRecord.close();
	warpTimeRecord.close();
	cudaDeviceReset();
}
#endif




//
//
//volatile __shared__ int p_input[ShaMemSize];
//volatile __shared__ int p_index[ShaMemSize];
//volatile __shared__ int p_value[ShaMemSize];
//volatile __shared__ int p_index_tmp[ShaMemSize];
//volatile __shared__ int p_value_tmp[ShaMemSize];
////volatile __shared__ int p_dense[K];
//int tid = threadIdx.x;
//int globalId = threadIdx.x + blockIdx.x * blockDim.x;
//int blockId = blockIdx.x;
//int indicator = 0;
//int GridDim = gridDim.x;
//
///*int wordIdWT = blockId + (*d_counter_0)*GridDim ;*/
///*long long tokenStart = d_TokenOffset[wordId];
//long long tokenEnd = d_TokenOffset[wordId] + d_TokenCount[wordId];*/
//
//
//
//if ((blockId > (*d_token_amount_0 - 1 - *d_counter_0*gridDim.x)) || (d_slotcount[blockId + (*d_counter_0)*GridDim] == 0))
//{
//	return;
//}
//int wordId = blockId + (*d_counter_0)*GridDim;
//p_input[tid] = 0;
//p_index[tid] = 0;
//p_value[tid] = 0;
//p_index_tmp[tid] = 0;
//p_value_tmp[tid] = 0;
//for (int k = tid; k < K; k += blockDim.x)
//{
//	d_dense[k + K*blockId] = 0;
//}
//
//__syncthreads();
//
//for (int i = tid; i < ((d_slotcount[wordId] - 1) / blockDim.x + 1)*blockDim.x; i += blockDim.x) {
//	if (i < d_slotcount[wordId]) {
//		int tmpIndex = d_slotoffset[wordId] + i + numOfTokenD;
//		p_input[tid] = d_a[tmpIndex];
//		//atomicAdd(&d_row_sum[p_input[tid] - 1], 1);
//	}
//
//	__syncthreads();
//	radix_sort(p_input);
//	__syncthreads();
//	index_value_count(p_input, p_index, p_value);
//	__syncthreads();
//	if (((d_slotcount[wordId] - indicator*blockDim.x) < blockDim.x) && (tid<(blockDim.x - 1)))
//	{
//		p_index_tmp[tid] = p_index[tid + 1];
//		p_value_tmp[tid] = p_value[tid + 1];
//	}
//	__syncthreads();
//
//	if (((d_slotcount[wordId] - indicator*blockDim.x) < blockDim.x) && (tid<(blockDim.x - 1)))
//	{
//		p_index[tid] = p_index_tmp[tid];
//		p_value[tid] = p_value_tmp[tid];
//	}
//	__syncthreads();
//
//	if (((d_slotcount[wordId] - indicator*blockDim.x) < blockDim.x) && (tid == (blockDim.x - 1)))
//	{
//		p_index[tid] = 0;
//		p_value[tid] = 0;
//	}
//	__syncthreads();
//	if (p_index[tid])
//	{
//		//atomicAdd(&p_dense[p_index[tid] - 1], 1);
//		d_dense[p_index[tid] - 1 + K*blockId] += p_value[tid];
//	}
//	__syncthreads();
//	p_index[tid] = 0;
//	p_value[tid] = 0;
//	p_input[tid] = 0;
//	p_index_tmp[tid] = 0;
//	p_index_tmp[tid] = 0;
//	indicator++;
//	__syncthreads();
//}
//__syncthreads();
///*if (globalId == 0) printf("%d mark\n", *d_counter_0);
//__syncthreads();*/
//dense_sparse_kernel(d_dense, d_index, d_value, d_count, d_slotcount, d_slotoffset, d_counter_0);
//__syncthreads();
//
//
//








