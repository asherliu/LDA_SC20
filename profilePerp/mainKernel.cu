
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
	double transferTimeCPU2GPU=0.0;
	double transferTimeGPU2CPU=0.0;
	double WTTime=0.0;
	double samplingTimeD=0.0;
	double samplingTimeS=0.0;
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
	int numIters = 200;

	
	string chunkFilePrefix ="/gpfs/alpine/proj-shared/csc289/lda/datasets/nytimes";

	ifstream lengthVec((chunkFilePrefix + string("/lengthVec.txt")).c_str(), ios::binary);//store max Doc and DT length

	ofstream timeRecord((chunkFilePrefix + string("/timeRecord.txt")).c_str(), ios::binary);
	ofstream SamplingDRecord((chunkFilePrefix + string("/SamplingDRecord.txt")).c_str(), ios::binary);

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
	srand(time(NULL));

	//curandState* randState[2];
	//srand(time(NULL));
	//for (int i = 0; i < 2; i++) {
	//	cudaSetDevice(i);
	//	cudaMalloc(&randState[i], sizeof(curandState)*GridDim*BlockDim);//may have bugs
	//}
	//H_ERR(cudaDeviceSynchronize());


	curandState* randState;

	cudaMalloc(&randState, sizeof(curandState)*GridDim*BlockDim);
	H_ERR(cudaDeviceSynchronize());

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
		WT.chunkWTGPU2CPU(chunkId);// marker
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
		samplingTimeD=0;
		
		for (int chunkId = 0; chunkId < numChunks; chunkId++) {
			
			// startTime1=clock();
			printf("step: %d\n",chunkId);
			//--------------update DT matrix-----------
			

			startTime1=clock();
			document.CPU2GPU(chunkId);
			endTime = clock();
			transferTimeCPU2GPU+=(double)(endTime-startTime1)/CLOCKS_PER_SEC;

			printf("%d\n", 1);
			chunkDT.GPUMemSet(chunkId);
			printf("%d\n", 2);
			chunkDT.CPU2GPUDTCountOffset(chunkId);
			printf("%d\n", 3);
			//chunkDT.CPU2GPU(chunkId, document.docLengthVec[chunkId]);
			startTime1=clock();
			UpdateDTKernel(chunkDT, document);
			endTime = clock();
			printf("%d\n", 4);
			//chunkDT.GPU2CPU(chunkId);
			//chunkDT.CPU2Disk(chunkFilePrefix, chunkId);// marker
			
			//--------------update DT matrix-----------
			// endTime = clock();
			printf("%d\n", 5);
			DTTime+=(double)(endTime-startTime1)/CLOCKS_PER_SEC;

			

			startTime1=clock();
			//--------------sampling-----------
			printf("%d\n", 6);
			SampleKernelD(WTDen, WT, chunkDT, document, randState);
			endTime = clock();
			samplingTimeD+=(double)(endTime-startTime1)/CLOCKS_PER_SEC;
			

            printf("%d\n", 7);
			//WTDen.WTDenGPU2CPU();// marker
			//WTDen.WTDenCPU2Disk(chunkFilePrefix);// marker
            startTime1=clock();

			SampleKernel(WT, chunkDT, document, randState);
			printf("%d\n", 8);
			endTime = clock();
			//WT.WTGPU2CPU();// marker
			//WT.CPU2Disk(chunkFilePrefix);// marker
		
			//--------------sampling-----------

		//	endTime = clock();
			samplingTimeS+=(double)(endTime-startTime1)/CLOCKS_PER_SEC;

			startTime1=clock();
            document.GPU2CPU(chunkId);
			endTime = clock();
			transferTimeGPU2CPU+=(double)(endTime-startTime1)/CLOCKS_PER_SEC;

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
		PerplexityKernel(document);
		printf("done!!!!!");
		/*document.GPU2CPUPerplexity();*/

		// document.CPU2DiskPerplexity(chunkFilePrefix);

		endTime = clock();
		totalTime=(double)(endTime-startTime)/CLOCKS_PER_SEC;
		timeRecord << WTTime << " " << DTTime << " " << samplingTimeD << " " << samplingTimeS << " " << transferTimeCPU2GPU << " " <<transferTimeGPU2CPU << " " <<totalTime << " " << document.sumPerplexity<< "\n";

		SamplingDRecord << samplingTimeD << "\n";

		printf("WTTime: %f, DTTime: %f, samplingTimeD:%f, samplingTimeS:%f,transferTimeCPU2GPU:%f,transferTimeGPU2CPU:%f,totalTime:%f，sumPerplexity：%f\n",WTTime,DTTime,samplingTimeD,samplingTimeS,transferTimeCPU2GPU,transferTimeGPU2CPU,totalTime, document.sumPerplexity);

	}
	
	timeRecord.close();
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








