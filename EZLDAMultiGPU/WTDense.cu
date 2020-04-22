#include "WTDense.cuh"

WTD::WTD(int argNumOfWordD, int argWordLength, int argNumGPUs) {
	numOfWordD = argNumOfWordD;
	wordLength = argWordLength;
	numGPUs = argNumGPUs;
	WTDenseLength = argNumOfWordD*K;
	WTDense = new int[WTDenseLength];
	WTDenseCopy = new int[WTDenseLength];
	/*WTRowSumDense = new int[K];*/
}

void WTD::CPUMemSet() {

	memset(WTDense, 0, WTDenseLength * sizeof(int));
	memset(WTDenseCopy, 0, WTDenseLength * sizeof(int));
	//memset(WTRowSumDense, 0, K * sizeof(int));

}


//void WTD::GPUMemAllocate(int argGPUId) {
//	GPUId = argGPUId;
//	cudaSetDevice(GPUId);
//	cudaMalloc((void**)&deviceWTDense, (WTDenseLength) * sizeof(int));
//	cudaMalloc((void**)&deviceWTDenseCopy, (WTDenseLength) * sizeof(int));
//	//cudaMalloc((void**)&deviceWTRowSumDense, (K) * sizeof(int));
//
//	WTMemory = (2*WTDenseLength + K ) / 1000000000.0 * sizeof(int);
//	printf("WT memory usage(Sparse Part):%f GB\n", WTMemory);
//	
//}


void WTD::InitWTGPU()
{
	for (int GPUId = 0; GPUId < numGPUs; GPUId++) {

		WTDChunk WTDenseGPUChunk(numOfWordD, wordLength, GPUId);
		WTDenseGPUChunk.GPUMemAllocate(GPUId);
		WTDenseGPUChunk.GPUMemInit(GPUId);
		WTDenseGPUChunkVec.push_back(WTDenseGPUChunk);
	}

}

void WTD::GPUMemAllocate() {

	cudaSetDevice(0);
	cudaMalloc((void**)&deviceZeroWTDense, (WTDenseLength) * sizeof(int));

}

void WTD::GPUDataTransfer(int argGPUId, cudaStream_t &stream) {
	cudaSetDevice(0);
	cudaMemcpyAsync(deviceZeroWTDense, WTDenseGPUChunkVec[argGPUId].deviceWTDenseCopy, (WTDenseLength) * sizeof(int), cudaMemcpyDeviceToDevice, stream);

}

void WTD::GPUDataDistribute(int argGPUId, cudaStream_t &stream) {

	cudaSetDevice(0);
	cudaMemcpyAsync(WTDenseGPUChunkVec[argGPUId].deviceWTDense, WTDenseGPUChunkVec[0].deviceWTDense, (WTDenseLength) * sizeof(int), cudaMemcpyDeviceToDevice, stream);

}



void WTD::GPUDataTransferBackCPU(int argGPUId) {
	cudaSetDevice(argGPUId);
	cudaMemcpy(WTDenseCopy, WTDenseGPUChunkVec[argGPUId].deviceWTDenseCopy, (WTDenseLength) * sizeof(int), cudaMemcpyDeviceToHost);

}


void WTD::GPUDataTransferToGPU(int argGPUId) {
	cudaSetDevice(0);
	cudaMemcpy(deviceZeroWTDense, WTDenseCopy, (WTDenseLength) * sizeof(int), cudaMemcpyHostToDevice);

}



void WTD::GPUDataDistributeBackCPU(int argGPUId) {

	cudaSetDevice(0);
	cudaMemcpy(WTDense, WTDenseGPUChunkVec[0].deviceWTDense, (WTDenseLength) * sizeof(int), cudaMemcpyDeviceToHost);

}

void WTD::GPUDataDistributeToGPU(int argGPUId) {

	cudaSetDevice(argGPUId);
	cudaMemcpy(WTDenseGPUChunkVec[argGPUId].deviceWTDense, WTDense, (WTDenseLength) * sizeof(int), cudaMemcpyHostToDevice);

}


















//void WTD::GPUMemInit()
//{
//
//
//
//
//	cudaSetDevice(GPUId);
//	cudaMemset(deviceWTDense, 0, (WTDenseLength) * sizeof(int));
//	cudaMemset(deviceWTDenseCopy, 0, (WTDenseLength) * sizeof(int));
//	//cudaMemset(deviceWTRowSumDense, 0, (K) * sizeof(int));
//}

void WTD::GPUMemCopy(int argGPUId)
{
	cudaSetDevice(argGPUId);
	cudaMemcpy(WTDenseGPUChunkVec[argGPUId].deviceWTDense, WTDenseGPUChunkVec[argGPUId].deviceWTDenseCopy, (WTDenseLength) * sizeof(int), cudaMemcpyDeviceToDevice);
	
}

void WTD::GPUMemset(int argGPUId)
{
	cudaSetDevice(argGPUId);
	cudaMemset(WTDenseGPUChunkVec[argGPUId].deviceWTDenseCopy, 0, (WTDenseLength) * sizeof(int));
	//cudaMemset(deviceWTRowSumDense, 0, (K) * sizeof(int));
}


void WTD::WTDenGPU2CPU(int argGPUId)
{
	cudaSetDevice(argGPUId);
	cudaMemcpy(WTDense, WTDenseGPUChunkVec[argGPUId].deviceWTDense, (WTDenseLength) * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(WTDenseCopy, WTDenseGPUChunkVec[argGPUId].deviceWTDenseCopy, (WTDenseLength) * sizeof(int), cudaMemcpyDeviceToHost);
	

}
void WTD::WTDenCPU2Disk(string argFilePrefix) {

	ofstream WTDen((argFilePrefix + string("/WTDen.txt")).c_str(), ios::binary);
	for (int i = 0; i < WTDenseLength; i++) {
		WTDen << WTDense[i] << "\n";
	}
	WTDen.close();
	ofstream WTDenCopy((argFilePrefix + string("/WTDenCopy.txt")).c_str(), ios::binary);
	for (int i = 0; i < WTDenseLength; i++) {
		WTDenCopy << WTDenseCopy[i] << "\n";
	}
	WTDen.close();

}