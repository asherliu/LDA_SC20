#include "WTDense.cuh"

WTD::WTD(int argNumOfWordD, int argWordLength) {
	numOfWordD = argNumOfWordD;
	wordLength = argWordLength;
	WTDenseLength = argNumOfWordD*K;

	/*WTDense = new int[WTDenseLength];
	WTDenseCopy = new int[WTDenseLength];*/

	//cudaMallocHost((void**)&WTDense, WTDenseLength * sizeof(int));
	//cudaMallocHost((void**)&WTDenseCopy, WTDenseLength * sizeof(int));

	//memset(WTDenseCopy, 0, WTDenseLength * sizeof(int));
	/*WTRowSumDense = new int[K];*/
}

void WTD::CPUMemSet() {

	memset(WTDense, 0, WTDenseLength * sizeof(int));
	memset(WTDenseCopy, 0, WTDenseLength * sizeof(int));
	//memset(WTRowSumDense, 0, K * sizeof(int));

}


void WTD::GPUMemAllocate() {

	cudaMalloc((void**)&deviceWTDense, (WTDenseLength) * sizeof(int));
	cudaMalloc((void**)&deviceWTDenseCopy, (WTDenseLength) * sizeof(int));
	//cudaMalloc((void**)&deviceWTRowSumDense, (K) * sizeof(int));

	WTMemory = (2*WTDenseLength + K ) / 1000000000.0 * sizeof(int);
	printf("WT memory usage(Sparse Part):%f GB\n", WTMemory);
	
}

void WTD::GPUMemInit()
{
	cudaMemset(deviceWTDense, 0, (WTDenseLength) * sizeof(int));
	cudaMemset(deviceWTDenseCopy, 0, (WTDenseLength) * sizeof(int));
	//cudaMemset(deviceWTRowSumDense, 0, (K) * sizeof(int));
}

void WTD::GPUMemCopy(cudaStream_t& stream)
{
	cudaMemcpyAsync(deviceWTDense, deviceWTDenseCopy, (WTDenseLength) * sizeof(int), cudaMemcpyDeviceToDevice, stream);
	
}

void WTD::GPUMemset(cudaStream_t& stream)
{
	cudaMemsetAsync(deviceWTDenseCopy, 0, (WTDenseLength) * sizeof(int), stream);
	//cudaMemcpyAsync(deviceWTDenseCopy, WTDenseCopy, (WTDenseLength) * sizeof(int), cudaMemcpyHostToDevice,stream);
	//cudaMemset(deviceWTRowSumDense, 0, (K) * sizeof(int));
}


void WTD::WTDenGPU2CPU()
{

	cudaMemcpy(WTDense, deviceWTDense, (WTDenseLength) * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(WTDenseCopy, deviceWTDenseCopy, (WTDenseLength) * sizeof(int), cudaMemcpyDeviceToHost);
	

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
