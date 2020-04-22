#include "utility.cuh"

#define	BUFF_SIZE_LONG	100000

__device__ struct maxStruct {
	float maxProb=0.0;
	unsigned short int maxK=0;

};
__global__ void WT_Update_Kernel(unsigned short int *d_a, int *d_count, unsigned short int *d_index, unsigned short int *d_value, int *d_slotcount, int *d_slotoffset, int *d_row_sum, unsigned int *d_counter_0, int d_token_amount_0, int *d_dense, int numOfTokenD) {

	int globalId = threadIdx.x + blockIdx.x * blockDim.x;
	int laneId = threadIdx.x % 32;
	int warpId = globalId / 32;
	int iterCounter = 0;
	unsigned int Counter;


	if (laneId == 0) {

		Counter = atomicAdd(&d_counter_0[0], 1);
	}
	Counter = __shfl(Counter, 0);

	while (Counter < d_token_amount_0)
		//while (warpId + iterCounter * gridDim.x*blockDim.x / 32< argD)
	{
		int wordId = Counter;
	
		for (int k = laneId; k < K; k += 32)
		{
			d_dense[k + K*warpId] = 0;
		}

		for (int i = d_slotoffset[wordId] + laneId; i < d_slotoffset[wordId] + d_slotcount[wordId]; i += 32)
		{

			unsigned short int topic = d_a[i+numOfTokenD];
			if ((topic < 1) || (topic > K)) printf("wrong Index:%d", topic);
			atomicAdd(&d_dense[K*warpId + topic - 1], 1);
		}

		int noneZeroCount = 0;
		for (int k = laneId; k < K; k += 32) {
			int value = d_dense[K*warpId + k];
			int flag = value > 0;
			int tmpNoneZeroCount = __popc(__ballot(value));

			if (tmpNoneZeroCount == 0) continue;

			flag += __shfl_up(flag, 1, 32)*(laneId >= 1);
			flag += __shfl_up(flag, 2, 32)*(laneId >= 2);
			flag += __shfl_up(flag, 4, 32)*(laneId >= 4);
			flag += __shfl_up(flag, 8, 32)*(laneId >= 8);
			flag += __shfl_up(flag, 16, 32)*(laneId >= 16);

			if (value) {
				int idx = d_slotoffset[wordId] + noneZeroCount + flag - 1;
				d_index[idx] = k + 1;
				d_value[idx] = value;
			}
			noneZeroCount += tmpNoneZeroCount;

		}
		/*if(laneId==0) d_count[docId] = noneZeroCount;*/
		if (laneId == 0) {
			d_count[wordId] = noneZeroCount;
			Counter = atomicAdd(&d_counter_0[0], 1);
		}
		Counter = __shfl(Counter, 0);

		/*iterCounter ++;*/

	}
	
}
__device__ short atomicAddShort(short* address, short val)

{

    unsigned int *base_address = (unsigned int *)((size_t)address & ~2);

    unsigned int old, assumed, new_;



    old = *base_address;

    do {

        assumed = old;

        new_ = ((size_t)address & 2) ? old + ((unsigned int)val << 16) : (old & 0xffff0000) | ((old & 0xffff) + val);

       old = atomicCAS(base_address, assumed, new_);

    } while (assumed != old);

    return old;

}

__global__ void DT_Update_Kernel(int *d_Index, unsigned short int *d_a, int *d_count, unsigned short int *d_index, int *d_value, int *d_slotcount, int *d_slotoffset, int *d_sparse_slotcount, int *d_sparse_slotoffset, unsigned int *d_counter_0, int argD, int *d_dense, unsigned short int* deviceMaxTokenCount, unsigned short int* deviceMaxTopic, unsigned short int* deviceSecondMaxTopic, unsigned short int* deviceSecondMaxTokenCount)
{

	int globalId = threadIdx.x + blockIdx.x * blockDim.x;
	int laneId = threadIdx.x % 32;
	int warpId = threadIdx.x / 32;
	int iterCounter = 0;
	unsigned int Counter;
	//__shared__ unsigned int DT[K*BlockDim/32];
	__shared__ int DT[K*BlockDim/32];
	__shared__ short int Index[K*BlockDim/32];
	//unsigned short int one=1;

	if (laneId == 0) {

		Counter = atomicAdd(&d_counter_0[0], 1);
	}
	Counter = __shfl(Counter, 0);

	while (Counter < argD)
	//while (warpId + iterCounter * gridDim.x*blockDim.x / 32< argD)
	{
		/*warpId = Counter;*/

		int docId = Counter;

		for (int k = laneId; k < K; k += 32)
		{
			DT[k + K*warpId] = 0;
		}
		// finish0 = clock64();
		// costtime0 += (double)(finish0 - start0);
		

		int numIter=d_slotcount[docId]/64*64;
		//int countLeft=d_slotcount[docId]-numIter;
		
		for (int i = d_slotoffset[docId] + laneId; (i+32)< d_slotoffset[docId] + numIter; i += 64)
		{	
			//start0 = clock64();
			int idx  = __ldg(&d_Index[i]);
			int idx1 = __ldg(&d_Index[i+32]);
			//finish0 = clock64();
			//costtime0 += (double)(finish0 - start0);
			//finish0 = clock64();
			unsigned short int topic = __ldg(&d_a[idx])-1;
			unsigned short int topic1 = __ldg(&d_a[idx1])-1;
			//finish1 = clock64();
			//costtime1 += (double)(finish1 - finish0);
			//if ((topic < 1) || (topic > K)) printf("wrong Index:%d", topic);
			//atomicAddShort(&DT[topic+ K*warpId], 1);
			atomicAdd(&DT[topic+ K*warpId], 1);
			atomicAdd(&DT[topic1+ K*warpId], 1);
			//atomicCAS(&DT[topic+ K*warpId], DT[topic+ K*warpId],  one);
		}
		for (int i = d_slotoffset[docId] + numIter + laneId; i< d_slotoffset[docId] + d_slotcount[docId]; i += 32)
		{	
			//start0 = clock64();
			int idx  = __ldg(&d_Index[i]);
			// int idx1 = __ldg(&d_Index[i+32]);
			//finish0 = clock64();
			//costtime0 += (double)(finish0 - start0);
			//finish0 = clock64();
			unsigned short int topic = __ldg(&d_a[idx])-1;
			// unsigned short int topic1 = __ldg(&d_a[idx1])-1;
			//finish1 = clock64();
			//costtime1 += (double)(finish1 - finish0);
			//if ((topic < 1) || (topic > K)) printf("wrong Index:%d", topic);
			//atomicAddShort(&DT[topic+ K*warpId], 1);
			atomicAdd(&DT[topic+ K*warpId], 1);
			// atomicAdd(&DT[topic1+ K*warpId], 1);
			//atomicCAS(&DT[topic+ K*warpId], DT[topic+ K*warpId],  one);
		}




		//finish1 = clock64();
		// finish1 = clock64();
		// costtime1 += (double)(finish1 - finish0);

		for (int i = d_slotoffset[docId] + laneId; i < d_slotoffset[docId] + d_slotcount[docId]; i += 32)
		{	
			// start0 = clock64();
			unsigned short int topic = __ldg(&deviceMaxTopic[d_Index[i]])-1;
			unsigned short int secondTopic = __ldg(&deviceSecondMaxTopic[d_Index[i]])-1;
			// finish0 = clock64();
			// costtime0 += (double)(finish0 - start0);
			// finish0 = clock64();
			deviceMaxTokenCount[d_Index[i]]= DT[topic + K*warpId];
			deviceSecondMaxTokenCount[d_Index[i]] = DT[secondTopic+ K*warpId];
			// finish1 = clock64();
			// costtime1 += (double)(finish1 - finish0);
		}
		// finish2 = clock64();
		// costtime2 += (double)(finish2 - finish1);

		int noneZeroCount = 0;
		for (int k = laneId; k < K; k += 32) {
			
			int value = DT[k + K*warpId];
			int flag = value > 0;
			
			int tmpNoneZeroCount = __popc(__ballot(value));

			

			if (tmpNoneZeroCount == 0) continue;
			
	
			
			flag += __shfl_up(flag, 1, 32)*(laneId >= 1);
			flag += __shfl_up(flag, 2, 32)*(laneId >= 2);
			flag += __shfl_up(flag, 4, 32)*(laneId >= 4);
			flag += __shfl_up(flag, 8, 32)*(laneId >= 8);
			flag += __shfl_up(flag, 16, 32)*(laneId >= 16);

			//finish2 = clock64();
			// if (value > 0) {

			// 	int idx = d_sparse_slotoffset[docId] + noneZeroCount+ flag-1;
				
			// 	d_index[idx] = k+1;
			// 	d_value[idx] = value;
				

			// }

			if (value > 0) {

				int idx = noneZeroCount+ flag-1+K*warpId;
				
				Index[idx] = k+1;
				DT[idx] = value;
				

			}

			// int idx = d_sparse_slotoffset[docId] + noneZeroCount+ flag-1;
				
			// d_index[idx] = k+1;
			// d_value[idx] = value;

			
			
			noneZeroCount += tmpNoneZeroCount;
			
		}
		/*if(laneId==0) d_count[docId] = noneZeroCount;*/
		if (laneId == 0) {
			d_count[docId] = noneZeroCount;
			Counter = atomicAdd(&d_counter_0[0], 1);
		}
		Counter = __shfl(Counter, 0);


		for (int i = d_sparse_slotoffset[docId] + laneId; i < d_sparse_slotoffset[docId] + noneZeroCount; i += 32){
			int idx=K*warpId + i - d_sparse_slotoffset[docId];
			d_index[i]=Index[idx];
			d_value[i]=DT[idx];
		}

//		finish3 = clock64();
//		costtime3 += (double)(finish3 - finish2);
		
		/*iterCounter ++;*/
		/*iterCounter ++;*/

	}
	//if (threadIdx.x + blockDim.x*blockIdx.x == 0) printf("costtime0,costtime1,costtime2,costtime3:%f,%f,%f,%f\n", costtime0 / (158200000 * 1.0), costtime1 / (158200000 * 1.0),costtime2 / (158200000 * 1.0), costtime3 / (158200000 * 1.0));

}




















__global__ void MaxTopicDense_Update_Kernel(unsigned short int* deviceWordMaxTopic, unsigned short int* deviceWordSecondMaxTopic, unsigned short int* deviceMaxTopic, int *deviceWTDense, int *deviceTLCount, int *deviceTLOffset, int *deviceWTOffset, int numOfWordD, unsigned int* deviceCounter, int *deviceWTRowSum,int wordLength, float beta, unsigned short int* deviceWordThirdMaxTopic, unsigned short int* deviceSecondMaxTopic) {

	int laneId = threadIdx.x % 32;
	int localId = threadIdx.x / 32;
	volatile __shared__ float WTHead[K];
	volatile __shared__ float MaxTree[32];
	volatile __shared__ float MaxWT[3];
	volatile __shared__ unsigned short int MaxKTree[32];
	volatile __shared__ unsigned short int MaxK[3];
	volatile __shared__ unsigned int Counter[1];
	if (threadIdx.x == 0) {
		Counter[0] = atomicAdd(&deviceCounter[0], 1);
	}
	__syncthreads();

	while (Counter[0] < numOfWordD)
	{
		int wordId = Counter[0];
		if (localId == 0) {
			MaxTree[laneId] = 0;
			MaxKTree[laneId] = 0;
		}

		int tokenStart = deviceTLOffset[wordId];
		int tokenEnd = deviceTLOffset[wordId] + deviceTLCount[wordId];
		int WTStart = deviceWTOffset[wordId];
		// Reconstruct dense WT vector from sparse WT matrix
		for (int i = threadIdx.x; i < K; i += blockDim.x)
		{
			WTHead[i] = (deviceWTDense[WTStart + i] + beta) / (deviceWTRowSum[i] + wordLength*beta);
		}
		__syncthreads();
		// Find maxK
		for (int i = localId; i < K / 32; i += blockDim.x / 32) {
			unsigned short int   tmpK = i * 32 + laneId;
			float tmpMax = 0.0;
			float tmpMax1 = 0.0;
			unsigned short int tmpK1 = 0;
			tmpMax = WTHead[tmpK];
			tmpMax1 = __shfl_down(tmpMax, 16);
			tmpK1 = __shfl_down(tmpK, 16);
			if (tmpMax < tmpMax1) {
				tmpMax = tmpMax1;
				tmpK = tmpK1;
			}
			tmpMax1 = __shfl_down(tmpMax, 8);
			tmpK1 = __shfl_down(tmpK, 8);
			if (tmpMax < tmpMax1) {
				tmpMax = tmpMax1;
				tmpK = tmpK1;
			}
			tmpMax1 = __shfl_down(tmpMax, 4);
			tmpK1 = __shfl_down(tmpK, 4);
			if (tmpMax < tmpMax1) {
				tmpMax = tmpMax1;
				tmpK = tmpK1;
			}
			tmpMax1 = __shfl_down(tmpMax, 2);
			tmpK1 = __shfl_down(tmpK, 2);
			if (tmpMax < tmpMax1) {
				tmpMax = tmpMax1;
				tmpK = tmpK1;
			}
			tmpMax1 = __shfl_down(tmpMax, 1);
			tmpK1 = __shfl_down(tmpK, 1);
			if (tmpMax < tmpMax1) {
				tmpMax = tmpMax1;
				tmpK = tmpK1;
			}
			tmpMax = __shfl(tmpMax, 0);
			tmpK = __shfl(tmpK, 0);
			MaxTree[i] = tmpMax;
			MaxKTree[i] = tmpK;
		}
		__syncthreads();

		if (localId == 0) {

			float tmpMax = 0.0;
			float tmpMax1 = 0.0;
			unsigned short int tmpMaxK = 0;
			unsigned short int tmpMaxK1 = 0;
			tmpMax = MaxTree[laneId];
			tmpMaxK = MaxKTree[laneId];
			tmpMax1 = __shfl_down(tmpMax, 16);
			tmpMaxK1 = __shfl_down(tmpMaxK, 16);
			if (tmpMax < tmpMax1) {
				tmpMax = tmpMax1;
				tmpMaxK = tmpMaxK1;
			}
			tmpMax1 = __shfl_down(tmpMax, 8);
			tmpMaxK1 = __shfl_down(tmpMaxK, 8);
			if (tmpMax < tmpMax1) {
				tmpMax = tmpMax1;
				tmpMaxK = tmpMaxK1;
			}
			tmpMax1 = __shfl_down(tmpMax, 4);
			tmpMaxK1 = __shfl_down(tmpMaxK, 4);
			if (tmpMax < tmpMax1) {
				tmpMax = tmpMax1;
				tmpMaxK = tmpMaxK1;
			}
			tmpMax1 = __shfl_down(tmpMax, 2);
			tmpMaxK1 = __shfl_down(tmpMaxK, 2);
			if (tmpMax < tmpMax1) {
				tmpMax = tmpMax1;
				tmpMaxK = tmpMaxK1;
			}
			tmpMax1 = __shfl_down(tmpMax, 1);
			tmpMaxK1 = __shfl_down(tmpMaxK, 1);
			if (tmpMax < tmpMax1) {
				tmpMax = tmpMax1;
				tmpMaxK = tmpMaxK1;
			}
			tmpMaxK = __shfl(tmpMaxK, 0);
			if (laneId == 0) {
				MaxWT[laneId] = tmpMax;
				MaxK[laneId] = tmpMaxK;
			}

		}
		__syncthreads();
		if (localId == 0) {

			float tmpMax = 0.0;
			float tmpMax1 = 0.0;
			unsigned short int tmpPosition = MaxK[0] / 32;
			unsigned short int tmpK = (tmpPosition) * 32 + laneId;
			unsigned short int tmpK1 = 0;
			

			tmpMax = WTHead[tmpK];
			if (tmpK == MaxK[0]) tmpMax = 0.0;
			tmpMax1 = __shfl_down(tmpMax, 16);
			tmpK1 = __shfl_down(tmpK, 16);
			if (tmpMax < tmpMax1) {
				tmpMax = tmpMax1;
				tmpK = tmpK1;
			}
			tmpMax1 = __shfl_down(tmpMax, 8);
			tmpK1 = __shfl_down(tmpK, 8);
			if (tmpMax < tmpMax1) {
				tmpMax = tmpMax1;
				tmpK = tmpK1;
			}
			tmpMax1 = __shfl_down(tmpMax, 4);
			tmpK1 = __shfl_down(tmpK, 4);
			if (tmpMax < tmpMax1) {
				tmpMax = tmpMax1;
				tmpK = tmpK1;
			}
			tmpMax1 = __shfl_down(tmpMax, 2);
			tmpK1 = __shfl_down(tmpK, 2);
			if (tmpMax < tmpMax1) {
				tmpMax = tmpMax1;
				tmpK = tmpK1;
			}
			tmpMax1 = __shfl_down(tmpMax, 1);
			tmpK1 = __shfl_down(tmpK, 1);
			if (tmpMax < tmpMax1) {
				tmpMax = tmpMax1;
				tmpK = tmpK1;
			}
			tmpMax = __shfl(tmpMax, 0);
			tmpK = __shfl(tmpK, 0);
			MaxTree[tmpPosition] = tmpMax;
			MaxKTree[tmpPosition] = tmpK;

			tmpMax = 0.0;
			tmpMax1 = 0.0;
			tmpMax = MaxTree[laneId];
			tmpMax1 = __shfl_down(tmpMax, 16);
			tmpK = MaxKTree[laneId];
			tmpK1 = __shfl_down(tmpK, 16);

			if (tmpMax < tmpMax1) {
				tmpMax = tmpMax1;
				tmpK = tmpK1;
			}
			tmpMax1 = __shfl_down(tmpMax, 8);
			tmpK1 = __shfl_down(tmpK, 8);
			if (tmpMax < tmpMax1) {
				tmpMax = tmpMax1;
				tmpK = tmpK1;
			}
			tmpMax1 = __shfl_down(tmpMax, 4);
			tmpK1 = __shfl_down(tmpK, 4);
			if (tmpMax < tmpMax1) {
				tmpMax = tmpMax1;
				tmpK = tmpK1;
			}
			tmpMax1 = __shfl_down(tmpMax, 2);
			tmpK1 = __shfl_down(tmpK, 2);
			if (tmpMax < tmpMax1) {
				tmpMax = tmpMax1;
				tmpK = tmpK1;
			}
			tmpMax1 = __shfl_down(tmpMax, 1);
			tmpK1 = __shfl_down(tmpK, 1);
			if (tmpMax < tmpMax1) {
				tmpMax = tmpMax1;
				tmpK = tmpK1;
			}
			if (laneId == 0) {
				MaxWT[1] = tmpMax;
				MaxK[1] = tmpK;
			}
		}
		__syncthreads();

		if (localId == 0) {

			float tmpMax = 0.0;
			float tmpMax1 = 0.0;
			unsigned short int tmpPosition = MaxK[1] / 32;
			unsigned short int tmpK = (tmpPosition) * 32 + laneId;
			unsigned short int tmpK1 = 0;


			tmpMax = WTHead[tmpK];
			if (tmpK == MaxK[0]) tmpMax = 0.0;
			if (tmpK == MaxK[1]) tmpMax = 0.0;
			tmpMax1 = __shfl_down(tmpMax, 16);
			tmpK1 = __shfl_down(tmpK, 16);
			if (tmpMax < tmpMax1) {
				tmpMax = tmpMax1;
				tmpK = tmpK1;
			}
			tmpMax1 = __shfl_down(tmpMax, 8);
			tmpK1 = __shfl_down(tmpK, 8);
			if (tmpMax < tmpMax1) {
				tmpMax = tmpMax1;
				tmpK = tmpK1;
			}
			tmpMax1 = __shfl_down(tmpMax, 4);
			tmpK1 = __shfl_down(tmpK, 4);
			if (tmpMax < tmpMax1) {
				tmpMax = tmpMax1;
				tmpK = tmpK1;
			}
			tmpMax1 = __shfl_down(tmpMax, 2);
			tmpK1 = __shfl_down(tmpK, 2);
			if (tmpMax < tmpMax1) {
				tmpMax = tmpMax1;
				tmpK = tmpK1;
			}
			tmpMax1 = __shfl_down(tmpMax, 1);
			tmpK1 = __shfl_down(tmpK, 1);
			if (tmpMax < tmpMax1) {
				tmpMax = tmpMax1;
				tmpK = tmpK1;
			}
			tmpMax = __shfl(tmpMax, 0);
			tmpK = __shfl(tmpK, 0);
			MaxTree[tmpPosition] = tmpMax;
			MaxKTree[tmpPosition] = tmpK;

			tmpMax = 0.0;
			tmpMax1 = 0.0;
			tmpMax = MaxTree[laneId];
			tmpMax1 = __shfl_down(tmpMax, 16);
			tmpK = MaxKTree[laneId];
			tmpK1 = __shfl_down(tmpK, 16);

			if (tmpMax < tmpMax1) {
				tmpMax = tmpMax1;
				tmpK = tmpK1;
			}
			tmpMax1 = __shfl_down(tmpMax, 8);
			tmpK1 = __shfl_down(tmpK, 8);
			if (tmpMax < tmpMax1) {
				tmpMax = tmpMax1;
				tmpK = tmpK1;
			}
			tmpMax1 = __shfl_down(tmpMax, 4);
			tmpK1 = __shfl_down(tmpK, 4);
			if (tmpMax < tmpMax1) {
				tmpMax = tmpMax1;
				tmpK = tmpK1;
			}
			tmpMax1 = __shfl_down(tmpMax, 2);
			tmpK1 = __shfl_down(tmpK, 2);
			if (tmpMax < tmpMax1) {
				tmpMax = tmpMax1;
				tmpK = tmpK1;
			}
			tmpMax1 = __shfl_down(tmpMax, 1);
			tmpK1 = __shfl_down(tmpK, 1);
			if (tmpMax < tmpMax1) {
				tmpMax = tmpMax1;
				tmpK = tmpK1;
			}
			if (laneId == 0) {
				MaxWT[2] = tmpMax;
				MaxK[2] = tmpK;
			}
		}

		__syncthreads();


		//float WTMax = MaxWT[0];
		//float WTMax2 = MaxWT[1];
		//float WTMax3 = MaxWT[2];
		unsigned short int WTMaxK = MaxK[0];
		unsigned short int WTSecondMaxK = MaxK[1];
		unsigned short int WTThirdMaxK = MaxK[2];
		if (threadIdx.x == 0) {
			deviceWordMaxTopic[wordId] = WTMaxK+1;
			deviceWordSecondMaxTopic[wordId] = WTSecondMaxK+1;
			deviceWordThirdMaxTopic[wordId] = WTThirdMaxK + 1;
		}
		for (int i = tokenStart+ threadIdx.x; i < tokenEnd; i += blockDim.x) {
			deviceMaxTopic[i] = WTMaxK+1;
			deviceSecondMaxTopic[i] = WTSecondMaxK + 1;

		}

		if (threadIdx.x == 0) Counter[0] = atomicAdd(&deviceCounter[0], 1);
		__syncthreads();
	}

}
__global__ void MaxTopicSparse_Update_Kernel(unsigned short int* deviceWordMaxTopic, unsigned short int* deviceWordSecondMaxTopic, unsigned short int* deviceMaxTopic, int *deviceTLCount, int *deviceTLOffset, int *deviceWTOffset, int numOfWordD, unsigned int* deviceCounter, int *deviceWTRowSum, int wordLength, int numOfWordS, int* d_WordListOffset, int* d_SparseWTCount, unsigned short int* d_SparseWTIndex, unsigned short int* d_SparseWTValue, float beta, unsigned short int* deviceWordThirdMaxTopic, unsigned short int* deviceSecondMaxTopic) {


	int laneId = threadIdx.x % 32;
	int localId = threadIdx.x / 32;
	volatile __shared__ float WTHead[K];
	volatile __shared__ float MaxTree[32];
	volatile __shared__ float MaxWT[3];
	volatile __shared__ unsigned short int MaxKTree[32];
	volatile __shared__ unsigned short int MaxK[3];
	volatile __shared__ unsigned int Counter[1];
	if (threadIdx.x == 0) {
		Counter[0] = atomicAdd(&deviceCounter[0], 1);
	}
	__syncthreads();

	while (Counter[0] < numOfWordS)
	{
		int wordId = Counter[0] + numOfWordD;
		if (localId == 0) {
			MaxTree[laneId] = 0;
			MaxKTree[laneId] = 0;
		}

		long long tokenStart = deviceTLOffset[wordId];
		long long tokenEnd = deviceTLOffset[wordId] + deviceTLCount[wordId];
		long long WTStart = d_WordListOffset[wordId] - d_WordListOffset[numOfWordD];
		long long WTEnd = d_WordListOffset[wordId] - d_WordListOffset[numOfWordD] + d_SparseWTCount[wordId - numOfWordD];
		// Reconstruct dense WT vector from sparse WT matrix
		for (int i = threadIdx.x; i < K; i += blockDim.x)
		{
			WTHead[i] = beta / (deviceWTRowSum[i] + wordLength*beta);

		}
		__syncthreads();

		for (int i = threadIdx.x + WTStart; i < WTEnd; i += blockDim.x)
		{
			WTHead[d_SparseWTIndex[i] - 1] = (d_SparseWTValue[i] + beta) / (deviceWTRowSum[d_SparseWTIndex[i] - 1] + wordLength*beta);

		}
		__syncthreads();

		// Find maxK
		for (int i = localId; i < K / 32; i += blockDim.x / 32) {
			unsigned short int   tmpK = i * 32 + laneId;
			float tmpMax = 0.0;
			float tmpMax1 = 0.0;
			unsigned short int tmpK1 = 0;
			tmpMax = WTHead[tmpK];
			tmpMax1 = __shfl_down(tmpMax, 16);
			tmpK1 = __shfl_down(tmpK, 16);
			if (tmpMax < tmpMax1) {
				tmpMax = tmpMax1;
				tmpK = tmpK1;
			}
			tmpMax1 = __shfl_down(tmpMax, 8);
			tmpK1 = __shfl_down(tmpK, 8);
			if (tmpMax < tmpMax1) {
				tmpMax = tmpMax1;
				tmpK = tmpK1;
			}
			tmpMax1 = __shfl_down(tmpMax, 4);
			tmpK1 = __shfl_down(tmpK, 4);
			if (tmpMax < tmpMax1) {
				tmpMax = tmpMax1;
				tmpK = tmpK1;
			}
			tmpMax1 = __shfl_down(tmpMax, 2);
			tmpK1 = __shfl_down(tmpK, 2);
			if (tmpMax < tmpMax1) {
				tmpMax = tmpMax1;
				tmpK = tmpK1;
			}
			tmpMax1 = __shfl_down(tmpMax, 1);
			tmpK1 = __shfl_down(tmpK, 1);
			if (tmpMax < tmpMax1) {
				tmpMax = tmpMax1;
				tmpK = tmpK1;
			}
			tmpMax = __shfl(tmpMax, 0);
			tmpK = __shfl(tmpK, 0);
			MaxTree[i] = tmpMax;
			MaxKTree[i] = tmpK;
		}
		__syncthreads();

		if (localId == 0) {

			float tmpMax = 0.0;
			float tmpMax1 = 0.0;
			unsigned short int tmpMaxK = 0;
			unsigned short int tmpMaxK1 = 0;
			tmpMax = MaxTree[laneId];
			tmpMaxK = MaxKTree[laneId];
			tmpMax1 = __shfl_down(tmpMax, 16);
			tmpMaxK1 = __shfl_down(tmpMaxK, 16);
			if (tmpMax < tmpMax1) {
				tmpMax = tmpMax1;
				tmpMaxK = tmpMaxK1;
			}
			tmpMax1 = __shfl_down(tmpMax, 8);
			tmpMaxK1 = __shfl_down(tmpMaxK, 8);
			if (tmpMax < tmpMax1) {
				tmpMax = tmpMax1;
				tmpMaxK = tmpMaxK1;
			}
			tmpMax1 = __shfl_down(tmpMax, 4);
			tmpMaxK1 = __shfl_down(tmpMaxK, 4);
			if (tmpMax < tmpMax1) {
				tmpMax = tmpMax1;
				tmpMaxK = tmpMaxK1;
			}
			tmpMax1 = __shfl_down(tmpMax, 2);
			tmpMaxK1 = __shfl_down(tmpMaxK, 2);
			if (tmpMax < tmpMax1) {
				tmpMax = tmpMax1;
				tmpMaxK = tmpMaxK1;
			}
			tmpMax1 = __shfl_down(tmpMax, 1);
			tmpMaxK1 = __shfl_down(tmpMaxK, 1);
			if (tmpMax < tmpMax1) {
				tmpMax = tmpMax1;
				tmpMaxK = tmpMaxK1;
			}
			tmpMaxK = __shfl(tmpMaxK, 0);
			if (laneId == 0) {
				MaxWT[laneId] = tmpMax;
				MaxK[laneId] = tmpMaxK;
			}

		}
		__syncthreads();
		if (localId == 0) {

			float tmpMax = 0.0;
			float tmpMax1 = 0.0;
			unsigned short int tmpPosition = MaxK[0] / 32;
			unsigned short int tmpK = (tmpPosition) * 32 + laneId;
			unsigned short int tmpK1 = 0;


			tmpMax = WTHead[tmpK];
			if (tmpK == MaxK[0]) tmpMax = 0.0;
			tmpMax1 = __shfl_down(tmpMax, 16);
			tmpK1 = __shfl_down(tmpK, 16);
			if (tmpMax < tmpMax1) {
				tmpMax = tmpMax1;
				tmpK = tmpK1;
			}
			tmpMax1 = __shfl_down(tmpMax, 8);
			tmpK1 = __shfl_down(tmpK, 8);
			if (tmpMax < tmpMax1) {
				tmpMax = tmpMax1;
				tmpK = tmpK1;
			}
			tmpMax1 = __shfl_down(tmpMax, 4);
			tmpK1 = __shfl_down(tmpK, 4);
			if (tmpMax < tmpMax1) {
				tmpMax = tmpMax1;
				tmpK = tmpK1;
			}
			tmpMax1 = __shfl_down(tmpMax, 2);
			tmpK1 = __shfl_down(tmpK, 2);
			if (tmpMax < tmpMax1) {
				tmpMax = tmpMax1;
				tmpK = tmpK1;
			}
			tmpMax1 = __shfl_down(tmpMax, 1);
			tmpK1 = __shfl_down(tmpK, 1);
			if (tmpMax < tmpMax1) {
				tmpMax = tmpMax1;
				tmpK = tmpK1;
			}
			tmpMax = __shfl(tmpMax, 0);
			tmpK = __shfl(tmpK, 0);
			MaxTree[tmpPosition] = tmpMax;
			MaxKTree[tmpPosition] = tmpK;

			tmpMax = 0.0;
			tmpMax1 = 0.0;
			tmpMax = MaxTree[laneId];
			tmpMax1 = __shfl_down(tmpMax, 16);
			tmpK = MaxKTree[laneId];
			tmpK1 = __shfl_down(tmpK, 16);

			if (tmpMax < tmpMax1) {
				tmpMax = tmpMax1;
				tmpK = tmpK1;
			}
			tmpMax1 = __shfl_down(tmpMax, 8);
			tmpK1 = __shfl_down(tmpK, 8);
			if (tmpMax < tmpMax1) {
				tmpMax = tmpMax1;
				tmpK = tmpK1;
			}
			tmpMax1 = __shfl_down(tmpMax, 4);
			tmpK1 = __shfl_down(tmpK, 4);
			if (tmpMax < tmpMax1) {
				tmpMax = tmpMax1;
				tmpK = tmpK1;
			}
			tmpMax1 = __shfl_down(tmpMax, 2);
			tmpK1 = __shfl_down(tmpK, 2);
			if (tmpMax < tmpMax1) {
				tmpMax = tmpMax1;
				tmpK = tmpK1;
			}
			tmpMax1 = __shfl_down(tmpMax, 1);
			tmpK1 = __shfl_down(tmpK, 1);
			if (tmpMax < tmpMax1) {
				tmpMax = tmpMax1;
				tmpK = tmpK1;
			}
			if (laneId == 0) {
				MaxWT[1] = tmpMax;
				MaxK[1] = tmpK;
			}
		}
		__syncthreads();


		if (localId == 0) {

			float tmpMax = 0.0;
			float tmpMax1 = 0.0;
			unsigned short int tmpPosition = MaxK[1] / 32;
			unsigned short int tmpK = (tmpPosition) * 32 + laneId;
			unsigned short int tmpK1 = 0;


			tmpMax = WTHead[tmpK];
			if (tmpK == MaxK[0]) tmpMax = 0.0;
			if (tmpK == MaxK[1]) tmpMax = 0.0;
			tmpMax1 = __shfl_down(tmpMax, 16);
			tmpK1 = __shfl_down(tmpK, 16);
			if (tmpMax < tmpMax1) {
				tmpMax = tmpMax1;
				tmpK = tmpK1;
			}
			tmpMax1 = __shfl_down(tmpMax, 8);
			tmpK1 = __shfl_down(tmpK, 8);
			if (tmpMax < tmpMax1) {
				tmpMax = tmpMax1;
				tmpK = tmpK1;
			}
			tmpMax1 = __shfl_down(tmpMax, 4);
			tmpK1 = __shfl_down(tmpK, 4);
			if (tmpMax < tmpMax1) {
				tmpMax = tmpMax1;
				tmpK = tmpK1;
			}
			tmpMax1 = __shfl_down(tmpMax, 2);
			tmpK1 = __shfl_down(tmpK, 2);
			if (tmpMax < tmpMax1) {
				tmpMax = tmpMax1;
				tmpK = tmpK1;
			}
			tmpMax1 = __shfl_down(tmpMax, 1);
			tmpK1 = __shfl_down(tmpK, 1);
			if (tmpMax < tmpMax1) {
				tmpMax = tmpMax1;
				tmpK = tmpK1;
			}
			tmpMax = __shfl(tmpMax, 0);
			tmpK = __shfl(tmpK, 0);
			MaxTree[tmpPosition] = tmpMax;
			MaxKTree[tmpPosition] = tmpK;

			tmpMax = 0.0;
			tmpMax1 = 0.0;
			tmpMax = MaxTree[laneId];
			tmpMax1 = __shfl_down(tmpMax, 16);
			tmpK = MaxKTree[laneId];
			tmpK1 = __shfl_down(tmpK, 16);

			if (tmpMax < tmpMax1) {
				tmpMax = tmpMax1;
				tmpK = tmpK1;
			}
			tmpMax1 = __shfl_down(tmpMax, 8);
			tmpK1 = __shfl_down(tmpK, 8);
			if (tmpMax < tmpMax1) {
				tmpMax = tmpMax1;
				tmpK = tmpK1;
			}
			tmpMax1 = __shfl_down(tmpMax, 4);
			tmpK1 = __shfl_down(tmpK, 4);
			if (tmpMax < tmpMax1) {
				tmpMax = tmpMax1;
				tmpK = tmpK1;
			}
			tmpMax1 = __shfl_down(tmpMax, 2);
			tmpK1 = __shfl_down(tmpK, 2);
			if (tmpMax < tmpMax1) {
				tmpMax = tmpMax1;
				tmpK = tmpK1;
			}
			tmpMax1 = __shfl_down(tmpMax, 1);
			tmpK1 = __shfl_down(tmpK, 1);
			if (tmpMax < tmpMax1) {
				tmpMax = tmpMax1;
				tmpK = tmpK1;
			}
			if (laneId == 0) {
				MaxWT[2] = tmpMax;
				MaxK[2] = tmpK;
			}
		}

		__syncthreads();

	/*	float WTMax = MaxWT[0];
		float WTMax2 = MaxWT[1];
		float WTMax3 = MaxWT[2];*/
		unsigned short int WTMaxK = MaxK[0];
		unsigned short int WTSecondMaxK = MaxK[1];
		unsigned short int WTThirdMaxK = MaxK[2];
		if (threadIdx.x == 0) {
			deviceWordMaxTopic[wordId] = WTMaxK + 1;
			deviceWordSecondMaxTopic[wordId] = WTSecondMaxK + 1;
			deviceWordThirdMaxTopic[wordId] = WTThirdMaxK + 1;
		}
		for (int i = tokenStart + threadIdx.x; i < tokenEnd; i += blockDim.x) {
			deviceMaxTopic[i] = WTMaxK + 1;
			deviceSecondMaxTopic[i] = WTSecondMaxK + 1;
		}
		if (threadIdx.x == 0) Counter[0] = atomicAdd(&deviceCounter[0], 1);
		__syncthreads();

	}

}




__global__ void WTDen_Update_Kernel(unsigned short int *deviceTopic, int *deviceWTDense, int *deviceTLCount, int *deviceTLOffset, int *deviceWTOffset, int numOfWordD, unsigned int* deviceCounter)
{
	int globalId = threadIdx.x + blockIdx.x * blockDim.x;
	int laneId = threadIdx.x % 32;
	int warpId = globalId / 32;
	unsigned int Counter;


	if (laneId == 0) {

		Counter = atomicAdd(&deviceCounter[0], 1);
	}
	Counter = __shfl(Counter, 0);

	while (Counter < numOfWordD)
		
	{
		int wordId = Counter;
		unsigned short int tmpTopic;
		int tmpWTOffset = deviceWTOffset[wordId];
		int tmpTLOffset = deviceTLOffset[wordId];

		for (int k = laneId; k < deviceTLCount[wordId]; k += 32)
		{
			tmpTopic = deviceTopic[tmpTLOffset + k];
			atomicAdd(&deviceWTDense[tmpWTOffset + tmpTopic - 1], 1);
		}

		if (laneId == 0)  Counter = atomicAdd(&deviceCounter[0], 1);
		Counter = __shfl(Counter, 0);

	}



}

__global__ void WTDen_Sum_Update_Kernel(int *deviceWTDense, int *deviceWTRowSum, int *deviceWTOffset, int numOfWordD)
{

	int input;
	int tid = threadIdx.x;
	int globalId = threadIdx.x + blockIdx.x * blockDim.x;
	int blockId = blockIdx.x;
	int tmpIndex;

	for (int k = globalId; k < K; k += GridDim*BlockDim)
	{
		for (int i = 0; i < numOfWordD; i ++)
		{
			tmpIndex = deviceWTOffset[i]  + k;
			deviceWTRowSum[k] += deviceWTDense[tmpIndex];

		}
	}
	__syncthreads();



}





__global__ void sparseMatrixAdd(int* argCount0, int* argOffset0, int* argNZCount0, unsigned short int* argIndex0, unsigned short int* argValue0, int* argCount1, int* argOffset1, int* argNZCount1, unsigned short int* argIndex1, unsigned short int* argValue1, int* argDense, int argNumRows, unsigned int* deviceCounter, int* argWTRowSum, int numOfWordD)
{

	int globalId = threadIdx.x + blockIdx.x * blockDim.x;
	int laneId = threadIdx.x % 32;
	int warpId = globalId / 32;
	int iterCounter = 0;
	unsigned int Counter;

	if (laneId == 0) {

		Counter = atomicAdd(&deviceCounter[0], 1);
	}
	Counter = __shfl(Counter, 0);

	while (Counter < argNumRows)
		//while (warpId + iterCounter * gridDim.x*blockDim.x / 32< argD)
	{
		int wordId = Counter;

		for (int k = laneId; k < K; k += 32)
		{
			argDense[k + K*warpId] = 0;
		}

		for (int k = laneId; k < argNZCount0[wordId]; k += 32)
		{
			int tmpIdx = argOffset0[wordId + numOfWordD] - K*numOfWordD + k;
			argDense[K*warpId + argIndex0[tmpIdx] - 1] += argValue0[tmpIdx];
		}

		for (int k = laneId; k < argNZCount1[wordId]; k += 32)
		{

			int tmpIdx = argOffset1[wordId] + k;
			atomicAdd(&argWTRowSum[argIndex1[tmpIdx] - 1], argValue1[tmpIdx]);
			argDense[K*warpId + argIndex1[tmpIdx] - 1] += argValue1[tmpIdx];
		}
		int noneZeroCount = 0;
		for (int k = laneId; k < K; k += 32) {
			int value = argDense[K*warpId + k];
			int flag = value > 0;
			int tmpNoneZeroCount = __popc(__ballot(value));

			if (tmpNoneZeroCount == 0) continue;

			flag += __shfl_up(flag, 1, 32)*(laneId >= 1);
			flag += __shfl_up(flag, 2, 32)*(laneId >= 2);
			flag += __shfl_up(flag, 4, 32)*(laneId >= 4);
			flag += __shfl_up(flag, 8, 32)*(laneId >= 8);
			flag += __shfl_up(flag, 16, 32)*(laneId >= 16);

			if (value) {
				int idx = argOffset0[wordId + numOfWordD] - K*numOfWordD + noneZeroCount + flag - 1;
				argIndex0[idx] = k + 1;
				argValue0[idx] = value;
			}
			noneZeroCount += tmpNoneZeroCount;

		}

		if (laneId == 0) {
			argNZCount0[wordId] = noneZeroCount;
			Counter = atomicAdd(&deviceCounter[0], 1);
		}
		Counter = __shfl(Counter, 0);


	}



}




__global__ void initRandState(curandState *state)
{
	int tid = blockIdx.x*blockDim.x + threadIdx.x;
	curand_init(clock() + tid, tid, 0, &state[tid]);
}



__global__ void LDAKernelTrain(float alpha, float beta, int* d_Index, unsigned short int* d_TopicIndex, int* d_SparseDTCount, unsigned short int* d_SparseDTIndex, int* d_SparseDTValue, int* d_TokenCountDT, int* d_TokenOffsetDT, int* d_DocListCount, int* d_DocListOffset, int* d_SparseWTCount, unsigned short int* d_SparseWTIndex, unsigned short int* d_SparseWTValue, int* d_TokenCount, int* d_TokenOffset, int* d_WordListCount, int* d_WordListOffset, int* d_WTRowSum, unsigned int* d_blockCounter, int*d_DocIndex, int D, int W, float* d_Perplexity, curandState *randState, float *WTHeadDense, int numOfWordD, int numOfWordS,  unsigned short int* deviceMaxTokenCount, unsigned short int* deviceMaxTopic, unsigned short int* deviceWordMaxTopic, unsigned short int* deviceWordSecondMaxTopic)

{
	int tid = threadIdx.x;
	int laneId = threadIdx.x % 32;
	int localId = threadIdx.x / 32;
	int blockId = blockIdx.x;
	volatile __shared__ float WTHead[K];
	volatile __shared__ float QTree[32];
	volatile __shared__ float WTMax[2];
	volatile __shared__ float STree[ShaMemSize / 32][K / 32];
	volatile __shared__ float prefixSumSample[ShaMemSize / 32][32];
	volatile __shared__ unsigned int Counter[1];
	__shared__ unsigned int WarpCounter[1];

	if (tid == 0) {
		Counter[0] = atomicAdd(&d_blockCounter[0], 1);	
	}
	__syncthreads();

	float sumPerplexity = 0.0;

	while (Counter[0]<numOfWordS)
	{
		int wordId = Counter[0]+ numOfWordD;
		if (localId == 0) {
			QTree[laneId] = 0;

		}
		float p_temp1 = 0.0;
		prefixSumSample[localId][laneId] = 0.0;
		long long tokenStart = d_TokenOffset[wordId];
		long long tokenEnd = d_TokenOffset[wordId] + d_TokenCount[wordId];
		long long WTStart = d_WordListOffset[wordId] - d_WordListOffset[numOfWordD];
		long long WTEnd = d_WordListOffset[wordId] - d_WordListOffset[numOfWordD] + d_SparseWTCount[wordId - numOfWordD];

		unsigned short int maxK = deviceWordMaxTopic[wordId];
		unsigned short int secondMaxK = deviceWordSecondMaxTopic[wordId];

		
		for (int i = tid; i < K; i += blockDim.x)
		{
			WTHead[i] = beta / (d_WTRowSum[i] + W*beta);
			
		}

		__syncthreads();

		for (int i = tid + WTStart; i < WTEnd; i += blockDim.x)
		{
			WTHead[d_SparseWTIndex[i] - 1] = (d_SparseWTValue[i] + beta) / (d_WTRowSum[d_SparseWTIndex[i] - 1] + W*beta);
			
		}
		
		__syncthreads();
		if (threadIdx.x == 0) {
			WTMax[0] = WTHead[maxK - 1];
			WTMax[1] = WTHead[secondMaxK - 1];
			WTHead[maxK - 1] = 0.0;
		}
		__syncthreads();
		

		for (int i = localId; i < K / 32; i += blockDim.x / 32) {
			unsigned short int   tmpK = i * 32 + laneId;
			//__syncthreads();
			float tmpVal = 0.0;
			tmpVal = alpha*WTHead[tmpK];
			tmpVal += __shfl_down(tmpVal, 16);
			tmpVal += __shfl_down(tmpVal, 8);
			tmpVal += __shfl_down(tmpVal, 4);
			tmpVal += __shfl_down(tmpVal, 2);
			tmpVal += __shfl_down(tmpVal, 1);
			tmpVal = __shfl(tmpVal, 0);
			QTree[i] = tmpVal;

			
		}
		__syncthreads();


		if (localId == 0) {

			float value = QTree[laneId];
			value += __shfl_up(value, 1, 32)*(laneId >= 1);
			value += __shfl_up(value, 2, 32)*(laneId >= 2);
			value += __shfl_up(value, 4, 32)*(laneId >= 4);
			value += __shfl_up(value, 8, 32)*(laneId >= 8);
			value += __shfl_up(value, 16, 32)*(laneId >= 16);

			QTree[laneId] = value;



		}



		if (tid == 0) WarpCounter[0] = tokenStart;
		
		__syncthreads();


		float Q = QTree[31];
		int tokenIdx;

		if (laneId == 0)
		{
			tokenIdx = atomicAdd(&WarpCounter[0], 1);

		}
		tokenIdx = __shfl(tokenIdx, 0);

		//unsigned short int maxTopic = deviceMaxTopic[tokenIdx];
		float WTMaxProb = WTMax[0];
		float WTSecondMaxProb = WTMax[1];

		while (tokenIdx<tokenEnd)
		{

			//int docId = __ldg(&d_Index[d_TopicIndex[tokenIdx]]);
			int oldZ=d_TopicIndex[tokenIdx];
			int docId = d_DocIndex[tokenIdx];

			
			unsigned short int maxTokenCount = deviceMaxTokenCount[tokenIdx];
			int totalTokenCount = d_TokenCountDT[docId - 1];
			float maxProbability = (maxTokenCount+alpha)*WTMaxProb;
			float maxS = (totalTokenCount - maxTokenCount)*WTSecondMaxProb;
			float thresProb = maxProbability / (maxProbability + maxS + Q);

			if (maxS < 0) printf("wrong maxS\n");
			float u;
			if (laneId == 0)u = curand_uniform(&(randState[threadIdx.x + blockDim.x*blockIdx.x])) / 1.00001;
			u = __shfl(u, 0);

			int newZ = 1;
			unsigned short int sFlag = 1;

			if (u < thresProb) {
		
				newZ = maxK;
				//if (laneId == 0) {
				//	//newZ = maxTopic;
				//	/*atomicAdd(&d_WTDenseCopy[WTStart + newZ - 1], 1);*/
				//	//d_TopicIndex[tokenIdx] = newZ;
				//	tokenIdx = atomicAdd(&WarpCounter[0], 1);
				//}
				//tokenIdx = __shfl(tokenIdx, 0);
				/*continue;*/
				sFlag = 0;
				goto endloop;

			}
			else{
			//computing S.
				float S = 0;
				long long DTStart = d_DocListOffset[docId - 1];
				long long DTEnd = d_DocListOffset[docId - 1] + ((d_SparseDTCount[docId - 1] - 1) / 32 + 1) * 32;

				//long long DTEnd = d_DocListOffset[docId - 1] + d_SparseDTCount[docId - 1];


				STree[localId][laneId] = 0;
				// //__syncthreads();
				int SIdx = 0;
				float  tmpP1k = 0.0;
				int   colVal;
				unsigned short int  colK;
				
				//maxStruct maxValue;
				for (int tmpIdx = DTStart + laneId; tmpIdx < DTEnd - 32; tmpIdx += 32) {

					colVal = d_SparseDTValue[tmpIdx];
					colK = d_SparseDTIndex[tmpIdx];
					tmpP1k = colVal*WTHead[colK - 1];	
					//if (colK == maxTopic) tmpP1k = 0.0;


					tmpP1k += __shfl_down(tmpP1k, 16);
					tmpP1k += __shfl_down(tmpP1k, 8);
					tmpP1k += __shfl_down(tmpP1k, 4);
					tmpP1k += __shfl_down(tmpP1k, 2);
					tmpP1k += __shfl_down(tmpP1k, 1);
					tmpP1k = __shfl(tmpP1k, 0);

					S += tmpP1k;
					STree[localId][SIdx] = S;

					SIdx++;
				}

				tmpP1k = 0.0;
				colVal = d_SparseDTValue[DTEnd - 32 + laneId];
				colK = d_SparseDTIndex[DTEnd - 32 + laneId];
				if (colK != 0) tmpP1k = colVal*WTHead[colK - 1];
				//if (colK == maxTopic) tmpP1k = 0.0;

				tmpP1k += __shfl_down(tmpP1k, 16);
				tmpP1k += __shfl_down(tmpP1k, 8);
				tmpP1k += __shfl_down(tmpP1k, 4);
				tmpP1k += __shfl_down(tmpP1k, 2);
				tmpP1k += __shfl_down(tmpP1k, 1);
				tmpP1k = __shfl(tmpP1k, 0);
				S += tmpP1k;
				STree[localId][SIdx] = S;


				//__syncthreads();
				/*STmp = S;

				S = __shfl(STmp, 0);*/
				S = __shfl(S, 0);
				//__syncthreads();
				//randomly generate u.


				/*if (maxProbability / (maxProbability + S + Q) < thresProb) printf("Wrong Prob!!!!%f,%f,%d\n", maxProbability / (maxProbability + S + Q), thresProb, maxK);*/

				if (u < maxProbability / (maxProbability + S + Q)) {

					newZ = maxK;

				}

				else if ((u >= maxProbability / (maxProbability + S + Q)) && (u< (maxProbability + S) / (maxProbability + S + Q)))
				{
					//float transU = u*(S + Q);
					float transU = u*(maxProbability + S + Q) - maxProbability;

					float tmpSumHigh, tmpSumLow = 0.0;
					tmpSumHigh = STree[localId][laneId];
					tmpSumLow = __shfl_up(tmpSumHigh, 1, 32);
					if (laneId == 0)tmpSumLow = 0;
					int voteFlag = 0;
					if ((transU < tmpSumHigh)) voteFlag = 1;
					int lvl1Idx = __ffs(__ballot(voteFlag)) - 1;
					//int overflowFlag = 0;

					if (lvl1Idx < 0) lvl1Idx = (DTEnd - DTStart) / 32 - 1;
					
					transU = transU - tmpSumLow;

					transU = __shfl(transU, lvl1Idx);
					int tmpIdx = DTStart + lvl1Idx * 32 + laneId;
					int tmpNewZ = d_SparseDTIndex[tmpIdx];
					int colVal = d_SparseDTValue[tmpIdx];
					float p1k = 0.0;
					if (tmpNewZ != 0)
					{
						p1k = colVal*WTHead[tmpNewZ - 1];
					}
					prefixSumSample[localId][laneId] = p1k;
					float value = prefixSumSample[localId][laneId];
					value += __shfl_up(value, 1, 32)*(laneId >= 1);
					value += __shfl_up(value, 2, 32)*(laneId >= 2);
					value += __shfl_up(value, 4, 32)*(laneId >= 4);
					value += __shfl_up(value, 8, 32)*(laneId >= 8);
					value += __shfl_up(value, 16, 32)*(laneId >= 16);
					prefixSumSample[localId][laneId] = value;
					float tmpSum = prefixSumSample[localId][laneId];
					voteFlag = 0;
					if (transU < tmpSum) voteFlag = 1;
					int offset = __ffs(__ballot(voteFlag)) - 1;
					// int tmpoffset=0;
					if (offset<0) offset = 0;

					// tmpoffset=__ldg(&d_SparseDTCount[docId - 1])-lvl1Idx*32-1;
					newZ = __shfl(tmpNewZ, offset);
					// if ((newZ < 1) || (newZ > K)) {
					// 	printf("wrong Index from sampling Dense:%d,%f,%f,%f,%f\n", newZ, u - S / (S + Q),u,S,Q);
					// 	printf("TmpNewZ and offset: %d,%d\n",tmpNewZ,offset);
					// 	printf("transU and tmpSum and voteFlag: %.10f,%.10f,%d\n",transU,tmpSum,voteFlag);
					// }
					if ((newZ == 0) || (newZ > K)) {
						int tmpoffset = d_SparseDTCount[docId - 1] - lvl1Idx * 32 - 1;
						newZ = __shfl(tmpNewZ, tmpoffset);
						// printf("Dense part:NewZ , tmpNewZ and tmpoffset: %d,%d,%d\n",newZ,tmpNewZ,tmpoffset);
					}

				}

				else //bucket Q
				{

					//float transU = (u - S / (S + Q))*(S + Q);

					float transU = (u - (maxProbability + S) / (maxProbability + S + Q))*(maxProbability + S + Q);
					//level 1: decide position
					float tmpSumHigh, tmpSumLow = 0.0;
					tmpSumHigh = QTree[laneId];
					tmpSumLow = __shfl_up(tmpSumHigh, 1, 32);
					if (laneId == 0)tmpSumLow = 0;
					//voting for lvl1Idx
					int voteFlag = 0;
					if (transU < tmpSumHigh) voteFlag = 1;
					int lvl1Idx = __ffs(__ballot(voteFlag)) - 1;
					if (lvl1Idx < 0) lvl1Idx = 31;
					transU = transU - tmpSumLow;
					transU = __shfl(transU, lvl1Idx);
					prefixSumSample[localId][laneId] = alpha*WTHead[32 * lvl1Idx + laneId];
					//accumulation

					float value = prefixSumSample[localId][laneId];
					value += __shfl_up(value, 1, 32)*(laneId >= 1);
					value += __shfl_up(value, 2, 32)*(laneId >= 2);
					value += __shfl_up(value, 4, 32)*(laneId >= 4);
					value += __shfl_up(value, 8, 32)*(laneId >= 8);
					value += __shfl_up(value, 16, 32)*(laneId >= 16);

					prefixSumSample[localId][laneId] = value;

					voteFlag = 0;
					tmpSumLow = 0;
					tmpSumHigh = prefixSumSample[localId][laneId];
					tmpSumLow = __shfl_up(tmpSumHigh, 1, 32);
					if (laneId == 0)tmpSumLow = 0;

					if (transU < tmpSumHigh)voteFlag = 1;
					int lvl2Idx = __ffs(__ballot(voteFlag)) - 1;
					if (lvl2Idx < 0)lvl2Idx = 31;
					newZ = lvl1Idx * 32 + lvl2Idx + 1;

					if ((newZ < 1) || (newZ > K)) {
						printf("wrong Index from sampling Dense else :%d,%f,%f,%f,%f\n", newZ, u - S / (S + Q), u, S, Q);
					}


				}
			}

		endloop:

			if (laneId == 0) {
				d_TopicIndex[tokenIdx] = newZ;
				//deviceMaxTopic[tokenIdx] = newZ;

				/*sumPerplexity += log((S + maxProbability + Q) / (totalTokenCount + K*alpha));*/

				sumPerplexity += 1.0;
				/*if(oldZ==newZ) sumPerplexity += 1.0;*/
				//d_Perplexity[tokenIdx] = log((S + Q) / (d_TokenCountDT[docId - 1] + K*alpha));

				//d_Perplexity[tokenIdx] = 1.0;
				// printf("Perplexity:%f, %d, %d, %d, %d\n",d_Perplexity[tokenIdx],tokenStart,tokenIdx,newZ,wordId);
				// printf("Perplexity: %d\n",tokenStart);

				tokenIdx = atomicAdd(&WarpCounter[0], 1);

				// sumPerplexity += log((S + Q) / (d_TokenCountDT[docId - 1] + K*alpha));

			}


			tokenIdx = __shfl(tokenIdx, 0);

		}

		if (tid == 0) Counter[0] = atomicAdd(&d_blockCounter[0], 1);

		__syncthreads();

	}

	if (laneId == 0) QTree[localId] = sumPerplexity;

	__syncthreads();

	if (localId == 0) {
		float perplexity = 0.0;
		perplexity = QTree[laneId] * (laneId < BlockDim / 32);
		perplexity += __shfl_down(perplexity, 16);
		perplexity += __shfl_down(perplexity, 8);
		perplexity += __shfl_down(perplexity, 4);
		perplexity += __shfl_down(perplexity, 2);
		perplexity += __shfl_down(perplexity, 1);
		if (laneId == 0) d_Perplexity[blockId] += perplexity;
	}


}

__device__ volatile int sem = 0;
__device__ unsigned int subCount = 0;

__device__ void acquire_semaphore(volatile int *lock) {
	while (atomicCAS((int *)lock, 0, 1) != 0);
}

__device__ void release_semaphore(volatile int *lock) {
	*lock = 0;
	__threadfence();
}

__global__ void LDAKernelTrainD(float alpha, float beta, int* d_Index, unsigned short int* d_TopicIndex, int* d_SparseDTCount, unsigned short int* d_SparseDTIndex, int* d_SparseDTValue, int* d_TokenCountDT, int* d_TokenOffsetDT, int* d_DocListCount, int* d_DocListOffset, int* d_WTDense, int* d_WTDenseCopy, int* d_TokenCount, int* d_TokenOffset, int* d_WordListCount, int* d_WordListOffset, int* d_WTRowSum, unsigned int* d_blockCounter, int*d_DocIndex, int D, int W, float* d_Perplexity, curandState *randState, float *WTHeadDense, int numOfWordD, unsigned short int* deviceMaxTokenCount, unsigned short int* deviceMaxTopic, unsigned short int* deviceWordMaxTopic, unsigned short int* deviceWordSecondMaxTopic, float* deviceMaxProb, float* deviceThresProb,float* deviceTimeRecord, int tokenSegment, float* deviceRandomfloat, int* deviceEffectiveTokenIndex, int* deviceNewTokenCount)

{
	
	int laneId = threadIdx.x % 32;
	int localId = threadIdx.x / 32;

	volatile __shared__ float WTHead[K];
	volatile __shared__ float QTree[32];



	volatile __shared__ float WTMax[2];

	volatile __shared__ float STree[ShaMemSize / 32][K / 32];
	volatile __shared__ float prefixSumSample[ShaMemSize / 32][32];
	volatile __shared__ unsigned int Counter[1];
	__shared__ unsigned int tokenRegionStart[1];
	volatile __shared__ unsigned int tokenEndFlag[1];
	__shared__ unsigned int WarpCounter[1];

	/*
	clock_t start0, finish0, finish1, finish2, finish3, finish4;
	double costtime0 = 0.0, costtime1 = 0.0, costtime2 = 0.0, costtime3 = 0.0, costtime4 = 0.0, total=0.0;*/
	
	clock_t start0, finish0, finish1;
	double costtime0 = 0.0, costtime1 = 0.0;

	

	if (threadIdx.x == 0)
	{
		acquire_semaphore(&sem);
		tokenEndFlag[0] = 0;
		Counter[0] = d_blockCounter[0];
		unsigned int numRegions = (deviceNewTokenCount[Counter[0]] == 0) ? 0 : ((deviceNewTokenCount[Counter[0]] - 1) / tokenSegment);
		tokenRegionStart[0] = atomicInc(&subCount, numRegions);
		if (subCount == 0) {
			d_blockCounter[0] = d_blockCounter[0] + 1;
			tokenEndFlag[0] = 1;
		}
		release_semaphore(&sem);
	}
	__syncthreads();

	/*if(threadIdx.x==0){
		Counter[0]=atomicAdd(&d_blockCounter[0],1);
	}
	__syncthreads();*/

	float sumPerplexity = 0.0;

	//start0 = clock64();


	while (Counter[0]<numOfWordD)
	{
		/*start0 = clock64();*/

		int wordId = Counter[0] ;
		if (localId == 0) {
			QTree[laneId] = 0;
		}

		prefixSumSample[localId][laneId] = 0.0;
		int tokenStart = d_TokenOffset[wordId]+ tokenRegionStart[0]*tokenSegment;
		int tokenEnd = d_TokenOffset[wordId] + (tokenRegionStart[0]+1) * tokenSegment;
		if (tokenEndFlag[0]) tokenEnd = d_TokenOffset[wordId] + deviceNewTokenCount[wordId];
		int WTStart = d_WordListOffset[wordId];
		unsigned short int maxK = deviceWordMaxTopic[wordId];
		unsigned short int secondMaxK = deviceWordSecondMaxTopic[wordId];

		// Reconstruct dense WT vector from sparse WT matrix
		for (int i = threadIdx.x; i < K; i += blockDim.x)
		{
			WTHead[i] = (d_WTDense[WTStart + i] + beta) / (d_WTRowSum[i] + W*beta);
			//__syncthreads();
		}
		__syncthreads();

		if (threadIdx.x == 0) {
			WTMax[0] = WTHead[maxK - 1];
			WTMax[1] = WTHead[secondMaxK - 1];
			WTHead[maxK - 1] = 0.0;
		}
		__syncthreads();


		for (int i = localId; i < K / 32; i += blockDim.x / 32) {
			short int   tmpK = i * 32 + laneId;
			float tmpVal = 0.0;
			tmpVal = alpha*WTHead[tmpK];
			tmpVal += __shfl_down(tmpVal, 16);
			tmpVal += __shfl_down(tmpVal, 8);
			tmpVal += __shfl_down(tmpVal, 4);
			tmpVal += __shfl_down(tmpVal, 2);
			tmpVal += __shfl_down(tmpVal, 1);
			tmpVal = __shfl(tmpVal, 0);
			QTree[i] = tmpVal;

		}
		__syncthreads();

		if (localId == 0) {

			float value = QTree[laneId];
			value += __shfl_up(value, 1, 32)*(laneId >= 1);
			value += __shfl_up(value, 2, 32)*(laneId >= 2);
			value += __shfl_up(value, 4, 32)*(laneId >= 4);
			value += __shfl_up(value, 8, 32)*(laneId >= 8);
			value += __shfl_up(value, 16, 32)*(laneId >= 16);

			QTree[laneId] = value;

		}

		if (threadIdx.x == 0) WarpCounter[0] = tokenStart;
		__syncthreads();

		//float WTMax = MaxWT[0];
		//float WTMax2 = MaxWT[1];
		//unsigned short int WTMaxK = MaxK[0];

		float Q = QTree[31];
		int tokenIdx;

		if (laneId == 0)
		{
			tokenIdx = atomicAdd(&WarpCounter[0], 1);

		}
		tokenIdx = __shfl(tokenIdx, 0);

		
		// float WTMaxProb = WTMax[0];
		// float WTSecondMaxProb = WTMax[1];

		// for (int tokenIdx = tokenStart + localId; tokenIdx < tokenEnd; tokenIdx += blockDim.x / 32) //iterate over tokens
		// {

		/*finish0 = clock64();
		costtime0 += (double)(finish0 - start0);*/

		while (tokenIdx<tokenEnd)
		{
			////int oldZ = d_TopicIndex[tokenIdx];
		 //   
			////unsigned short int sFlag = 1;
			//start0 = clock64();
			//
			//if (deviceMflag[tokenIdx]) {
			//	if (laneId==0) tokenIdx = atomicAdd(&WarpCounter[0], 1);
			//	tokenIdx = __shfl(tokenIdx, 0);
			//	finish0 = clock64();
			//	costtime0 += (double)(finish0 - start0);
			//	continue;
			//	//newZ = maxK;
			//	//sFlag = 0;
			//	//goto endloop;

			//}


			int tokenNewIdx = deviceEffectiveTokenIndex[tokenIdx];

			unsigned short int newZ = 1;
			int docId = d_DocIndex[tokenNewIdx]-1;
			float maxProbability = deviceMaxProb[tokenNewIdx];
			float thresProb = deviceThresProb[tokenNewIdx];
			float u = deviceRandomfloat[tokenNewIdx];

		/*	if (laneId == 0)u = curand_uniform(&(randState[threadIdx.x + blockDim.x*blockIdx.x])) / 1.00001;
			u = __shfl(u, 0);*/

			//computing S.
			float S = 0;

			int DTStart = d_DocListOffset[docId];
			int DTEnd = d_DocListOffset[docId] + ((d_SparseDTCount[docId] - 1) / 32 + 1) * 32;

			STree[localId][laneId] = 0;
			short int SIdx = 0;
			float  tmpP1k = 0.0;
			short int  colVal;
			short int  colK;
			/*finish1 = clock64();
			costtime1 += (double)(finish1 - finish0);*/
			//maxStruct maxValue;
			for (int tmpIdx = DTStart + laneId; tmpIdx < DTEnd - 32; tmpIdx += 32) {

				colVal = d_SparseDTValue[tmpIdx];
				colK = d_SparseDTIndex[tmpIdx];
				tmpP1k = colVal*WTHead[colK - 1];
				//tmpP1k = tmpP1k*(colK != maxTopic);
				//if (colK == maxTopic) tmpP1k = 0.0;
				tmpP1k += __shfl_down(tmpP1k, 16);
				tmpP1k += __shfl_down(tmpP1k, 8);
				tmpP1k += __shfl_down(tmpP1k, 4);
				tmpP1k += __shfl_down(tmpP1k, 2);
				tmpP1k += __shfl_down(tmpP1k, 1);
				tmpP1k = __shfl(tmpP1k, 0);
				S += tmpP1k;
				STree[localId][SIdx] = S;
				SIdx++;
			}

			tmpP1k = 0.0;
			colVal = d_SparseDTValue[DTEnd - 32 + laneId];
			colK = d_SparseDTIndex[DTEnd - 32 + laneId];
			if (colK != 0) tmpP1k = colVal*WTHead[colK - 1];
			//if (colK == maxTopic) tmpP1k = 0.0;
			//tmpP1k = tmpP1k*(colK != maxTopic);

			tmpP1k += __shfl_down(tmpP1k, 16);
			tmpP1k += __shfl_down(tmpP1k, 8);
			tmpP1k += __shfl_down(tmpP1k, 4);
			tmpP1k += __shfl_down(tmpP1k, 2);
			tmpP1k += __shfl_down(tmpP1k, 1);
			tmpP1k = __shfl(tmpP1k, 0);
			S += tmpP1k;
			STree[localId][SIdx] = S;


			//__syncthreads();
			/*STmp = S;

			S = __shfl(STmp, 0);*/
			S = __shfl(S, 0);
			//__syncthreads();
			//randomly generate u.

			float totalProb = maxProbability+S+Q;

			if (maxProbability / totalProb <thresProb) printf("What!!!!%f,%f,%d,%d\n", maxProbability / totalProb, thresProb, maxK - 1,  wordId);

			/*finish2 = clock64();
			costtime2 += (double)(finish2 - finish1);*/

			/*if ((wordId == 40) && (laneId == 0) && (tokenIdx - tokenStart<50)) printf("thresProb: %f,%f,%d,%d,%f,%f\n", thresProb, (maxProbability + alpha*(WTHead[maxTopic - 1])) / (maxProbability + S + Q), maxTokenCount, totalTokenCount, WTHead[maxTopic - 1] / Q, WTMax / Q);*/

			//if (maxProbability / (maxProbability + S + Q) < thresProb) printf("Wrong Prob!!!!%f,%f\n", maxProbability / (maxProbability + S + Q), thresProb);

				

			//if (maxProbability / (maxProbability + S + Q) < thresProb) printf("Wrong Prob!!!!");

			if (u < maxProbability / totalProb) {

				newZ = maxK;

			}
				

			else if ((u>= maxProbability / totalProb) && (u< (maxProbability + S) / totalProb))
			{
				//float transU = u*(S + Q);
				float transU = u*totalProb- maxProbability;
				float tmpSumHigh, tmpSumLow = 0.0;
				tmpSumHigh = STree[localId][laneId];
				tmpSumLow = __shfl_up(tmpSumHigh, 1, 32);
				if (laneId == 0) tmpSumLow = 0;
				int voteFlag = 0;
				if ((transU < tmpSumHigh)) voteFlag = 1;
				int lvl1Idx = __ffs(__ballot(voteFlag)) - 1;

				if (lvl1Idx < 0) lvl1Idx = (DTEnd - DTStart) / 32 - 1;
				//tmpU1 = transU;
				transU = transU - tmpSumLow;
				/*tmpU = transU;*/
				transU = __shfl(transU, lvl1Idx);
				int tmpIdx = DTStart + lvl1Idx * 32 + laneId;
				int tmpNewZ = d_SparseDTIndex[tmpIdx];
				int colVal = d_SparseDTValue[tmpIdx];
				float p1k = 0.0;
				if (tmpNewZ != 0)
				{
					p1k = colVal*WTHead[tmpNewZ - 1];
				}

				//p1k = p1k*(colK != maxTopic);

				prefixSumSample[localId][laneId] = p1k;
				float value = prefixSumSample[localId][laneId];
				value += __shfl_up(value, 1, 32)*(laneId >= 1);
				value += __shfl_up(value, 2, 32)*(laneId >= 2);
				value += __shfl_up(value, 4, 32)*(laneId >= 4);
				value += __shfl_up(value, 8, 32)*(laneId >= 8);
				value += __shfl_up(value, 16, 32)*(laneId >= 16);
				prefixSumSample[localId][laneId] = value;
				float tmpSum = prefixSumSample[localId][laneId];
				voteFlag = 0;
				if (transU < tmpSum) voteFlag = 1;
				int offset = __ffs(__ballot(voteFlag)) - 1;
				// int tmpoffset=0;
				if(offset<0) offset=0;

				// tmpoffset=__ldg(&d_SparseDTCount[docId - 1])-lvl1Idx*32-1;
				newZ = __shfl(tmpNewZ, offset);
				// if ((newZ < 1) || (newZ > K)) {
				// 	printf("wrong Index from sampling Dense:%d,%f,%f,%f,%f\n", newZ, u - S / (S + Q),u,S,Q);
				// 	printf("TmpNewZ and offset: %d,%d\n",tmpNewZ,offset);
				// 	printf("transU and tmpSum and voteFlag: %.10f,%.10f,%d\n",transU,tmpSum,voteFlag);
				// }
				if ((newZ == 0) || (newZ > K)){
					int tmpoffset=d_SparseDTCount[docId]-lvl1Idx*32-1;
					newZ=__shfl(tmpNewZ, tmpoffset);
					// printf("Dense part:NewZ , tmpNewZ and tmpoffset: %d,%d,%d\n",newZ,tmpNewZ,tmpoffset);
				}

			}

			else //bucket Q
			{

				//float transU = (u - S / (S + Q))*(S + Q);

				float transU = (u - (maxProbability + S) / totalProb)*totalProb;
				//level 1: decide position
				float tmpSumHigh, tmpSumLow = 0.0;
				tmpSumHigh = QTree[laneId];
				tmpSumLow = __shfl_up(tmpSumHigh, 1, 32);
				if (laneId == 0)tmpSumLow = 0;
				//voting for lvl1Idx
				int voteFlag = 0;
				if (transU < tmpSumHigh) voteFlag = 1;
				int lvl1Idx = __ffs(__ballot(voteFlag)) - 1;
				if (lvl1Idx < 0) lvl1Idx = 31;
				/*tmpU1 = transU;*/
				transU = transU - tmpSumLow;
				/*tmpU = transU;*/
				transU = __shfl(transU, lvl1Idx);
				prefixSumSample[localId][laneId] = alpha*WTHead[32 * lvl1Idx + laneId];
				//accumulation

				float value = prefixSumSample[localId][laneId];
				value += __shfl_up(value, 1, 32)*(laneId >= 1);
				value += __shfl_up(value, 2, 32)*(laneId >= 2);
				value += __shfl_up(value, 4, 32)*(laneId >= 4);
				value += __shfl_up(value, 8, 32)*(laneId >= 8);
				value += __shfl_up(value, 16, 32)*(laneId >= 16);

				prefixSumSample[localId][laneId] = value;

				voteFlag = 0;
				tmpSumLow = 0;
				tmpSumHigh = prefixSumSample[localId][laneId];
				tmpSumLow = __shfl_up(tmpSumHigh, 1, 32);
				if (laneId == 0)tmpSumLow = 0;

				if (transU < tmpSumHigh)voteFlag = 1;
				int lvl2Idx = __ffs(__ballot(voteFlag)) - 1;
				if (lvl2Idx < 0)lvl2Idx = 31;
				newZ = lvl1Idx * 32 + lvl2Idx + 1;

				if ((newZ < 1) || (newZ > K)) {
					printf("wrong Index from sampling Dense else :%d,%f,%f,%f,%f\n", newZ, u - S / (S + Q),u,S,Q);
				}

			}

			if (laneId == 0) {
				d_TopicIndex[tokenNewIdx] = newZ;
				//deviceMaxTopic[tokenIdx] = newZ;

				atomicAdd(&d_WTDenseCopy[WTStart + newZ - 1], 1);

				//p_temp = S + Q;
				// d_S[tokenIdx] = Q;

				/*sumPerplexity+= log(totalProb / (d_TokenCountDT[docId] + K*alpha));*/
				sumPerplexity += 1.0;

				//if(oldZ==newZ) sumPerplexity += 1.0;

				//d_Perplexity[tokenIdx] = log((S + Q) / (d_TokenCountDT[docId - 1] + K*alpha));

				//d_Perplexity[tokenIdx] = 1.0;
				// printf("Perplexity:%f, %d, %d, %d, %d\n",d_Perplexity[tokenIdx],tokenStart,tokenIdx,newZ,wordId);
				// printf("Perplexity: %d\n",tokenStart);

				tokenIdx = atomicAdd(&WarpCounter[0], 1);

				// sumPerplexity += log((S + Q) / (d_TokenCountDT[docId - 1] + K*alpha));

			}

		//	// if(laneId==0) 
		//	// {

		//	// 	__threadfence_block();
		//	// }
		tokenIdx = __shfl(tokenIdx, 0);
		//	finish1 = clock64();
		//	costtime1 += (double)(finish1 - start0);


		//	/*finish4 = clock64();
		//	costtime4 += (double)(finish4 - finish3);*/
	
		//		/*finish3 = clock64();
		//		costtime3 += (double)(finish3 - finish2);*/

		///*endloop:*/
		//	/*finish3 = clock64();*/
		//	


	        
		}

		/*if(threadIdx.x==0) Counter[0]=atomicAdd(&d_blockCounter[0],1);

		__syncthreads();*/
		if (threadIdx.x == 0)
		{
			acquire_semaphore(&sem);
			tokenEndFlag[0] = 0;
			Counter[0] = d_blockCounter[0];
			unsigned int numRegions = (deviceNewTokenCount[Counter[0]] == 0) ? 0 : ((deviceNewTokenCount[Counter[0]] - 1) / tokenSegment);
			tokenRegionStart[0] = atomicInc(&subCount, numRegions);
			if (subCount == 0) {
				d_blockCounter[0] = d_blockCounter[0] + 1;
				tokenEndFlag[0] = 1;
			}
			release_semaphore(&sem);
		}
		__syncthreads();
		//if (localId == 0) {
		//	unsigned short int topic = 0;
		//	topic = maxTopicVec[laneId] * (laneId < BlockDim / 32);
		//	topic += __shfl_down(topic, 16);
		//	topic += __shfl_down(topic, 8);
		//	topic += __shfl_down(topic, 4);
		//	topic += __shfl_down(topic, 2);
		//	topic += __shfl_down(topic, 1);
		//	if (laneId == 0) atomicAdd(&d_WTDenseCopy[WTStart + MaxK[0]], topic);
		//}
		//__syncthreads();

	}

	if (laneId == 0) QTree[localId] = sumPerplexity;
	__syncthreads();
	
	if (localId == 0) {
		float perplexity = 0.0;
		perplexity = QTree[laneId] * (laneId < BlockDim / 32);
		perplexity += __shfl_down(perplexity, 16);
		perplexity += __shfl_down(perplexity, 8);
		perplexity += __shfl_down(perplexity, 4);
		perplexity += __shfl_down(perplexity, 2);
		perplexity += __shfl_down(perplexity, 1);
		if (laneId == 0) d_Perplexity[blockIdx.x] += perplexity;
	}

	/*finish0 = clock64();
	costtime0 = (double)(finish0 - start0);*/

	//if (threadIdx.x + blockDim.x*blockIdx.x == 0) printf("costtime0,costtime1:%f,%f", costtime0 / (158200000 * 1.0), costtime1 / (158200000 * 1.0)); 

	/*deviceTimeRecord[(threadIdx.x + blockDim.x*blockIdx.x)/32] = costtime0 / (158200000 * 1.0);

	if (threadIdx.x+blockDim.x*blockIdx.x == 0) printf("costtime0,costtime1,costtime2,costtime3,costtime4,total:%f,%f,%f,%f,%f,%f", costtime0/(158200000*1.0), costtime1 / (158200000 * 1.0), costtime2 / (158200000 * 1.0), costtime3 / (158200000 * 1.0), costtime4 / (158200000 * 1.0), (costtime0 + costtime1+ costtime2+ costtime3+ costtime4)/ (158200000 * 1.0));*/
	//if (threadIdx.x % 32 == 0)
	//	d_Perplexity[(threadIdx.x + blockDim.x*blockIdx.x) / 32] = sumPerplexity;
	////wordPerplexity[(threadIdx.x + blockDim.x*blockIdx.x) / 32] = sumPerplexity;
	//__syncthreads();

}


__global__ void LDATrainPerplexityReduce(float *perplexity,float numOfTokens,float* devicePerplexityAve) {

	int tid = threadIdx.x;
	int laneId = threadIdx.x % 32;
	int localId = threadIdx.x / 32;
	
	float S = 0.0;
	volatile __shared__ float perplexityMid[32];
	for (int i = tid; i < GridDim; i += BlockDim) {

		float tmpPerplexity = 0.0;
		tmpPerplexity = perplexity[i];
		tmpPerplexity += __shfl_down(tmpPerplexity, 16);
		tmpPerplexity += __shfl_down(tmpPerplexity, 8);
		tmpPerplexity += __shfl_down(tmpPerplexity, 4);
		tmpPerplexity += __shfl_down(tmpPerplexity, 2);
		tmpPerplexity += __shfl_down(tmpPerplexity, 1);
		S += tmpPerplexity;
	}
	if (laneId == 0) perplexityMid[localId] = S;
	__syncthreads();
	if (localId == 0) {
		float AveragePerplexity = 0.0;
		S = 0.0;
		S = perplexityMid[laneId] * (laneId < BlockDim / 32);
		//printf("\nS=:%f\n", S);
		S += __shfl_down(S, 16);
		S += __shfl_down(S, 8);
		S += __shfl_down(S, 4);
		S += __shfl_down(S, 2);
		S += __shfl_down(S, 1);

		
		if (laneId == 0)
		{
			AveragePerplexity = S / numOfTokens; 
			devicePerplexityAve[0]=AveragePerplexity;
			printf("\nAverage Perplexity:%f\n", AveragePerplexity);
		}
		
		
	}



}





__global__ void LDATrainPerplexityReduce1(float *perplexity, float *perplexityMid, int numVals) {


	int numWarps = gridDim.x*blockDim.x / 32;
	int tid = threadIdx.x + blockIdx.x*blockDim.x;
	int warpId = tid / 32;
	int laneId = tid % 32;

	int perWarpSize = ((numVals + numWarps - 1) / numWarps + 31) / 32 * 32;
	int perWarpSizeMax = (numVals + numWarps - 1) / numWarps;
	int startIdx = perWarpSizeMax*warpId;
	int endIdx = perWarpSizeMax*warpId + perWarpSize;
	int endMax = perWarpSizeMax*warpId + perWarpSizeMax;
	
	float totalProd = 0.0;
	for (long long i = startIdx + laneId; i < endIdx; i += 32) {

		float tmpProd = 0.0;
		if ((i < numVals) && (i < endMax))tmpProd = perplexity[i];

		tmpProd += __shfl_down(tmpProd, 16);
		tmpProd += __shfl_down(tmpProd, 8);
		tmpProd += __shfl_down(tmpProd, 4);
		tmpProd += __shfl_down(tmpProd, 2);
		tmpProd += __shfl_down(tmpProd, 1);
		tmpProd = __shfl(tmpProd, 0);
		totalProd += tmpProd;
		//__syncthreads();
	}
	__syncthreads();
	if (laneId == 0) perplexityMid[warpId] += totalProd;

}



__global__ void UpdateProbKernelTrainD(float alpha, float beta, int* d_Index, unsigned short int* d_TopicIndex, int* d_SparseDTCount, unsigned short int* d_SparseDTIndex, int* d_SparseDTValue, int* d_TokenCountDT, int* d_TokenOffsetDT, int* d_DocListCount, int* d_DocListOffset, int* d_WTDense, int* d_WTDenseCopy, int* d_TokenCount, int* d_TokenOffset, int* d_WordListCount, int* d_WordListOffset, int* d_WTRowSum, unsigned int* d_blockCounter, int*d_DocIndex, int D, int W, float* d_Perplexity, curandState *randState, float *WTHeadDense, int numOfWordD, unsigned short int* deviceMaxTokenCount, unsigned short int* deviceMaxTopic, unsigned short int* deviceWordMaxTopic, unsigned short int* deviceWordSecondMaxTopic,float* deviceMaxProb, float* deviceThresProb, unsigned short int* deviceSecondMaxTokenCount, unsigned short int* deviceWordThirdMaxTopic, float* deviceRandomfloat,  int* deviceEffectiveTokenIndex, int* deviceNewTokenCount)

{

	int laneId = threadIdx.x % 32;
	int localId = threadIdx.x / 32;
	volatile __shared__ float WTHead[K];
	volatile __shared__ float QTree[32];
	volatile __shared__ float WTMax[3];
	volatile __shared__ unsigned int Counter[1];
	__shared__ unsigned int WarpCounter[1];

	if (threadIdx.x == 0) {
		Counter[0] = atomicAdd(&d_blockCounter[0], 1);
	}
	__syncthreads();
	float sumPerplexity = 0.0;

	while (Counter[0]<numOfWordD)
	{
		int wordId = Counter[0];
		if (localId == 0) {
			QTree[laneId] = 0;
		}
		int tokenStart = d_TokenOffset[wordId];
		int tokenEnd = d_TokenOffset[wordId] + d_TokenCount[wordId];
		int WTStart = d_WordListOffset[wordId];
		unsigned short int maxK = deviceWordMaxTopic[wordId];
		unsigned short int secondMaxK = deviceWordSecondMaxTopic[wordId];
		unsigned short int thirdMaxK = deviceWordThirdMaxTopic[wordId];
		// Reconstruct dense WT vector from sparse WT matrix
		for (int i = threadIdx.x; i < K; i += blockDim.x)
		{
			WTHead[i] = (d_WTDense[WTStart + i] + beta) / (d_WTRowSum[i] + W*beta);
			//__syncthreads();
		}
		__syncthreads();

		if (threadIdx.x == 0) {
			WTMax[0] = WTHead[maxK - 1];
			WTMax[1] = WTHead[secondMaxK - 1];
			WTMax[2] = WTHead[thirdMaxK - 1];
			WTHead[maxK - 1] = 0.0;
			//WTHead[secondMaxK - 1] = 0.0;
		}
		__syncthreads();


		for (int i = localId; i < K / 32; i += blockDim.x / 32) {
			unsigned short int   tmpK = i * 32 + laneId;
			float tmpVal = 0.0;
			tmpVal = alpha*WTHead[tmpK];
			tmpVal += __shfl_down(tmpVal, 16);
			tmpVal += __shfl_down(tmpVal, 8);
			tmpVal += __shfl_down(tmpVal, 4);
			tmpVal += __shfl_down(tmpVal, 2);
			tmpVal += __shfl_down(tmpVal, 1);
			tmpVal = __shfl(tmpVal, 0);
			QTree[i] = tmpVal;

		}
		__syncthreads();

		if (localId == 0) {

			float value = QTree[laneId];
			value += __shfl_up(value, 1, 32)*(laneId >= 1);
			value += __shfl_up(value, 2, 32)*(laneId >= 2);
			value += __shfl_up(value, 4, 32)*(laneId >= 4);
			value += __shfl_up(value, 8, 32)*(laneId >= 8);
			value += __shfl_up(value, 16, 32)*(laneId >= 16);

			QTree[laneId] = value;

		}
		if (threadIdx.x == 0) WarpCounter[0] = 0;
		__syncthreads();
		float Q = QTree[31];
		int tokenIdx;
		float WTMaxProb = WTMax[0];
		float WTSecondMaxProb = WTMax[1];
		float WTThirdMaxProb = WTMax[2];
		
		
		for (int tokenIdx = tokenStart + threadIdx.x; tokenIdx < tokenEnd; tokenIdx += blockDim.x)  
		{
			
			int docId = d_DocIndex[tokenIdx]-1;
			int nonSkipTokenIdx;
			float u = curand_uniform(&(randState[threadIdx.x + blockDim.x*blockIdx.x])) / 1.00001;
			deviceRandomfloat[tokenIdx] = u;
			unsigned short int maxTokenCount = deviceMaxTokenCount[tokenIdx];
			unsigned short int maxSecondTokenCount = deviceSecondMaxTokenCount[tokenIdx];
			int totalTokenCount = d_TokenCountDT[docId];
			deviceMaxProb[tokenIdx] = (maxTokenCount + alpha)*WTMaxProb;
			//float maxS = (totalTokenCount - maxTokenCount)*WTSecondMaxProb;
			float maxS = (totalTokenCount - maxTokenCount - maxSecondTokenCount)*WTThirdMaxProb + maxSecondTokenCount*WTSecondMaxProb;
			deviceThresProb[tokenIdx] = deviceMaxProb[tokenIdx] / (deviceMaxProb[tokenIdx] + maxS + Q);
			if (u < deviceThresProb[tokenIdx]) {

				
				d_TopicIndex[tokenIdx] = maxK;
				atomicAdd(&d_WTDenseCopy[WTStart + maxK - 1], 1);
				//sumPerplexity += 1;
			}
			else {
				nonSkipTokenIdx = atomicAdd(&deviceNewTokenCount[wordId], 1);
				deviceEffectiveTokenIndex[nonSkipTokenIdx + tokenStart] = tokenIdx;
			}
		}
		if (threadIdx.x == 0) Counter[0] = atomicAdd(&d_blockCounter[0], 1);
		__syncthreads();

		/*if (threadIdx.x == 0) deviceNewTokenCount[wordId] = WarpCounter[0];
		__syncthreads();
*/
	}

	sumPerplexity += __shfl_down(sumPerplexity, 16);
	sumPerplexity += __shfl_down(sumPerplexity, 8);
	sumPerplexity += __shfl_down(sumPerplexity, 4);
	sumPerplexity += __shfl_down(sumPerplexity, 2);
	sumPerplexity += __shfl_down(sumPerplexity, 1);
	sumPerplexity = __shfl(sumPerplexity, 0);

	if (laneId == 0) QTree[localId] = sumPerplexity;
	__syncthreads();

	if (localId == 0) {
		float perplexity = 0.0;
		perplexity = QTree[laneId] * (laneId < BlockDim / 32);
		perplexity += __shfl_down(perplexity, 16);
		perplexity += __shfl_down(perplexity, 8);
		perplexity += __shfl_down(perplexity, 4);
		perplexity += __shfl_down(perplexity, 2);
		perplexity += __shfl_down(perplexity, 1);
		if (laneId == 0) d_Perplexity[blockIdx.x] += perplexity;
	}

}






