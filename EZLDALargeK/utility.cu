#include "utility.cuh"

#define	BUFF_SIZE_LONG	100000


__device__ void prefix_sum(volatile int* p) {

	int tid = threadIdx.x;
	volatile __shared__ int p_temp[ShaMemSize + 1];
	int pTmp = 0;

	if (tid == blockDim.x - 1)
	{
		p_temp[tid + 1] = 0;
	}
	__syncthreads();
	p_temp[tid] = p[tid];
	__syncthreads();
	for (int i = 0; i <__ffs(blockDim.x) - 1; i++)
	{
		int step = 1 << (i + 1);
		int index = (tid + 1)*step - 1;
		pTmp = 0;
		__syncthreads();
		if (index < blockDim.x) {
			pTmp = p_temp[index - step / 2];
		}
		__syncthreads();
		if (index < blockDim.x) {
			p_temp[index] += pTmp;
		}
		__syncthreads();


	}
	__syncthreads();
	if (tid == blockDim.x - 1)
	{
		pTmp = p_temp[tid];
		p_temp[tid + 1] = pTmp;
		p_temp[tid] = 0;
	}
	__syncthreads();
	for (int j = 0; j <__ffs(blockDim.x) - 1; j++)
	{
		int step = blockDim.x >> j;
		int index = (tid + 1)*step - 1;
		pTmp = 0;
		__syncthreads();
		if (index < blockDim.x) {
			pTmp = p_temp[index - step / 2];
		}
		__syncthreads();
		if (index < blockDim.x) {
			p_temp[index] += pTmp;
		}
		__syncthreads();
		pTmp = 0;
		__syncthreads();
		if (index < blockDim.x) {
			//p_temp[index] += p_temp[index - step / 2];
			pTmp = p_temp[index];
		}
		__syncthreads();

		if (index < blockDim.x) {
			//p_temp[index] += p_temp[index - step / 2];
			p_temp[index - step / 2] = pTmp - p_temp[index - step / 2];
		}
		__syncthreads();

	}
	__syncthreads();
	p[tid] = p_temp[tid + 1];
	__syncthreads();
	//p_temp[tid] = 0;
	/*if (tid == blockDim.x - 1)
	{
	p_temp[tid + 1] = 0;

	}
	__syncthreads();*/

}


__device__ void radix_sort(volatile int* p) {
	volatile __shared__ int p_input[ShaMemSize];
	volatile __shared__ int p_split[ShaMemSize];
	volatile __shared__ int p_split_inverse[ShaMemSize];
	volatile __shared__ int p_t[ShaMemSize];
	volatile __shared__ int p_index[ShaMemSize];
	int pTmp = 0;
	int tid = threadIdx.x;

	p_input[tid] = 0;
	p_split[tid] = 0;
	p_split_inverse[tid] = 0;
	p_t[tid] = 0;
	p_index[tid] = 0;
	p_input[tid] = p[tid];
	__syncthreads();
	for (int i = 0; i < 32; i++) {
		int p_totalFalses = 0;
		p_split[tid] = (p_input[tid] >> i) & 1;
		__syncthreads();
		p_split_inverse[tid] = (p_split[tid] + 1) & 1;
		__syncthreads();
		p_totalFalses = p_split_inverse[blockDim.x - 1];
		__syncthreads();
		for (int j = 0; j <__ffs(blockDim.x) - 1; j++)
		{
			int step = 1 << (j + 1);
			int index = (tid + 1)*step - 1;
			pTmp = 0;
			__syncthreads();
			if (index < blockDim.x) {
				pTmp = p_split_inverse[index - step / 2];
			}
			__syncthreads();
			if (index < blockDim.x) {
				p_split_inverse[index] += pTmp;
			}
			__syncthreads();


		}
		__syncthreads();
		if (tid == blockDim.x - 1)
		{
			p_split_inverse[tid] = 0;
		}
		__syncthreads();
		for (int j = 0; j <__ffs(blockDim.x) - 1; j++)
		{
			int step = blockDim.x >> j;
			int index = (tid + 1)*step - 1;
			pTmp = 0;
			__syncthreads();
			if (index < blockDim.x) {
				pTmp = p_split_inverse[index - step / 2];
				//p_split_inverse[index - step / 2] = p_split_inverse[index] - p_split_inverse[index - step / 2];
			}
			__syncthreads();
			if (index < blockDim.x) {
				p_split_inverse[index] += pTmp;
				//p_split_inverse[index - step / 2] = p_split_inverse[index] - p_split_inverse[index - step / 2];
			}
			__syncthreads();

			pTmp = 0;
			if (index < blockDim.x) {
				//p_split_inverse[index] += p_split_inverse[index - step / 2];
				pTmp = p_split_inverse[index];
			}
			__syncthreads();

			if (index < blockDim.x) {
				//p_split_inverse[index] += p_split_inverse[index - step / 2];
				p_split_inverse[index - step / 2] = pTmp - p_split_inverse[index - step / 2];
			}
			__syncthreads();

		}
		__syncthreads();
		p_totalFalses += p_split_inverse[blockDim.x - 1];
		__syncthreads();
		p_t[tid] = tid - p_split_inverse[tid] + p_totalFalses;
		__syncthreads();
		p_index[tid] = (p_split[tid]>0) ? p_t[tid] : p_split_inverse[tid];
		__syncthreads();
		pTmp = p_input[tid];
		__syncthreads();
		p_input[p_index[tid]] = pTmp;
		__syncthreads();
	}
	p[tid] = p_input[tid];
	__syncthreads();

	/*p_input[tid] = 0;
	p_split[tid] = 0;
	p_split_inverse[tid] = 0;
	p_t[tid] = 0;
	p_index[tid] = 0;
	__syncthreads();*/

}

__device__ void index_value_count(volatile int* p, volatile int *index, volatile int *value) {

	volatile __shared__ int p_input[ShaMemSize];
	volatile __shared__ int adj_diff[ShaMemSize];

	volatile __shared__ int adj_diff_tmp[ShaMemSize];
	volatile __shared__ int nonezero_index[ShaMemSize];
	volatile __shared__ int nonezero_index_move[ShaMemSize];
	volatile __shared__ int p_input_tmp[ShaMemSize];
	int tid = threadIdx.x;
	int pTmp = 0;
	adj_diff[tid] = 0;
	nonezero_index[tid] = 0;
	nonezero_index_move[tid] = 0;
	p_input[tid] = p[tid];
	p_input_tmp[tid] = 0;
	adj_diff_tmp[tid] = 0;
	__syncthreads();
	if (tid <blockDim.x - 1)
	{
		p_input_tmp[tid] = p[tid + 1];
	}

	__syncthreads();
	if (tid <blockDim.x - 1)
	{
		adj_diff[tid] = p_input_tmp[tid] - p_input[tid];
	}
	__syncthreads();
	if (tid <blockDim.x - 1)
	{
		pTmp = (adj_diff[tid] != 0);
	}
	__syncthreads();
	if (tid <blockDim.x - 1)
	{
		adj_diff[tid] = pTmp;
	}
	__syncthreads();

	if (tid == (blockDim.x - 1))
	{
		adj_diff[tid] = 1;
	}
	__syncthreads();
	adj_diff_tmp[tid] = adj_diff[tid];
	__syncthreads();
	prefix_sum(adj_diff);
	//d_aout[tid] = adj_diff[tid];
	__syncthreads();
	if (adj_diff_tmp[tid])
		index[adj_diff[tid] - 1] = p_input[tid];
	__syncthreads();
	/*if (tid < blockDim.x - 1)
	{
	nonezero_index[tid] = (p_input_tmp[tid] - p_input[tid] != 0);
	}
	__syncthreads();
	if (tid == blockDim.x - 1) {
	nonezero_index[tid] = 1;
	}*/
	nonezero_index[tid] = adj_diff_tmp[tid];
	__syncthreads();
	if (nonezero_index[tid] != 0) {
		nonezero_index[tid] = tid + 1;
		//value[adj_diff[tid] - 1] = nonezero_index[tid];
	}
	__syncthreads();
	if (nonezero_index[tid] != 0) {
		//nonezero_index[tid] = tid + 1;
		value[adj_diff[tid] - 1] = nonezero_index[tid];
	}
	__syncthreads();

	if (tid < blockDim.x - 1)
		nonezero_index_move[tid + 1] = value[tid];
	__syncthreads();
	if ((value[tid] > 0) && (tid>0))
	{
		value[tid] = value[tid] - nonezero_index_move[tid];
	}
	__syncthreads();
	/*adj_diff[tid] = 0;
	nonezero_index[tid] = 0;
	nonezero_index_move[tid] = 0;
	p_input[tid] = 0;
	adj_diff_tmp[tid] = 0;
	p_input_tmp[tid] = 0;
	__syncthreads();*/


}

__device__ void dense_sparse_kernel(volatile int *p, int* index, int* value, int* count, int *d_sparse_slotcount, int *d_slotoffset, int *d_counter_0)
{
	volatile __shared__ int p_input[ShaMemSize];
	volatile __shared__ int p_nonezero[ShaMemSize];
	volatile __shared__ int p_temp[ShaMemSize];
	int p_blocksum = 0;
	int p_count = 0;
	int m = 0;
	int f = 0;
	m = d_slotoffset[blockIdx.x + (*d_counter_0)*gridDim.x];
	//int p_nonezero_tmp = 0;
	//__shared__ int p_temp[K];
	int tid = threadIdx.x;
	int blockId = blockIdx.x;
	p_input[tid] = 0;
	p_nonezero[tid] = 0;
	p_temp[tid] = 0;
	__syncthreads();
	for (int k = 0; k < ((K - 1) / blockDim.x + 1); k++)
	{
		if ((k*blockDim.x + tid) < K) {
			p_input[tid] = p[k*blockDim.x + tid + blockId*K];
			//p_nonezero[tid] = p_input[tid] && 1;
			//p[k] = 0;
		}
		__syncthreads();
		if ((k*blockDim.x + tid) < K) {
			//p_input[tid] = p[k];
			p_nonezero[tid] = p_input[tid] && 1;
			//p[k] = 0;
		}
		__syncthreads();
		/*if (p_nonezero[tid]) {
		p_temp[tid] = p_nonezero[tid] + p_blocksum;
		}*/
		//p_nonezero_tmp= p_nonezero[tid];
		p_temp[tid] = p_nonezero[tid];
		__syncthreads();
		p_count += __syncthreads_count(p_nonezero[tid]);
		__syncthreads();
		prefix_sum(p_temp);
		__syncthreads();
		p_temp[tid] += p_blocksum;
		__syncthreads();
		f = p_temp[tid] - 1;
		__syncthreads();
		f += m;
		__syncthreads();
		if (p_input[tid]) {
			index[f] = (k*blockDim.x + tid) + 1;
			value[f] = p_input[tid];
		}
		__syncthreads();
		p_blocksum = p_temp[blockDim.x - 1];
		__syncthreads();
		p_temp[tid] = 0;
		p_input[tid] = 0;
		p_nonezero[tid] = 0;
		f = 0;

		//m = 0;
		__syncthreads();
	}
	if (tid == 0) count[blockIdx.x + (*d_counter_0)*gridDim.x] = p_count;
	__syncthreads();
	/*p_input[tid] = 0;
	p_nonezero[tid] = 0;
	p_temp[tid] = 0;
	p_blocksum = 0;
	p_count = 0;
	__syncthreads();*/
}

__device__ void dense_sparse_kernel2(volatile int *p, int* index, int* value, int* count, int *d_sparse_slotcount, int *d_slotoffset, int *d_counter_0, int numOfwordD)
{
	volatile __shared__ int p_input[ShaMemSize];
	volatile __shared__ int p_nonezero[ShaMemSize];
	volatile __shared__ int p_temp[ShaMemSize];
	int p_blocksum = 0;
	int p_count = 0;
	int m = 0;
	int f = 0;
	m = d_slotoffset[blockIdx.x + (*d_counter_0)*gridDim.x+ numOfwordD]- K*numOfwordD;
	//int p_nonezero_tmp = 0;
	//__shared__ int p_temp[K];
	int tid = threadIdx.x;
	int blockId = blockIdx.x;
	p_input[tid] = 0;
	p_nonezero[tid] = 0;
	p_temp[tid] = 0;
	__syncthreads();
	for (int k = 0; k < ((K - 1) / blockDim.x + 1); k++)
	{
		if ((k*blockDim.x + tid) < K) {
			p_input[tid] = p[k*blockDim.x + tid + blockId*K];
			//p_nonezero[tid] = p_input[tid] && 1;
			//p[k] = 0;
		}
		__syncthreads();
		if ((k*blockDim.x + tid) < K) {
			//p_input[tid] = p[k];
			p_nonezero[tid] = p_input[tid] && 1;
			//p[k] = 0;
		}
		__syncthreads();
		/*if (p_nonezero[tid]) {
		p_temp[tid] = p_nonezero[tid] + p_blocksum;
		}*/
		//p_nonezero_tmp= p_nonezero[tid];
		p_temp[tid] = p_nonezero[tid];
		__syncthreads();
		p_count += __syncthreads_count(p_nonezero[tid]);
		__syncthreads();
		prefix_sum(p_temp);
		__syncthreads();
		p_temp[tid] += p_blocksum;
		__syncthreads();
		f = p_temp[tid] - 1;
		__syncthreads();
		f += m;
		__syncthreads();
		if (p_input[tid]) {
			index[f] = (k*blockDim.x + tid) + 1;
			value[f] = p_input[tid];
		}
		__syncthreads();
		p_blocksum = p_temp[blockDim.x - 1];
		__syncthreads();
		p_temp[tid] = 0;
		p_input[tid] = 0;
		p_nonezero[tid] = 0;
		f = 0;

		//m = 0;
		__syncthreads();
	}
	if (tid == 0) count[blockIdx.x + (*d_counter_0)*gridDim.x] = p_count;
	__syncthreads();
	/*p_input[tid] = 0;
	p_nonezero[tid] = 0;
	p_temp[tid] = 0;
	p_blocksum = 0;
	p_count = 0;
	__syncthreads();*/
}


__device__ void warp_prefix_sum(int& p) {

	int laneId = threadIdx.x % warpSize;

	int warpId = threadIdx.x / warpSize;
	int tid = threadIdx.x %warpSize;
	int p_temp;
	p_temp = p;
	for (int iter = 1; iter < 2 * warpSize; iter *= 2)
	{
		p_temp = __shfl_up(p, iter);
		p = (laneId >(iter - 1)) ? (p + p_temp) : p;
	}
}

__device__ void warp_radix_sort(volatile int* p) {
	volatile __shared__ int p_input[ShaMemSize];
	volatile __shared__ int p_split[ShaMemSize];
	volatile __shared__ int p_t[ShaMemSize];
	volatile __shared__ int p_index[ShaMemSize];
	int blockId = blockIdx.x;
	int tid = threadIdx.x;
	int laneId = threadIdx.x % warpSize;
	int warpId = threadIdx.x / warpSize;

	p_split[tid] = 0;
	p_t[tid] = 0;
	p_index[tid] = 0;
	p_input[tid] = p[tid];
	__syncthreads();
	for (int i = 0; i < 32; i++) {
		int p_totalFalses = 0;
		int p_split_inverse = 0;
		int pTmp = 0;
		p_split[tid] = (p_input[tid] >> i) & 1;
		__syncthreads();
		p_split_inverse = (p_split[tid] + 1) & 1;
		__syncthreads();
		warp_prefix_sum(p_split_inverse);
		__syncthreads();
		pTmp = __shfl(p_split_inverse, warpSize - 1, warpSize);
		__syncthreads();
		p_totalFalses = pTmp;
		__syncthreads();
		pTmp = __shfl_up(p_split_inverse, 1);
		__syncthreads();
		p_split_inverse = pTmp;
		__syncthreads();
		if (laneId == 0) {
			p_split_inverse = 0;
		}
		__syncthreads();
		p_t[tid] = laneId - p_split_inverse + p_totalFalses;
		__syncthreads();
		p_index[tid] = (p_split[tid]>0) ? p_t[tid] : p_split_inverse;
		__syncthreads();
		pTmp = p_input[tid];
		__syncthreads();
		p_input[p_index[tid] + warpId*warpSize] = pTmp;
		__syncthreads();
	}
	p[tid] = p_input[tid];
}

//scan a token list to generate the sparse WT/DT matrix
__device__ void warp_index_value_count(volatile int* p, volatile int *index, volatile int *value) {

	__shared__ int p_input[ShaMemSize];
	int adj_diff = 0;
	int nonezero_index = 0;
	int nonezero_index_move = 0;
	int blockId = blockIdx.x;
	int tid = threadIdx.x;
	int laneId = threadIdx.x % warpSize;
	int warpId = threadIdx.x / warpSize;
	int pTmp = 0;
	p_input[tid] = p[tid];
	__syncthreads();



	if (laneId  > 0)
	{
		pTmp = p_input[tid - 1];
	}
	__syncthreads();
	if (laneId  > 0)
	{
		adj_diff = (p_input[tid] != pTmp);
	}
	__syncthreads();

	if (laneId == 0) {
		adj_diff = 0;
	}
	__syncthreads();
	warp_prefix_sum(adj_diff);
	__syncthreads();
	//if (laneId == warpSize - 1)
	//{
	//	count[warpId+ blockId *blockDim.x/ warpSize] = adj_diff + 1;
	//}
	index[adj_diff + warpId*warpSize] = p_input[tid];
	__syncthreads();
	if (laneId < warpSize - 1)
	{
		pTmp = p_input[tid + 1];
	}
	__syncthreads();
	if (laneId < warpSize - 1)
	{
		nonezero_index = (p_input[tid] != pTmp);
	}
	__syncthreads();
	if (laneId == warpSize - 1) {
		nonezero_index = 1;
	}
	__syncthreads();
	if (nonezero_index != 0) {
		nonezero_index = laneId + 1;
		//value[adj_diff + warpId*warpSize] = nonezero_index;
	}
	__syncthreads();
	if (nonezero_index != 0) {
		//nonezero_index = laneId + 1;
		value[adj_diff + warpId*warpSize] = nonezero_index;
	}
	__syncthreads();
	nonezero_index_move = __shfl_up(value[laneId + warpId*warpSize], 1);
	__syncthreads();
	if ((value[laneId + warpId*warpSize] > 0) && (laneId>0))
	{
		value[laneId + warpId*warpSize] -= nonezero_index_move;
	}
	__syncthreads();

}


//reconstruct sparse matrix from token list
__global__ void tokenlist_to_matrix_warp(int *d_a, int *d_count, int *d_index, int *d_value, int *d_slotcount, int *d_slotoffset,  int *d_row_sum, int *d_counter_1, int* d_token_amount_0, int* d_token_amount_1,  int numOfTokenD)
{
	volatile __shared__ int p_input[ShaMemSize];
	volatile __shared__ int p_index[ShaMemSize];
	volatile __shared__ int p_value[ShaMemSize];
	int tid = threadIdx.x;
	int globalId = threadIdx.x + blockIdx.x * blockDim.x;
	int laneId = threadIdx.x % warpSize;
	int warpId = threadIdx.x / warpSize;
	int blockId = blockIdx.x;
	int GridWarpDim = gridDim.x*blockDim.x / 32;
	int GridWarpId = globalId / warpSize;
	int p_indexTmp;
	int p_valueTmp;
	if ((GridWarpId > (*d_token_amount_1 - 1 - *d_counter_1*GridWarpDim))|| (d_slotcount[GridWarpId + *d_counter_1*GridWarpDim + *d_token_amount_0] == 0))
	{
		return;
	}
	__syncthreads();
	p_input[tid] = 0;
	p_index[tid] = 0;
	p_value[tid] = 0;
	p_indexTmp = 0;
	p_valueTmp = 0;
	if (laneId < d_slotcount[GridWarpId + *d_counter_1*GridWarpDim + *d_token_amount_0])
	{
		p_input[tid] = d_a[d_slotoffset[GridWarpId + *d_counter_1*GridWarpDim + *d_token_amount_0] + numOfTokenD+laneId];
		//atomicAdd(&d_row_sum[p_input[tid] - 1], 1);
		//p_index[tid] = d_index[d_slotoffset[GridWarpId + *d_token_amount_0] + laneId];
		//p_value[tid] = d_value[d_slotoffset[GridWarpId + *d_token_amount_0] + laneId];
	}
	__syncthreads();
	warp_radix_sort(p_input);
	__syncthreads();

	warp_index_value_count(p_input, p_index, p_value);
	__syncthreads();
	p_indexTmp = p_index[tid];
	p_valueTmp = p_value[tid];
	__syncthreads();
	if (d_slotcount[GridWarpId + *d_counter_1*GridWarpDim + *d_token_amount_0] != warpSize)
	{
		p_index[tid] = __shfl_down(p_indexTmp, 1);
		p_value[tid] = __shfl_down(p_valueTmp, 1);
		if (laneId == warpSize - 1)
		{
			p_index[tid] = 0;
			p_value[tid] = 0;
		}
	}
	__syncthreads();
	d_count[GridWarpId + *d_counter_1*GridWarpDim + *d_token_amount_0] = __popc(__ballot(p_value[tid]));
	__syncthreads();
	//d_tmp[globalId] = p_value[tid];
	if (laneId < d_slotcount[GridWarpId + *d_counter_1*GridWarpDim + *d_token_amount_0])
	{
		d_index[d_slotoffset[GridWarpId + *d_counter_1*GridWarpDim + *d_token_amount_0] + laneId] = p_index[tid];
		d_value[d_slotoffset[GridWarpId + *d_counter_1*GridWarpDim + *d_token_amount_0] + laneId] = p_value[tid];
	}
	__syncthreads();

}
//reconstruct sparse matrix from token list
__global__ void tokenlist_to_matrix(int *d_a, int *d_count, int *d_index, int *d_value, int *d_slotcount, int *d_slotoffset, int *d_row_sum, int *d_counter_0, int *d_token_amount_0, int *d_dense,  int numOfTokenD)
{
	volatile __shared__ int p_input[ShaMemSize];
	volatile __shared__ int p_index[ShaMemSize];
	volatile __shared__ int p_value[ShaMemSize];
	volatile __shared__ int p_index_tmp[ShaMemSize];
	volatile __shared__ int p_value_tmp[ShaMemSize];
	//volatile __shared__ int p_dense[K];
	int tid = threadIdx.x;
	int globalId = threadIdx.x + blockIdx.x * blockDim.x;
	int blockId = blockIdx.x;
	int indicator = 0;
	int GridDim = gridDim.x;
	
	/*int wordIdWT = blockId + (*d_counter_0)*GridDim ;*/
	/*long long tokenStart = d_TokenOffset[wordId];
	long long tokenEnd = d_TokenOffset[wordId] + d_TokenCount[wordId];*/



	if ((blockId > (*d_token_amount_0 - 1 - *d_counter_0*gridDim.x))|| (d_slotcount[blockId + (*d_counter_0)*GridDim]==0))
	{
		return;
	}
	int wordId = blockId + (*d_counter_0)*GridDim;
	p_input[tid] = 0;
	p_index[tid] = 0;
	p_value[tid] = 0;
	p_index_tmp[tid] = 0;
	p_value_tmp[tid] = 0;
	for (int k = tid; k < K; k += blockDim.x)
	{
		d_dense[k + K*blockId] = 0;
	}

	__syncthreads();

	for (int i = tid; i < ((d_slotcount[wordId] - 1) / blockDim.x + 1)*blockDim.x; i += blockDim.x) {
		if (i < d_slotcount[wordId]) {
			int tmpIndex = d_slotoffset[wordId] + i + numOfTokenD;
			p_input[tid] = d_a[tmpIndex];
			//atomicAdd(&d_row_sum[p_input[tid] - 1], 1);
		}

		__syncthreads();
		radix_sort(p_input);
		__syncthreads();
		index_value_count(p_input, p_index, p_value);
		__syncthreads();
		if (((d_slotcount[wordId] - indicator*blockDim.x) < blockDim.x) && (tid<(blockDim.x - 1)))
		{
			p_index_tmp[tid] = p_index[tid + 1];
			p_value_tmp[tid] = p_value[tid + 1];
		}
		__syncthreads();

		if (((d_slotcount[wordId] - indicator*blockDim.x) < blockDim.x) && (tid<(blockDim.x - 1)))
		{
			p_index[tid] = p_index_tmp[tid];
			p_value[tid] = p_value_tmp[tid];
		}
		__syncthreads();

		if (((d_slotcount[wordId] - indicator*blockDim.x) < blockDim.x) && (tid == (blockDim.x - 1)))
		{
			p_index[tid] = 0;
			p_value[tid] = 0;
		}
		__syncthreads();
		if (p_index[tid])
		{
			//atomicAdd(&p_dense[p_index[tid] - 1], 1);
			d_dense[p_index[tid] - 1 + K*blockId] += p_value[tid];
		}
		__syncthreads();
		p_index[tid] = 0;
		p_value[tid] = 0;
		p_input[tid] = 0;
		p_index_tmp[tid] = 0;
		p_index_tmp[tid] = 0;
		indicator++;
		__syncthreads();
	}
	__syncthreads();
	/*if (globalId == 0) printf("%d mark\n", *d_counter_0);
	__syncthreads();*/
	dense_sparse_kernel(d_dense, d_index, d_value, d_count, d_slotcount, d_slotoffset, d_counter_0);
	__syncthreads();

}

__global__ void DT_Update_Kernel(int *d_Index, int *d_a, int *d_count, int *d_index, int *d_value, int *d_slotcount, int *d_slotoffset, int *d_sparse_slotcount, int *d_sparse_slotoffset, int *d_counter_0, int argD, int *d_dense)
{
	volatile __shared__ int p_input[ShaMemSize];
	volatile __shared__ int p_index[ShaMemSize];
	volatile __shared__ int p_value[ShaMemSize];
	volatile __shared__ int p_index_tmp[ShaMemSize];
	volatile __shared__ int p_value_tmp[ShaMemSize];
	//volatile __shared__ int p_dense[K];
	int tid = threadIdx.x;
	int globalId = threadIdx.x + blockIdx.x * blockDim.x;
	int blockId = blockIdx.x;
	int indicator = 0;
	int GridDim = gridDim.x;
	if (blockId > (argD - 1 - *d_counter_0*gridDim.x))
	{
		return;
	}
	p_input[tid] = 0;
	p_index[tid] = 0;
	p_value[tid] = 0;
	p_index_tmp[tid] = 0;
	p_value_tmp[tid] = 0;
	for (int k = tid; k < K; k += blockDim.x)
	{
		d_dense[k + K*blockId] = 0;
	}

	__syncthreads();

	for (int i = tid; i < ((d_slotcount[blockId + (*d_counter_0)*GridDim] - 1) / blockDim.x + 1)*blockDim.x; i += blockDim.x) {
		if (i < d_slotcount[blockId + (*d_counter_0)*GridDim]) {
			p_input[tid] = d_a[d_Index[d_slotoffset[blockId + (*d_counter_0)*GridDim] + i]];
			//atomicAdd(&d_row_sum[p_input[tid] - 1], 1);
		}
		else
		{
			p_input[tid] = 0;
		}
		__syncthreads();
		radix_sort(p_input);
		__syncthreads();
		index_value_count(p_input, p_index, p_value);
		__syncthreads();
		if (((d_slotcount[blockId + (*d_counter_0)*GridDim] - indicator*blockDim.x) < blockDim.x) && (tid<(blockDim.x - 1)))
		{
			p_index_tmp[tid] = p_index[tid + 1];
			p_value_tmp[tid] = p_value[tid + 1];
		}
		__syncthreads();
		if (((d_slotcount[blockId + (*d_counter_0)*GridDim] - indicator*blockDim.x) < blockDim.x) && (tid<(blockDim.x - 1)))
		{
			p_index[tid] = p_index_tmp[tid];
			p_value[tid] = p_value_tmp[tid];
		}
		__syncthreads();

		if (((d_slotcount[blockId + (*d_counter_0)*GridDim] - indicator*blockDim.x) < blockDim.x) && (tid == (blockDim.x - 1)))
		{
			p_index[tid] = 0;
			p_value[tid] = 0;
		}
		__syncthreads();
		if (p_index[tid])
		{
			//atomicAdd(&p_dense[p_index[tid] - 1], 1);
			d_dense[p_index[tid] - 1 + K*blockId] += p_value[tid];
		}
		__syncthreads();
		p_index[tid] = 0;
		p_value[tid] = 0;
		p_input[tid] = 0;
		p_index_tmp[tid] = 0;
		p_index_tmp[tid] = 0;
		indicator++;
		__syncthreads();
	}
	__syncthreads();

	dense_sparse_kernel(d_dense, d_index, d_value, d_count, d_sparse_slotcount, d_sparse_slotoffset, d_counter_0);
	__syncthreads();

}





__global__ void WTDen_Update_Kernel(int *deviceTopic, int *deviceWTDense, int *deviceTLCount, int *deviceTLOffset, int *deviceWTOffset, int numOfWordD, int counter)
{

	int input;
	
	int tid = threadIdx.x;
	int globalId = threadIdx.x + blockIdx.x * blockDim.x;
	int blockId = blockIdx.x;
	int wordId=blockId+counter*gridDim.x;
	int tmpTopic;
	int tmpWTOffset= deviceWTOffset[wordId];
	int tmpTLOffset= deviceTLOffset[wordId];

	if (wordId > numOfWordD - 1)
	{
		return;
	}

	for (int k = tid; k < deviceTLCount[wordId]; k += blockDim.x)
	{
		tmpTopic = deviceTopic[tmpTLOffset + k];
		atomicAdd(&deviceWTDense[tmpWTOffset + tmpTopic - 1], 1);
	}
	__syncthreads();

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





__global__ void sparseMatrixAdd(int* argCount0, int* argOffset0, int* argNZCount0, int* argIndex0, int* argValue0, int* argCount1, int* argOffset1, int* argNZCount1, int* argIndex1, int* argValue1, int* argDense, int argNumRows, int* argBlockCounter, int* argWTRowSum, int numOfWordD)
{

	int tid = threadIdx.x;
	int globalId = threadIdx.x + blockIdx.x * blockDim.x;
	int blockId = blockIdx.x;
	int row = *argBlockCounter*gridDim.x + blockId;

	if (blockId > (argNumRows - 1 - *argBlockCounter*gridDim.x))
	{
		return;
	}


	for (int k = tid; k < K; k += blockDim.x)
	{
		argDense[k + K*blockId] = 0;
	}

	for (int k = tid; k < argNZCount0[row]; k += blockDim.x)
	{
		argDense[K*blockId + argIndex0[argOffset0[row+ numOfWordD]-K*numOfWordD + k] - 1] += argValue0[argOffset0[row+ numOfWordD] - K*numOfWordD + k];
	}
	__syncthreads();

	for (int k = tid; k < argNZCount1[row]; k += blockDim.x)
	{
		atomicAdd(&argWTRowSum[argIndex1[argOffset1[row] + k] - 1], argValue1[argOffset1[row] + k]);
		argDense[K*blockId + argIndex1[argOffset1[row] + k] - 1] += argValue1[argOffset1[row] + k];
	}

	__syncthreads();

	dense_sparse_kernel2(argDense, argIndex0, argValue0, argNZCount0, argCount0, argOffset0, argBlockCounter,numOfWordD);

	__syncthreads();

}




__global__ void initRandState(curandState *state)
{
	int tid = blockIdx.x*blockDim.x + threadIdx.x;
	curand_init(clock() + tid, tid, 0, &state[tid]);
}



__global__ void LDAKernelTrain(double alpha, double beta, int* d_Index, int* d_TopicIndex, int* d_SparseDTCount, int* d_SparseDTIndex, int* d_SparseDTValue, int* d_TokenCountDT, int* d_TokenOffsetDT, int* d_DocListCount, int* d_DocListOffset, int* d_SparseWTCount, int* d_SparseWTIndex, int* d_SparseWTValue, int* d_TokenCount, int* d_TokenOffset, int* d_WordListCount, int* d_WordListOffset, int* d_WTRowSum, int* d_blockCounter, int*d_DocIndex, int D, int W, double* d_Perplexity, curandState *randState, double *WTHeadDense, int numOfWordD, int numOfWordS)

{

	int tid = threadIdx.x;

	int laneId = threadIdx.x % 32;
	int localId = threadIdx.x / 32;
	int wordId = *d_blockCounter*gridDim.x + blockIdx.x + numOfWordD;

	int blockId = blockIdx.x;

	volatile __shared__ double STree[ShaMemSize / 32][32];
	volatile __shared__ double pTemp[ShaMemSize];
	volatile __shared__ double prefixSumSample[ShaMemSize / 32][32];
	volatile __shared__ double pTmp[ShaMemSize / 32][32];

	volatile __shared__ double QTreeL1[K / 32];
	volatile __shared__ double QTreeL2[32];

	if (blockId > (numOfWordS - 1 - *d_blockCounter*gridDim.x))
	{
		return;
	}
	if (localId == 0) {
		QTreeL2[laneId] = 0;
	}

	double p_temp1 = 0.0;
	//double pTemp = 0.0;
	pTmp[localId][laneId] = 0.0;
	prefixSumSample[localId][laneId] = 0.0;

	__syncthreads();
	long long tokenStart = d_TokenOffset[wordId];
	long long tokenEnd = d_TokenOffset[wordId] + d_TokenCount[wordId];
	long long WTStart = d_WordListOffset[wordId] - d_WordListOffset[numOfWordD];
	long long WTEnd = d_WordListOffset[wordId] - d_WordListOffset[numOfWordD] + d_SparseWTCount[wordId- numOfWordD];
	//double WTHeadDenom;
	__syncthreads();

	pTemp[tid] = 0.0;
	//__syncthreads();
	// Reconstruct dense WT vector from sparse WT matrix
	//for (int i = tid; i < K; i += blockDim.x)
	//{
	//	WTHead[i] = beta / (d_WTRowSum[i] + W*beta);
	//	//__syncthreads();
	//}
	//__syncthreads();

	for (int i = tid; i < K; i += blockDim.x)
	{
		WTHeadDense[i + K*blockId] = beta / (d_WTRowSum[i] + W*beta);

	}
	__syncthreads();

	for (int i = tid + WTStart; i < WTEnd; i += blockDim.x)
	{
		if ((d_SparseWTIndex[i] < 1) || (d_SparseWTIndex[i] > K)) printf("wrong WTIndex:%d", d_SparseWTIndex[i]);
		WTHeadDense[d_SparseWTIndex[i] - 1 + K*blockId] = (d_SparseWTValue[i] + beta) / (d_WTRowSum[d_SparseWTIndex[i] - 1] + W*beta);
	}
	__syncthreads();

	// Construct Q tree from WTHead
	for (int i = localId; i < K / 32; i += blockDim.x / 32) {
		int   tmpK = i * 32 + laneId;
		//__syncthreads();
		double tmpVal = 0;
		tmpVal = WTHeadDense[tmpK + K*blockId];
		tmpVal += __shfl_down(tmpVal, 16);
		tmpVal += __shfl_down(tmpVal, 8);
		tmpVal += __shfl_down(tmpVal, 4);
		tmpVal += __shfl_down(tmpVal, 2);
		tmpVal += __shfl_down(tmpVal, 1);
		tmpVal = __shfl(tmpVal, 0);
		QTreeL1[i] = tmpVal;

	}
	__syncthreads();


	for (int i = localId; i < K / 32 / 32; i += blockDim.x / 32) {
		int   tmpK = i * 32 + laneId;
		//__syncthreads();
		double tmpVal = 0;
		tmpVal = QTreeL1[tmpK];
		tmpVal += __shfl_down(tmpVal, 16);
		tmpVal += __shfl_down(tmpVal, 8);
		tmpVal += __shfl_down(tmpVal, 4);
		tmpVal += __shfl_down(tmpVal, 2);
		tmpVal += __shfl_down(tmpVal, 1);
		tmpVal = __shfl(tmpVal, 0);
		QTreeL2[i] = tmpVal;
	}
	__syncthreads();



	if (localId == 0) {

		double value = alpha*QTreeL2[laneId];
		value += __shfl_up(value, 1, 32)*(laneId >= 1);
		value += __shfl_up(value, 2, 32)*(laneId >= 2);
		value += __shfl_up(value, 4, 32)*(laneId >= 4);
		value += __shfl_up(value, 8, 32)*(laneId >= 8);
		value += __shfl_up(value, 16, 32)*(laneId >= 16);
		QTreeL2[laneId] = value;
	}
	__syncthreads();

	double Q = QTreeL2[31];
	//__syncthreads();
	double sumPerplexity = 0.0;


	for (int tokenIdx = tokenStart + localId; tokenIdx < tokenEnd; tokenIdx += blockDim.x / 32) //iterate over tokens
	{
		//int docId = __ldg(&d_Index[d_TopicIndex[tokenIdx]]);
		int docId = d_DocIndex[tokenIdx];
		if ((docId < 1) || (docId > D)) printf("wrong docId:%d", docId);
		//computing S.
		double S = 0;
		double STmp = 0;
		double uTmp = 0;
		long long DTStart = d_DocListOffset[docId - 1];
		long long DTEnd = d_DocListOffset[docId - 1] + ((d_SparseDTCount[docId - 1] - 1) / 32 + 1) * 32;


		STree[localId][laneId] = 0;
		__syncthreads();
		for (int tmpIdx = DTStart + laneId, SIdx = 0; tmpIdx < DTEnd; tmpIdx += 32) {


			int   colVal = d_SparseDTValue[tmpIdx];
			int   colK = d_SparseDTIndex[tmpIdx];

			//if ((colK < 1) || (colK> K)) printf("wrong docIndex:%d", colK);

			double tmpP1k = 0.0;
			double ptmpP1k = 0.0;
			if (colK != 0) {
				tmpP1k = colVal*WTHeadDense[colK - 1 + K*blockId];
			}
			tmpP1k += __shfl_down(tmpP1k, 16);
			tmpP1k += __shfl_down(tmpP1k, 8);
			tmpP1k += __shfl_down(tmpP1k, 4);
			tmpP1k += __shfl_down(tmpP1k, 2);
			tmpP1k += __shfl_down(tmpP1k, 1);
			tmpP1k = __shfl(tmpP1k, 0);



			S += tmpP1k;
			//__syncthreads();
			STree[localId][SIdx] = S;
			//__syncthreads();
			SIdx++;
			//__syncthreads();
		}
		__syncthreads();
		/*STmp = S;

		S = __shfl(STmp, 0);*/
		S = __shfl(S, 0);
		//__syncthreads();
		//randomly generate u.
		double u;
		if (laneId == 0)u = curand_uniform(&(randState[threadIdx.x + blockDim.x*blockIdx.x])) / 1.00001;

		//if ((u == 1.0))printf("what's this");
		//if (laneId == 0)u = d_randu[tokenIdx];
		uTmp = u;
		//__syncthreads();
		u = __shfl(uTmp, 0);
		int newZ = 0;
		//__syncthreads();
		double tmpU = 0;
		double tmpU1 = 0;
		//__syncthreads();
		__syncthreads();


		if (u < S / (S + Q))
		{

			//totalS ++;
			//tmpClock = clock64();

			double transU = u*(S + Q);

			double tmpSumHigh, tmpSumLow = 0.0;
			tmpSumHigh = STree[localId][laneId];
			tmpSumLow = __shfl_up(tmpSumHigh, 1, 32);
			if (laneId == 0)tmpSumLow = 0;
			int voteFlag = 0;
			if ((transU < tmpSumHigh)) voteFlag = 1;
			int lvl1Idx = __ffs(__ballot(voteFlag)) - 1;
			if (lvl1Idx < 0) lvl1Idx = (DTEnd - DTStart) / 32 - 1;

			tmpU1 = transU;

			transU = tmpU1 - tmpSumLow;

			tmpU = transU;

			transU = __shfl(tmpU, lvl1Idx);


			int tmpIdx = DTStart + lvl1Idx * 32 + laneId;

			int tmpNewZ = d_SparseDTIndex[tmpIdx];
			int colVal = d_SparseDTValue[tmpIdx];

			double p1k = 0.0;
			if (tmpNewZ != 0)
			{
				p1k = colVal*WTHeadDense[tmpNewZ - 1 + K*blockId];
			}

			//__syncthreads();
			prefixSumSample[localId][laneId] = p1k;
			//__syncthreads();

			double value = prefixSumSample[localId][laneId];
			value += __shfl_up(value, 1, 32)*(laneId >= 1);
			value += __shfl_up(value, 2, 32)*(laneId >= 2);
			value += __shfl_up(value, 4, 32)*(laneId >= 4);
			value += __shfl_up(value, 8, 32)*(laneId >= 8);
			value += __shfl_up(value, 16, 32)*(laneId >= 16);

			prefixSumSample[localId][laneId] = value;

			double tmpSum = prefixSumSample[localId][laneId];

			voteFlag = 0;
			if (transU < tmpSum) voteFlag = 1;

			int offset = __ffs(__ballot(voteFlag)) - 1;
			//if (offset < 0) printf("bugs!");
			newZ = __shfl(tmpNewZ, offset);
			if ((newZ > K) || (newZ < 1)) {
				//printf("part1: u=%f, %d,%d,%d\n", u, newZ, lvl1Idx, offset);
				newZ = 5;
			}

		}


		//if (u > S / (S + Q))
		else //bucket Q
		{

			double transU = (u - S / (S + Q))*(S + Q);
			/*double tmpU;
			double tmpU1;*/
			//totalQ ++;
			//double originalU = transU;

			//level 1: decide position
			double tmpSumHigh, tmpSumLow = 0.0;
			tmpSumHigh = QTreeL2[laneId];
			//__syncthreads();
			tmpSumLow = __shfl_up(tmpSumHigh, 1, 32);
			//__syncthreads();
			if (laneId == 0)tmpSumLow = 0;

			//voting for lvl1Idx
			int voteFlag = 0;
			if (transU < tmpSumHigh) voteFlag = 1; //voteFlag = transU < tmpSumHigh;	
			int lvl1Idx = __ffs(__ballot(voteFlag)) - 1;
			//if (lvl1Idx == 10) printf("bugs! position1");
			if (lvl1Idx < 0) lvl1Idx = K / 1024 - 1;
			//if (lvl1Idx == 10) printf("lvl1Idx bugs! position2");
			//if (lvl1Idx == 31) printf("bugs!");
			tmpU1 = transU;
			transU = tmpU1 - tmpSumLow;
			tmpU = transU;
			transU = __shfl(tmpU, lvl1Idx);
			prefixSumSample[localId][laneId] = alpha*QTreeL1[32 * lvl1Idx + laneId];

			//accumulation

			double value = prefixSumSample[localId][laneId];
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
			if (transU < tmpSumHigh)voteFlag = 1; //voteFlag = transU < tmpSumHigh;		
			int lvl2Idx = __ffs(__ballot(voteFlag)) - 1;
			if (lvl2Idx < 0)lvl2Idx = 31;

			tmpU1 = transU;
			transU = tmpU1 - tmpSumLow;
			tmpU = transU;
			transU = __shfl(tmpU, lvl2Idx);
			/*	transU = transU - tmpSumLow;
			transU = __shfl(transU, lvl2Idx)*/;

			prefixSumSample[localId][laneId] = alpha*WTHeadDense[1024 * lvl1Idx + 32 * lvl2Idx + laneId + K*blockId];

			value = prefixSumSample[localId][laneId];

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
			if (transU < tmpSumHigh)voteFlag = 1; //voteFlag = transU < tmpSumHigh;		
			int lvl3Idx = __ffs(__ballot(voteFlag)) - 1;
			if (lvl3Idx < 0)lvl3Idx = 31;
			newZ = lvl1Idx * 1024 + 32 * lvl2Idx + lvl3Idx + 1;
			if ((newZ > K) || (newZ < 1)) {
				printf("part2: u=%f, %d,%d,%d,%d\n", u, newZ, lvl1Idx, lvl2Idx, lvl3Idx);
				newZ = 5;
			}
			//__syncthreads();
			//if(tmpFlag == 1)return;
		}


		if (laneId == 0) {
			d_TopicIndex[tokenIdx] = newZ;
			/*if (newZ > K) {
			printf("u=%f, %d,%d,%d,%d", u, newZ, lvl1Idx, lvl2Idx, lvl3Idx);
			}*/
			//p_temp = S + Q;
			//d_S[tokenIdx] = Q;
			d_Perplexity[tokenIdx] = log((S + Q) / (d_TokenCountDT[docId - 1] + K*alpha));
			//sumPerplexity += log((S + Q) / (d_TokenCountDT[docId - 1] + K*alpha));

		}
	}
	__syncthreads();
	//if (threadIdx.x % 32 == 0)
	//	d_Perplexity[(threadIdx.x + blockDim.x*blockIdx.x) / 32] = sumPerplexity;
	////wordPerplexity[(threadIdx.x + blockDim.x*blockIdx.x) / 32] = sumPerplexity;
	//__syncthreads();

}

__global__ void LDAKernelTrainD(double alpha, double beta, int* d_Index, int* d_TopicIndex, int* d_SparseDTCount, int* d_SparseDTIndex, int* d_SparseDTValue, int* d_TokenCountDT, int* d_TokenOffsetDT, int* d_DocListCount, int* d_DocListOffset, int* d_WTDense, int* d_WTDenseCopy, int* d_TokenCount, int* d_TokenOffset, int* d_WordListCount, int* d_WordListOffset, int* d_WTRowSum, int* d_blockCounter, int*d_DocIndex, int D, int W, double* d_Perplexity, curandState *randState, double *WTHeadDense, int numOfWordD)

{

	int tid = threadIdx.x;

	int laneId = threadIdx.x % 32;
	int localId = threadIdx.x / 32;
	int wordId = *d_blockCounter*gridDim.x + blockIdx.x ;

	int blockId = blockIdx.x;

	volatile __shared__ double STree[ShaMemSize / 32][32];
	volatile __shared__ double pTemp[ShaMemSize];
	volatile __shared__ double prefixSumSample[ShaMemSize / 32][32];
	volatile __shared__ double pTmp[ShaMemSize / 32][32];

	volatile __shared__ double QTreeL1[K / 32];
	volatile __shared__ double QTreeL2[32];

	if (blockId > (numOfWordD - 1 - *d_blockCounter*gridDim.x))
	{
		return;
	}
	if (localId == 0) {
		QTreeL2[laneId] = 0;
	}

	double p_temp1 = 0.0;
	//double pTemp = 0.0;
	pTmp[localId][laneId] = 0.0;
	prefixSumSample[localId][laneId] = 0.0;

	__syncthreads();
	long long tokenStart = d_TokenOffset[wordId];
	long long tokenEnd = d_TokenOffset[wordId] + d_TokenCount[wordId];
	long long WTStart = d_WordListOffset[wordId];
	/*long long WTEnd = d_WordListOffset[wordId] + d_SparseWTCount[wordId];*/
	/*double WTHeadDenom;*/
	__syncthreads();

	pTemp[tid] = 0.0;
	//__syncthreads();
	// Reconstruct dense WT vector from sparse WT matrix
	//for (int i = tid; i < K; i += blockDim.x)
	//{
	//	WTHead[i] = beta / (d_WTRowSum[i] + W*beta);
	//	//__syncthreads();
	//}
	//__syncthreads();

	for (int i = tid; i < K; i += blockDim.x)
	{
		WTHeadDense[i + K*blockId] = (d_WTDense[WTStart+i]+beta) / (d_WTRowSum[i] + W*beta);
	}
	__syncthreads();

	//for (int i = tid + WTStart; i < WTEnd; i += blockDim.x)
	//{
	//	if ((d_SparseWTIndex[i] < 1) || (d_SparseWTIndex[i] > K)) printf("wrong WTIndex:%d", d_SparseWTIndex[i]);
	//	WTHeadDense[d_SparseWTIndex[i] - 1 + K*blockId] = (d_SparseWTValue[i] + beta) / (d_WTRowSum[d_SparseWTIndex[i] - 1] + W*beta);
	//}
	//__syncthreads();

	// Construct Q tree from WTHead
	for (int i = localId; i < K / 32; i += blockDim.x / 32) {
		int   tmpK = i * 32 + laneId;
		//__syncthreads();
		double tmpVal = 0;
		tmpVal = WTHeadDense[tmpK + K*blockId];
		tmpVal += __shfl_down(tmpVal, 16);
		tmpVal += __shfl_down(tmpVal, 8);
		tmpVal += __shfl_down(tmpVal, 4);
		tmpVal += __shfl_down(tmpVal, 2);
		tmpVal += __shfl_down(tmpVal, 1);
		tmpVal = __shfl(tmpVal, 0);
		QTreeL1[i] = tmpVal;

	}
	__syncthreads();


	for (int i = localId; i < K / 32 / 32; i += blockDim.x / 32) {
		int   tmpK = i * 32 + laneId;
		//__syncthreads();
		double tmpVal = 0;
		tmpVal = QTreeL1[tmpK];
		tmpVal += __shfl_down(tmpVal, 16);
		tmpVal += __shfl_down(tmpVal, 8);
		tmpVal += __shfl_down(tmpVal, 4);
		tmpVal += __shfl_down(tmpVal, 2);
		tmpVal += __shfl_down(tmpVal, 1);
		tmpVal = __shfl(tmpVal, 0);
		QTreeL2[i] = tmpVal;
	}
	__syncthreads();



	if (localId == 0) {

		double value = alpha*QTreeL2[laneId];
		value += __shfl_up(value, 1, 32)*(laneId >= 1);
		value += __shfl_up(value, 2, 32)*(laneId >= 2);
		value += __shfl_up(value, 4, 32)*(laneId >= 4);
		value += __shfl_up(value, 8, 32)*(laneId >= 8);
		value += __shfl_up(value, 16, 32)*(laneId >= 16);
		QTreeL2[laneId] = value;
	}
	__syncthreads();

	double Q = QTreeL2[31];
	//__syncthreads();
	double sumPerplexity = 0.0;


	for (int tokenIdx = tokenStart + localId; tokenIdx < tokenEnd; tokenIdx += blockDim.x / 32) //iterate over tokens
	{
		//int docId = __ldg(&d_Index[d_TopicIndex[tokenIdx]]);
		int docId = d_DocIndex[tokenIdx];
		if ((docId < 1) || (docId > D)) printf("wrong docId:%d", docId);
		//computing S.
		double S = 0;
		double STmp = 0;
		double uTmp = 0;
		long long DTStart = d_DocListOffset[docId - 1];
		long long DTEnd = d_DocListOffset[docId - 1] + ((d_SparseDTCount[docId - 1] - 1) / 32 + 1) * 32;


		STree[localId][laneId] = 0;
		__syncthreads();
		for (int tmpIdx = DTStart + laneId, SIdx = 0; tmpIdx < DTEnd; tmpIdx += 32) {


			int   colVal = d_SparseDTValue[tmpIdx];
			int   colK = d_SparseDTIndex[tmpIdx];

			//if ((colK < 1) || (colK> K)) printf("wrong docIndex:%d", colK);

			double tmpP1k = 0.0;
			double ptmpP1k = 0.0;
			if (colK != 0) {
				tmpP1k = colVal*WTHeadDense[colK - 1 + K*blockId];
			}
			tmpP1k += __shfl_down(tmpP1k, 16);
			tmpP1k += __shfl_down(tmpP1k, 8);
			tmpP1k += __shfl_down(tmpP1k, 4);
			tmpP1k += __shfl_down(tmpP1k, 2);
			tmpP1k += __shfl_down(tmpP1k, 1);
			tmpP1k = __shfl(tmpP1k, 0);



			S += tmpP1k;
			//__syncthreads();
			STree[localId][SIdx] = S;
			//__syncthreads();
			SIdx++;
			//__syncthreads();
		}
		__syncthreads();
		/*STmp = S;

		S = __shfl(STmp, 0);*/
		S = __shfl(S, 0);
		//__syncthreads();
		//randomly generate u.
		double u;
		if (laneId == 0)u = curand_uniform(&(randState[threadIdx.x + blockDim.x*blockIdx.x])) / 1.00001;

		//if ((u == 1.0))printf("what's this");
		//if (laneId == 0)u = d_randu[tokenIdx];
		uTmp = u;
		//__syncthreads();
		u = __shfl(uTmp, 0);
		int newZ = 0;
		//__syncthreads();
		double tmpU = 0;
		double tmpU1 = 0;
		//__syncthreads();
		__syncthreads();


		if (u < S / (S + Q))
		{

			//totalS ++;
			//tmpClock = clock64();

			double transU = u*(S + Q);

			double tmpSumHigh, tmpSumLow = 0.0;
			tmpSumHigh = STree[localId][laneId];
			tmpSumLow = __shfl_up(tmpSumHigh, 1, 32);
			if (laneId == 0)tmpSumLow = 0;
			int voteFlag = 0;
			if ((transU < tmpSumHigh)) voteFlag = 1;
			int lvl1Idx = __ffs(__ballot(voteFlag)) - 1;
			if (lvl1Idx < 0) lvl1Idx = (DTEnd - DTStart) / 32 - 1;

			tmpU1 = transU;

			transU = tmpU1 - tmpSumLow;

			tmpU = transU;

			transU = __shfl(tmpU, lvl1Idx);


			int tmpIdx = DTStart + lvl1Idx * 32 + laneId;

			int tmpNewZ = d_SparseDTIndex[tmpIdx];
			int colVal = d_SparseDTValue[tmpIdx];

			double p1k = 0.0;
			if (tmpNewZ != 0)
			{
				p1k = colVal*WTHeadDense[tmpNewZ - 1 + K*blockId];
			}

			//__syncthreads();
			prefixSumSample[localId][laneId] = p1k;
			//__syncthreads();

			double value = prefixSumSample[localId][laneId];
			value += __shfl_up(value, 1, 32)*(laneId >= 1);
			value += __shfl_up(value, 2, 32)*(laneId >= 2);
			value += __shfl_up(value, 4, 32)*(laneId >= 4);
			value += __shfl_up(value, 8, 32)*(laneId >= 8);
			value += __shfl_up(value, 16, 32)*(laneId >= 16);

			prefixSumSample[localId][laneId] = value;

			double tmpSum = prefixSumSample[localId][laneId];

			voteFlag = 0;
			if (transU < tmpSum) voteFlag = 1;

			int offset = __ffs(__ballot(voteFlag)) - 1;
			//if (offset < 0) printf("bugs!");
			newZ = __shfl(tmpNewZ, offset);
			if ((newZ > K) || (newZ < 1)) {
				//printf("part1: u=%f, %d,%d,%d\n", u, newZ, lvl1Idx, offset);
				newZ = 5;
			}

		}


		//if (u > S / (S + Q))
		else //bucket Q
		{

			double transU = (u - S / (S + Q))*(S + Q);
			/*double tmpU;
			double tmpU1;*/
			//totalQ ++;
			//double originalU = transU;

			//level 1: decide position
			double tmpSumHigh, tmpSumLow = 0.0;
			tmpSumHigh = QTreeL2[laneId];
			//__syncthreads();
			tmpSumLow = __shfl_up(tmpSumHigh, 1, 32);
			//__syncthreads();
			if (laneId == 0)tmpSumLow = 0;

			//voting for lvl1Idx
			int voteFlag = 0;
			if (transU < tmpSumHigh) voteFlag = 1; //voteFlag = transU < tmpSumHigh;	
			int lvl1Idx = __ffs(__ballot(voteFlag)) - 1;
			//if (lvl1Idx == 10) printf("bugs! position1");
			if (lvl1Idx < 0) lvl1Idx = K / 1024 - 1;
			//if (lvl1Idx == 10) printf("lvl1Idx bugs! position2");
			//if (lvl1Idx == 31) printf("bugs!");
			tmpU1 = transU;
			transU = tmpU1 - tmpSumLow;
			tmpU = transU;
			transU = __shfl(tmpU, lvl1Idx);
			prefixSumSample[localId][laneId] = alpha*QTreeL1[32 * lvl1Idx + laneId];

			//accumulation

			double value = prefixSumSample[localId][laneId];
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
			if (transU < tmpSumHigh)voteFlag = 1; //voteFlag = transU < tmpSumHigh;		
			int lvl2Idx = __ffs(__ballot(voteFlag)) - 1;
			if (lvl2Idx < 0)lvl2Idx = 31;

			tmpU1 = transU;
			transU = tmpU1 - tmpSumLow;
			tmpU = transU;
			transU = __shfl(tmpU, lvl2Idx);
			/*	transU = transU - tmpSumLow;
			transU = __shfl(transU, lvl2Idx)*/;

			prefixSumSample[localId][laneId] = alpha*WTHeadDense[1024 * lvl1Idx + 32 * lvl2Idx + laneId + K*blockId];

			value = prefixSumSample[localId][laneId];

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
			if (transU < tmpSumHigh)voteFlag = 1; //voteFlag = transU < tmpSumHigh;		
			int lvl3Idx = __ffs(__ballot(voteFlag)) - 1;
			if (lvl3Idx < 0)lvl3Idx = 31;
			newZ = lvl1Idx * 1024 + 32 * lvl2Idx + lvl3Idx + 1;
			if ((newZ > K) || (newZ < 1)) {
				printf("part2: u=%f, %d,%d,%d,%d\n", u, newZ, lvl1Idx, lvl2Idx, lvl3Idx);
				newZ = 5;
			}
			//__syncthreads();
			//if(tmpFlag == 1)return;
		}


		if (laneId == 0) {
			d_TopicIndex[tokenIdx] = newZ;
			atomicAdd(&d_WTDenseCopy[WTStart + newZ - 1], 1);

			d_Perplexity[tokenIdx] = log((S + Q) / (d_TokenCountDT[docId - 1] + K*alpha));
			//sumPerplexity += log((S + Q) / (d_TokenCountDT[docId - 1] + K*alpha));

		}
	}
	__syncthreads();
	//if (threadIdx.x % 32 == 0)
	//	d_Perplexity[(threadIdx.x + blockDim.x*blockIdx.x) / 32] = sumPerplexity;
	////wordPerplexity[(threadIdx.x + blockDim.x*blockIdx.x) / 32] = sumPerplexity;
	//__syncthreads();

}






__global__ void LDATrainPerplexityReduce1(double *perplexity, double *perplexityMid, int numVals) {


	int numWarps = gridDim.x*blockDim.x / 32;
	int tid = threadIdx.x + blockIdx.x*blockDim.x;
	int warpId = tid / 32;
	int laneId = tid % 32;

	int perWarpSize = ((numVals + numWarps - 1) / numWarps + 31) / 32 * 32;
	int perWarpSizeMax = (numVals + numWarps - 1) / numWarps;
	int startIdx = perWarpSizeMax*warpId;
	int endIdx = perWarpSizeMax*warpId + perWarpSize;
	int endMax = perWarpSizeMax*warpId + perWarpSizeMax;

	double totalProd = 0.0;
	for (long long i = startIdx + laneId; i < endIdx; i += 32) {

		double tmpProd = 0.0;
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









