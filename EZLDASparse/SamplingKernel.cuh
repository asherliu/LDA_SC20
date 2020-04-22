#ifndef _SAMPLINGKERNEL_H_
#define _SAMPLINGKERNEL_H_
#include "Argument.cuh"
#include "utility.cuh"
#include "Doc.cuh"
#include "WT.cuh"
#include "DT.cuh"
#include "WTDense.cuh"
#include "WTUpdateKernel.cuh"
void SampleKernelD(WTAll &argWT, DTChunk &argDT, Document &argDoc, curandState* randState, int argStreamId, cudaStream_t& stream);
//void SampleKernel(WTAll &argWT, DTChunk &argDT, Document &argDoc, curandState* randState, int argStreamId, cudaStream_t& stream);
void MaxTopicKernel(WTAll &argWT, Document &argDoc, int argStreamId, cudaStream_t& stream);
void PerplexityKernel(Document &argDoc, int argStreamId, cudaStream_t& stream);

void UpdateProbKernelD(WTAll &argWT, DTChunk &argDT, Document &argDoc, curandState* randState, int argStreamId, cudaStream_t& stream);
#endif
