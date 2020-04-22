#ifndef _SAMPLINGKERNEL_H_
#define _SAMPLINGKERNEL_H_
#include "Argument.cuh"
#include "utility.cuh"
#include "Doc.cuh"
#include "WT.cuh"
#include "DT.cuh"
#include "WTDense.cuh"
#include "WTUpdateKernel.cuh"
void SampleKernelD(WTD &argWTDen, WTAll &argWT, DTChunk &argDT, Document &argDoc, curandState* randState, int argChunkId, int argGPUId, cudaStream_t &stream);
void SampleKernel(WTAll &argWT, DTChunk &argDT, Document &argDoc, curandState* randState, int argChunkId, int argGPUId, cudaStream_t &stream);

#endif