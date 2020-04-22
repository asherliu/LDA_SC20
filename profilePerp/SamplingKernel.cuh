#ifndef _SAMPLINGKERNEL_H_
#define _SAMPLINGKERNEL_H_
#include "Argument.cuh"
#include "utility.cuh"
#include "Doc.cuh"
#include "WT.cuh"
#include "DT.cuh"
#include "WTDense.cuh"
#include "WTUpdateKernel.cuh"
void SampleKernelD(WTD &argWTDen, WTAll &argWT, DTChunk &argDT, Document &argDoc, curandState* randState);
void SampleKernel(WTAll &argWT, DTChunk &argDT, Document &argDoc, curandState* randState);
void PerplexityKernel(Document &argDoc);

#endif