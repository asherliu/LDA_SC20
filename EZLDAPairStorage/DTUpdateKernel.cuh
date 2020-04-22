#ifndef _DTUPDATEKERNEL_H_
#define _DTUPDATEKERNEL_H_

#include "DT.cuh"
#include "DataChunk.cuh"
#include "utility.cuh"
#include "Doc.cuh"
#include "Argument.cuh"
#include "WTUpdateKernel.cuh"



void UpdateDTKernel(DTChunk &argDT, Document &argDoc, int argStreamId, cudaStream_t& stream);






#endif
