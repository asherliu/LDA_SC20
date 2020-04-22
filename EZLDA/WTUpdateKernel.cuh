#ifndef _WTUPDATEKERNEL_H_
#define _WTUPDATEKERNEL_H_

#include "WT.cuh"
#include "DataChunk.cuh"
#include "utility.cuh"
#include "Doc.cuh"
#include "Argument.cuh"
//
static void HandleError(cudaError_t err,
	const char *file,
	int line) {
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n", \
			cudaGetErrorString(err),
			file, line);
		exit(EXIT_FAILURE);
	}
}
#define H_ERR( err ) \
  (HandleError( err, __FILE__, __LINE__ ))

void UpdateWTKernel(WTAll &argWT, Document &argDoc, int argChunkId, int argStreamId, cudaStream_t& stream);





#endif