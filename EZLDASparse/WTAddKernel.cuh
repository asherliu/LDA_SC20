#ifndef _WTADDKERNEL_H_
#define _WTADDKERNEL_H_

#include "WT.cuh"
//#include "DataChunk.cuh"
#include "utility.cuh"
#include "Doc.cuh"
#include "Argument.cuh"
#include "WTUpdateKernel.cuh"

//static void HandleError(cudaError_t err,
//	const char *file,
//	int line) {
//	if (err != cudaSuccess) {
//		printf("%s in %s at line %d\n", \
//			cudaGetErrorString(err),
//			file, line);
//		exit(EXIT_FAILURE);
//	}
//}
//#define H_ERR( err ) \
//  (HandleError( err, __FILE__, __LINE__ ))

//void WTAdditionKernel(WTAll &argWT, Document &argDoc, int argStreamId, cudaStream_t& stream);


#endif
