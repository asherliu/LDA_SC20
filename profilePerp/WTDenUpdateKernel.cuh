#ifndef _WTDENUPDATEKERNEL_H_
#define _WTDENUPDATEKERNEL_H_

#include "WTDense.cuh"
#include "WT.cuh"
#include "DataChunk.cuh"
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

void UpdateWTDenKernel(WTD &argWTDen, WTAll &argWT, Document &argDoc, int argChunkId);
void UpdateWTDenRowSumKernel(WTD &argWTDen, WTAll &argWT);




#endif