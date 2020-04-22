#go!

CXX = nvcc -Xptxas="-v" -w --std=c++11 -lineinfo
COMPILED_OBJECTS = mainKernel.o DTUpdateKernel.o WTUpdateKernel.o WTDenUpdateKernel.o WTAddKernel.o SamplingKernel.o DT.o Doc.o WT.o WTDense.o WTChunk.o DataChunk.o utility.o 

dslda : $(COMPILED_OBJECTS)
	$(CXX) -o pslda $(COMPILED_OBJECTS)

mainKernel.o : mainKernel.cu DTUpdateKernel.cuh DT.cuh DataChunk.cuh utility.cuh Argument.cuh Doc.cuh WTUpdateKernel.cuh WTDenUpdateKernel.cuh WT.cuh WTDense.cuh WTChunk.cuh WTAddKernel.cuh SamplingKernel.cuh
	$(CXX) -c mainKernel.cu -o mainKernel.o

DTUpdateKernel.o : DTUpdateKernel.cu DTUpdateKernel.cuh DT.cuh DataChunk.cuh utility.cuh Argument.cuh Doc.cuh 
	$(CXX) -c DTUpdateKernel.cu -o DTUpdateKernel.o

WTUpdateKernel.o : WTUpdateKernel.cu WTUpdateKernel.cuh WT.cuh Argument.cuh WTChunk.cuh DataChunk.cuh utility.cuh Doc.cuh 

	$(CXX) -c WTUpdateKernel.cu -o WTUpdateKernel.o

WTDenUpdateKernel.o : WTDenUpdateKernel.cu WTDenUpdateKernel.cuh WT.cuh Argument.cuh WTChunk.cuh WTDense.cuh utility.cuh Doc.cuh DataChunk.cuh 

	$(CXX) -c WTDenUpdateKernel.cu -o WTDenUpdateKernel.o

WTAddKernel.o : WTAddKernel.cu WTAddKernel.cuh WT.cuh Argument.cuh WTChunk.cuh utility.cuh Doc.cuh DataChunk.cuh WTUpdateKernel.cuh

	$(CXX) -c WTAddKernel.cu -o WTAddKernel.o

SamplingKernel.o : SamplingKernel.cu SamplingKernel.cuh Argument.cuh utility.cuh Doc.cuh DataChunk.cuh WT.cuh WTChunk.cuh WTDense.cuh DT.cuh

	$(CXX) -c SamplingKernel.cu -o SamplingKernel.o

DT.o : DT.cu DT.cuh

	$(CXX) -c DT.cu -o DT.o

Doc.o : Doc.cu Doc.cuh DataChunk.cuh Argument.cuh

	$(CXX) -c Doc.cu -o Doc.o

WT.o : WT.cu WT.cuh Argument.cuh WTChunk.cuh

	$(CXX) -c WT.cu -o WT.o

WTDense.o : WTDense.cu WTDense.cuh WTChunk.cuh Argument.cuh

	$(CXX) -c WTDense.cu -o WTDense.o

WTChunk.o : WTChunk.cu WTChunk.cuh Argument.cuh
	
	$(CXX) -c WTChunk.cu -o WTChunk.o

DataChunk.o : DataChunk.cu DataChunk.cuh

	$(CXX) -c DataChunk.cu -o DataChunk.o

utility.o : utility.cu utility.cuh

	$(CXX) -c utility.cu -o utility.o



clean:
	rm -f pslda
	rm -f *.o
