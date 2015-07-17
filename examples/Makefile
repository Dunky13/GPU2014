
debug	:= 0
include	../Makefile.inc


all: clean vectorAddCUDA vectorAddCL
 
vectorAddCUDA: vectorAdd.cu
	$(NVCC) -o $(PROJ_BASE)/bin/vectorAddCUDA vectorAdd.cu $(PROJ_BASE)/src/Timer.cpp $(INCLUDES) $(NVCFLAGS) $(NVLDFLAGS)

vectorAddCL: vectorAdd.cl.cpp
	$(CC) -o $(PROJ_BASE)/bin/vectorAddCL vectorAdd.cl.cpp $(PROJ_BASE)/src/Timer.cpp $(INCLUDES) $(OPENCL_INCLUDES) $(OPENCL_LIBS) $(CLLDFLAGS) $(CFLAGS)

clean:
	rm -f $(PROJ_BASE)/bin/vectorAddCUDA
	rm -f $(PROJ_BASE)/bin/vectorAddCL

