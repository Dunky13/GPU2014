
debug := 0
include ../Makefile.inc


all: clean exercise00.cu exercise01.cu exercise02.cu exercise00.cl exercise01.cl exercise02.cl

exercise00.cu: 00_main.cu
	$(NVCC) -o $(PROJ_BASE)/bin/exercise00.cu 00_main.cu $(PROJ_BASE)/src/Timer.cpp $(INCLUDES) $(NVCFLAGS) $(NVLDFLAGS) $(IMGLDFLAGS)

exercise01.cu: 01_main.cu
	$(NVCC) -o $(PROJ_BASE)/bin/exercise01.cu 01_main.cu $(PROJ_BASE)/src/Timer.cpp $(INCLUDES) $(NVCFLAGS) $(NVLDFLAGS) $(IMGLDFLAGS)

exercise02.cu: 02_main.cu
	$(NVCC) -o $(PROJ_BASE)/bin/exercise02.cu 02_main.cu $(PROJ_BASE)/src/Timer.cpp $(INCLUDES) $(NVCFLAGS) $(NVLDFLAGS)

exercise00.cl: 00_main.cl.cpp
	$(CC) -o $(PROJ_BASE)/bin/exercise00.cl 00_main.cl.cpp $(PROJ_BASE)/src/Timer.cpp $(INCLUDES) $(OPENCL_INCLUDES) $(OPENCL_LIBS) $(CLLDFLAGS) $(LDFLAGS) $(IMGLDFLAGS)

exercise01.cl: 01_main.cl.cpp
	$(CC) -o $(PROJ_BASE)/bin/exercise01.cl 01_main.cl.cpp $(PROJ_BASE)/src/Timer.cpp $(INCLUDES) $(OPENCL_INCLUDES) $(OPENCL_LIBS) $(CLLDFLAGS) $(LDFLAGS) $(IMGLDFLAGS)

exercise02.cl: 02_main.cl.cpp
	$(CC) -o $(PROJ_BASE)/bin/exercise02.cl 02_main.cl.cpp $(PROJ_BASE)/src/Timer.cpp $(INCLUDES) $(OPENCL_INCLUDES) $(OPENCL_LIBS) $(CLLDFLAGS) $(LDFLAGS)

clean:
	rm -f $(PROJ_BASE)/bin/exercise00.cu
	rm -f $(PROJ_BASE)/bin/exercise01.cu
	rm -f $(PROJ_BASE)/bin/exercise02.cu
	rm -f $(PROJ_BASE)/bin/exercise00.cl
	rm -f $(PROJ_BASE)/bin/exercise01.cl
	rm -f $(PROJ_BASE)/bin/exercise02.cl

