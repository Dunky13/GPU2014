
PROJ_BASE	:= $(HOME)/gpu2014

INCLUDES := -I"$(PROJ_BASE)/include"
CUDA_LIBS := -L"$(CUDA_INSTALL_PATH)/lib64/"
OPENCL_INCLUDES := -I"$(OPENCL_INCLUDE)"
OPENCL_LIBS := -L"$(OPENCL_LIB)"


CFLAGS := -std=c++0x
NVCFLAGS	:= -arch compute_20 -code sm_20 --ptxas-options=-v -w
ifneq ($(debug), 1)
	CFLAGS += -O3 -g0
	NVCFLAGS	+= -O3
else
	CFLAGS += -O0 -g3
	NVCFLAGS += -O0
endif

LDFLAGS := -lm
NVLDFLAGS := $(LDFLAGS) -lrt -lcudart
IMGLDFLAGS := -lX11 -lpthread
CLLDFLAGS := -lOpenCL

CC := g++
NVCC := nvcc

