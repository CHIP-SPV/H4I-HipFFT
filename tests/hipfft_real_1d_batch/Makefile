INCLUDE_PATH =

## uncomment if you do NOT have any of the intel env defined
#INCLUDE_PATH += -I$(MKL_ROOT)/include

## needed to find the local headers (such as hipfft.h and descriptor.hpp)
PATH_BASE = /home/nicholsds/FROM_CHIPSTAR/INSTALLATION
INCLUDE_PATH += -I$(PATH_BASE)/H4I-HIPFFT/include
INCLUDE_PATH += -I$(PATH_BASE)/MKLSHIM_DEVELOP/include

## I think libmkl_intel_lp64 allows 32 bit ints but libmkl_intel_ilp64 requires
## 64 bit ints ... key word here is "think" ... need to read the intel docs again
LDFLAGS =
#LDFLAGS += -L$(MKL_ROOT)/lib/intel64 -lmkl_sycl -lmkl_core -lmkl_intel_thread -lmkl_intel_lp64
LDFLAGS += -L$(MKL_ROOT)/lib/intel64 -lmkl_sycl -lmkl_core -lmkl_intel_thread -lmkl_intel_ilp64

#LDFLAGS += -L$(HIP_PATH)/lib -lCHIP
LDFLAGS += -L$(HIP_PATH)/lib64 -lCHIP
LDFLAGS += -Wl,-rpath,$(PATH_BASE)/MKLSHIM_DEVELOP/lib64
LDFLAGS += -L$(PATH_BASE)/MKLSHIM_DEVELOP/lib64 -lMKLShim
LDFLAGS += -Wl,-rpath,$(PATH_BASE)/H4I-HIPFFT/lib64
LDFLAGS += -L$(PATH_BASE)/H4I-HIPFFT/lib64 -lHipFFT

#LD=icpx
#CXX=icpx
LD=clang++
CXX=clang++

OBJS=hipfft_real_1d_batch.o
EXE=hipfft_real_1d_batch.x

## needed since I'm doing this on my office ubuntu machine without any form of hip installed
DEFINES=-D__HIP_PLATFORM_SPIRV__

## -fopenmp to allow the use of omp_get_wtime() for timings ... if desired
## -std=c++17 to match Wael's cmake
OPENMP_FLAG=-fopenmp #-fsycl
OPT_FLAGS=-O3 -std=c++17

%.o: %.cpp
	$(CXX) -c -o $@ $< $(OPT_FLAGS) $(CFLAGS) $(OPENMP_FLAG) $(INCLUDE_PATH) $(DEFINES)

$(EXE) : $(OBJS)
	$(LD) $(OBJS) $(OPENMP_FLAG) $(LDFLAGS) -o $@

clean:
	rm -f *.x *.o

realclean:
	make clean
	rm -f *.mod *.s *~ include/*~
