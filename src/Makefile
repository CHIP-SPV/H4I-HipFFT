INCLUDE_PATH =

## uncomment if you do NOT have any of the intel env defined either with
## intel's setvars.sh script (see my setUpModules.sh script) or with 
## JLSE's CNDA oneapi module (non-CNDA still needs to set
## intel env via setvars.sh or uncomment this line if I remember correctly)
#INCLUDE_PATH += -I$(MKL_ROOT)/include

## needed to find the local headers (such as hipfft.h and descriptor.hpp)
INCLUDE_PATH += -I../include
INCLUDE_PATH += -I/home/ac.nichols/H4I/INSTALLATION/MKLSHIM_DEVELOP/include

## I think libmkl_intel_lp64 allows 32 bit ints but libmkl_intel_ilp64 requires
## 64 bit ints ... key word here is "think" ... need to read the intel docs again
LDFLAGS =
LDFLAGS += -shared
LDFLAGS += -L$(MKL_ROOT)/lib/intel64 -lmkl_sycl -lmkl_core -lmkl_intel_thread -lmkl_intel_lp64
#LDFLAGS += -L$(MKL_ROOT)/lib/intel64 -lmkl_sycl -lmkl_core -lmkl_intel_thread -lmkl_intel_ilp64

#LDFLAGS += -L$(HIP_PATH)/lib -lCHIP
LDFLAGS += -L$(HIP_PATH)/lib64 -lCHIP
LDFLAGS += -Wl,-rpath,/home/ac.nichols/H4I/INSTALLATION/MKLSHIM_DEVELOP/lib64
LDFLAGS += -L/home/ac.nichols/H4I/INSTALLATION/MKLSHIM_DEVELOP/lib64 -lMKLShim

LD=icpx
CXX=icpx

OBJS=hipfft.o
LIBNAME=libhipfft.so

## needed since I'm doing this on my office ubuntu machine without any form of hip installed
DEFINES=-D__HIP_PLATFORM_SPIRV__

## -fopenmp to allow the use of omp_get_wtime() for timings ... if desired
## -std=c++17 to match Wael's cmake
BOTH_FLAG=-fsycl
OPT_FLAGS=-O3 -std=c++17 -fPIC

%.o: %.cpp
	$(CXX) -c -o $@ $< $(OPT_FLAGS) $(CFLAGS) $(BOTH_FLAG) $(INCLUDE_PATH) $(DEFINES)

$(LIBNAME) : $(OBJS)
	$(LD) $(OBJS) $(BOTH_FLAG) $(LDFLAGS) -o $@

install:
	cp $(LIBNAME) $(LIB_DIR)/.
	cp ../include/* $(INCLUDE_DIR)/.
clean:
	rm -f *.x *.o

realclean:
	make clean
	rm -f *.mod *.s *~ $(LIBNAME)
