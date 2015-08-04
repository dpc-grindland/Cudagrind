CC= nvcc
NVCC= nvcc

# Should point to Valgrind's and CUDA driver's lib/lib64 and include directory
LIB=
INCLUDE=
# Defaults for HLRS' Laki cluster
ifeq ($(SITE_PLATFORM_NAME),laki)
   LIB=-L/usr/lib64/valgrind/ -L/opt/cuda/driver-5.0/lib64/
   INCLUDE=-I/usr/include/valgrind/ -I/opt/cuda/driver-5.0/include
# Other cases
else
   ifndef ($(LIB))
      $(error Unknown platform. Point LIB and INCLUDE to VALGRIND/CUDA driver)
   endif
endif

INCLUDE += -Isrc/

LINKFLAGS=-lcuda -lcudart -lpthread
# -std=gnu99 instead of -std=c99 to enable POSIX mutex
CFLAGS= -O3 -Xcompiler -std=gnu99
CUDACFLAGS= -g -G -O0 -arch sm_13

# Set to enable debug output
#DEBUGFLAG=-DCUDAGRIND_DEBUG

CUDAWRAP_OFILES = $(patsubst %.c,%.o,$(wildcard src/*.c))

CG_V_MAJOR  = $(shell ./version.sh major)
CG_V_MINOR  = $(shell ./version.sh minor)
CG_V_REV 	= $(shell ./version.sh revision)

.PHONY: all
all: libcudaWrap.so.$(CG_V_MAJOR).$(CG_V_MINOR).$(CG_V_REV) examples/example examples/example_rc

.PHONY: new
new: clean all

examples/example: examples/example.o
	$(CC) $^ -o $@ $(LINKFLAGS) $(LIB)
examples/example_rc: examples/example_rc.o
	$(CC) $^ -o $@ $(LINKFLAGS) $(LIB)

libcudaWrap.so.$(CG_V_MAJOR).$(CG_V_MINOR).$(CG_V_REV): $(CUDAWRAP_OFILES)
	$(CC) $(CFLAGS) $(LIB) $^ -shared -Xlinker -soname -Xlinker libcudaWrap.so.$(CG_V_MAJOR) -o $@
	ln -s -f libcudaWrap.so.$(CG_V_MAJOR).$(CG_V_MINOR).$(CG_V_REV) libcudaWrap.so.$(CG_V_MAJOR).$(CG_V_MINOR)
	ln -s -f libcudaWrap.so.$(CG_V_MAJOR).$(CG_V_MINOR).$(CG_V_REV) libcudaWrap.so.$(CG_V_MAJOR)
	ln -s -f libcudaWrap.so.$(CG_V_MAJOR).$(CG_V_MINOR).$(CG_V_REV) libcudaWrap.so
	
src/%.o: src/%.c
	$(CC) -Xcompiler -fPIC $(CFLAGS) -c $(INCLUDE) $^ $(DEBUGFLAG) -o $@

%.o: %.c
	$(CC) $(CFLAGS) -c $(INCLUDE) $^ $(DEBUGFLAG) -o $@
	
%.o: %.cu
	$(NVCC) $(CUDACFLAGS) -c $(INCLUDE) $^ $(DEBUGFLAG) -o $@
	
.PHONY: clean
clean:
	rm -f src/*.o
	rm -f libcudaWrap.so.$(CG_V_MAJOR).$(CG_V_MINOR).$(CG_V_REV)
	rm -f libcudaWrap.so.$(CG_V_MAJOR).$(CG_V_MINOR)
	rm -f libcudaWrap.so.$(CG_V_MAJOR)
	rm -f libcudaWrap.so
	rm -f examples/example examples/example_rc examples/*.o
