NVCC ?= nvcc

# GPU compute capability. Default targets the Jetson Nano (Maxwell, sm_53).
# Override for other hardware, e.g.:
#   make ARCH=sm_75   (Turing, e.g. Colab/Kaggle T4)
#   make ARCH=sm_86   (Ampere, e.g. RTX 30xx)
ARCH ?= sm_53

NVCCFLAGS = -O3 -arch=$(ARCH) -lm

ANN_SRCS = main.cu ann.cu matrix.cu mnist.cu
TEST_SRCS = test.cu matrix.cu

.PHONY: all clean profile

all: ann

ann: $(ANN_SRCS) ann.h matrix.h mnist.h err.h
	$(NVCC) $(NVCCFLAGS) -o $@ $(ANN_SRCS)

test: $(TEST_SRCS) matrix.h err.h
	$(NVCC) $(NVCCFLAGS) -o $@ $(TEST_SRCS)

# Host-side profiling with gprof (CUDA kernels themselves are not covered by
# gprof; use `nvprof` or Nsight Systems/Compute to profile GPU kernels).
profile: NVCCFLAGS += -Xcompiler -pg -Xlinker -pg
profile: ann

clean:
	rm -f ann test gmon.out *.o
