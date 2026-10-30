NVCC ?= nvcc
CXX ?= g++
NVCC_FLAGS ?= -O3 -std=c++17 --fmad=false -Xcompiler -ffp-contract=off
CXX_FLAGS ?= -O3 -std=c++17 -ffp-contract=off

.PHONY: all cpu gpu clean
all: cpu gpu
cpu:
	mkdir -p build
	$(CXX) $(CXX_FLAGS) src/cpu_only.cpp -o build/knn_cpu
gpu:
	mkdir -p build
	$(NVCC) $(NVCC_FLAGS) src/main.cu -o build/knn_cuda
clean:
	rm -rf build
