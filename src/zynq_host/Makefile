CROSS_COMPILE=aarch64-linux-gnu-g++

STENCIL_ROOT = ${HALIDE_HLS_ROOT}/apps/hls_examples/hls_support
XILINX_ROOT = /cad/xilinx/vivado/2016.4/Vivado_HLS/2016.4/include	
RUNTIME_ROOT = ${HALIDE_HLS_ROOT}/src/runtime

INCLUDE_DIRS := ../host include
INCLUDE_DIRS +=  ${XILINX_ROOT}
INCLUDE_DIRS += ${STENCIL_ROOT} ${RUNTIME_ROOT}

FLAGS := $(foreach include_dir, $(INCLUDE_DIRS), -I$(include_dir))
CXXFLAGS := -std=c++11 -pthread

all:
	$(CROSS_COMPILE) -static $(CXXFLAGS) $(FLAGS) -Wall -fomit-frame-pointer -Wno-unused-label -Wno-uninitalized -Wno-literal-suffix -Wno-unknown-pragmas linux_gemm_app.cpp -ldl -o run_zynq
