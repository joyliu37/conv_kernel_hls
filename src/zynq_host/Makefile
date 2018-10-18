XILINX_ROOT = /cad/xilinx/vivado/2016.4/Vivado_HLS/2016.4
STENCIL_ROOT = /home/joeyliu/Halide-HLS/apps/hls_examples/hls_support
RUNTIME_ROOT = /home/joeyliu/Halide-HLS/src/runtime

HLS_IFLAG += -I ${XILINX_ROOT}/include
HLS_IFLAG += -I ${STENCIL_ROOT}/
HLS_IFLAG += -I ${RUNTIME_ROOT}/
all:
	$(CROSS_COMPILE)g++-5 -static -std=c++11 $(HLS_IFLAG) -Wall -fomit-frame-pointer -Wno-unused-label -Wno-uninitalized -Wno-literal-suffix -Wno-unknown-pragmas linux_gemm_app.cpp  -lpthread -ldl -o run_zynq
