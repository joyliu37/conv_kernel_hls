XILINX_ROOT = /cad/xilinx/vivado/2017.2/Vivado_HLS/2017.2	
#STENCIL_ROOT = /home/joeyliu/Halide-HLS/apps/hls_examples/hls_support
#RUNTIME_ROOT = /home/joeyliu/Halide-HLS/src/runtime
CROSS_COMPILE=aarch64-linux-gnu-gcc
HLS_IFLAG += -I ${XILINX_ROOT}
#HLS_IFLAG += -I ${STENCIL_ROOT}/
#HLS_IFLAG += -I ${RUNTIME_ROOT}/
all:
	$(CROSS_COMPILE) -static -std=c++11 -Wall -fomit-frame-pointer -Wno-unused-label -Wno-uninitalized -Wno-literal-suffix -Wno-unknown-pragmas linux_gemm_app.cpp  -lpthread -ldl -o run_zynq
