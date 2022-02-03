source /cad/modules/tcl/init/bash
module load base
module load vivado/2017.1
export HALIDE_HLS_ROOT=~/ahaFromDocker/aha/Halide-to-Hardware/

mkdir pranil/$2

if [ "$#" -ne 2 ]; then
    echo "cmd template: $0 <network>" >&2
    echo "<network> = mobilenet, vgg" >&2
    exit 1
fi

NETWORK=$1

if [ -z "$HALIDE_HLS_ROOT"  ]; then
    echo "Need to set HALIDE_HLS_ROOT"
    exit 1
fi

src_dir=src
#module load base vivado/2017.2
#vivado_hls=vivado_hls
#vivado_hls=/nobackup/xuany/xilinx/Vivado_HLS/2016.4/bin/vivado_hls 
#vivado_hls=/cad/xilinx/vivado/2017.2/Vivado_HLS/2017.2/bin/vivado_hls
# vivado_hls=/cad/xilinx/vivado/2019.2/Vivado/2019.2/bin/vivado_hls

vivado_hls -f run_hls.tcl -tclargs $src_dir $NETWORK

cp out.log pranil/$2/
cp src/hw/vgg/config.h pranil/$2/
cp src/hw/vgg/hls_target.cpp pranil/$2/
cp src/hw/vgg/hls_test.cpp pranil/$2/
