if [ "$#" -ne 1 ]; then
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
vivado_hls=/nobackup/xuany/xilinx/Vivado_HLS/2016.4/bin/vivado_hls 

$vivado_hls -f run_hls.tcl -tclargs $src_dir $NETWORK
