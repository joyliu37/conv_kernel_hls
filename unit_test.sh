if [ "$#" -ne 1 ]; then
    echo "cmd template: $0 <testcase>" >&2
    echo "<network> = IObuf, featurebuf, a" >&2
    exit 1
fi

TEST_MODE=$1
SRC_DIR=src
TEST_CASE=${SRC_DIR}/hw/test

if [ -z "$HALIDE_HLS_ROOT"  ]; then
    echo "Need to set HALIDE_HLS_ROOT"
    exit 1
fi

#vivado_hls=/nobackup/xuany/xilinx/Vivado_HLS/2016.4/bin/vivado_hls 
vivado_hls=/cad/xilinx/vivado/2018.3/Vivado_HLS/2018.3/bin/vivado_hls

if [ "$TEST_MODE" == "a" ]; then
    echo "Test all test case."
    for TEST_DIR in `find $TEST_CASE -mindepth 1 -type d -exec basename {} \;`
    do
        echo "Test $TEST_DIR."
        $vivado_hls -f unit_test.tcl -tclargs $SRC_DIR $TEST_DIR
        echo "Finish test $TEST_DIR"
    done
else
    TEST_DIR=${TEST_MODE}
    echo "Test $TEST_DIR."
    $vivado_hls -f unit_test.tcl -tclargs $SRC_DIR $TEST_DIR
    echo "Finish test $TEST_DIR"
fi
