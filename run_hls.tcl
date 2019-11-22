############################################################
## This file is generated automatically by Vivado HLS.
## Please DO NOT edit it.
## Copyright (C) 1986-2017 Xilinx, Inc. All Rights Reserved.
############################################################
set srcdir [lindex $argv 2]
set app [lindex $argv 3]
puts "HW source dir: $srcdir"

set libdir "$srcdir/library"
puts "HW library dir: $libdir"

set hostdir "$srcdir/host"
puts "Host library dir: $hostdir"

set hwdir "$srcdir/hw/$app"
puts "HW top dir: $hwdir"

set halide_include "$::env(HALIDE_HLS_ROOT)/include"
puts "Halide include dir: $halide_include"

set hls_support "$::env(HALIDE_HLS_ROOT)/apps/hls_examples/hls_support"
puts "HLS support dir: $hls_support"

open_project -reset hls_cnn_db_${app}
#open_project hls_cnn_db
set_top hls_target
#add_files $srcdir/library/doublebuffer.h -cflags "-std=c++0x -I$halide_include -I$hls_support"
#add_files $srcdir/library/addrgen.h -cflags "-std=c++0x -I$halide_include -I$hls_support"
#add_files $srcdir/library/streamtools.h -cflags "-std=c++0x -I$halide_include -I$hls_support"
#add_files $srcdir/library/util.h -cflags "-std=c++0x -I$halide_include -I$hls_support"
#add_files $srcdir/library/dma.h -cflags "-std=c++0x -I$halide_include -I$hls_support"
#add_files $srcdir/library/convkernel.h -cflags "-std=c++0x -I$halide_include -I$hls_support"
#add_files $srcdir/hw/mobilenet/wrapper.h -cflags "-std=c++0x -I$libdir -I$halide_include -I$hls_support"
add_files $hwdir/hls_target.cpp -cflags "-std=c++0x -I$libdir -I$halide_include -I$hls_support -Wno-parentheses-equality -Wno-deprecated-register -Wno-tautological-compare"
#add_files $srcdir/hw/mobilenet/hls_target.h -cflags "-std=c++0x -I$libdir -I$halide_include -I$hls_support"
#add_files $srcdir/host/conv_test.h -cflags "-std=c++0x -I$hwdir -I$libdir -I$halide_include -I$hls_support"
add_files -tb $hwdir/hls_test.cpp -cflags "-std=c++0x -I$hwdir -I$hostdir -I$libdir -I$halide_include -I$hls_support -Wno-parentheses-equality -Wno-deprecated-register -Wno-tautological-compare "
open_solution -reset "test_pynq_debug"
#set_part {xczu9eg-ffvb1156-2-i-es2} -tool vivado
set_part {xc7z020clg484-1} -tool vivado

create_clock -period 7 -name default
#source "./hls_cnn_db/solution_conf1/directives.tcl"
csim_design -clean -compiler clang
csynth_design
cosim_design -compiler clang -mflags "ExtraCXXFlags=-D_GLIBCXX_USE_CXX11_ABI=0" -trace_level all
export_design -rtl verilog -format ip_catalog
