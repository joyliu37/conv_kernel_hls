############################################################
## This file is generated automatically by Vivado HLS.
## Please DO NOT edit it.
## Copyright (C) 1986-2017 Xilinx, Inc. All Rights Reserved.
############################################################
set srcdir [lindex $argv 2]
puts "HW source dir: $srcdir"

set libdir "$srcdir/library"
puts "HW library dir: $libdir"

set hwdir "$srcdir/hw"
puts "HW top dir: $hwdir"

set halide_include "$::env(HALIDE_HLS_ROOT)/include"
puts "Halide include dir: $halide_include"

set hls_support "$::env(HALIDE_HLS_ROOT)/apps/hls_examples/hls_support"
puts "HLS support dir: $hls_support"

open_project -reset hls_cnn_db
set_top hls_target
add_files $srcdir/library/doublebuffer.h -cflags "-std=c++0x -I$halide_include -I$hls_support"
add_files $srcdir/library/streamtools.h -cflags "-std=c++0x -I$halide_include -I$hls_support"
add_files $srcdir/library/util.h -cflags "-std=c++0x -I$halide_include -I$hls_support"
add_files $srcdir/library/dma.h -cflags "-std=c++0x -I$halide_include -I$hls_support"
add_files $srcdir/library/convkernel.h -cflags "-std=c++0x -I$halide_include -I$hls_support"
add_files $srcdir/hw/wrapper.h -cflags "-std=c++0x -I$libdir -I$halide_include -I$hls_support"
add_files $srcdir/hw/hls_target.cpp -cflags "-std=c++0x -I$libdir -I$halide_include -I$hls_support"
add_files $srcdir/hw/hls_target.h -cflags "-std=c++0x -I$libdir -I$halide_include -I$hls_support"
add_files $srcdir/host/conv_test.h -cflags "-std=c++0x -I$hwdir -I$libdir -I$halide_include -I$hls_support"
add_files -tb $srcdir/host/conv_test.cpp -cflags "-std=c++0x -I$hwdir -I$libdir -I$halide_include -I$hls_support"
open_solution "solution_tiny2"
set_part {xczu9eg-ffvb1156-2-i-es2} -tool vivado
create_clock -period 4 -name default
#source "./cnn_db_new/solution1/directives.tcl"
csim_design -clean -compiler gcc
csynth_design
cosim_design -trace_level all
export_design -rtl verilog -format ip_catalog
