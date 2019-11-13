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

set hwdir "$srcdir/hw/test/$app"
puts "HW top dir: $hwdir"

set halide_include "$::env(HALIDE_HLS_ROOT)/include"
puts "Halide include dir: $halide_include"

set hls_support "$::env(HALIDE_HLS_ROOT)/apps/hls_examples/hls_support"
puts "HLS support dir: $hls_support"

open_project "test_$app"
set_top top
add_files $hwdir/top.cpp -cflags "-std=c++0x -D_GLIBCXX_USE_CXX11_ABI=0 -v -I$libdir -I$halide_include -I$hls_support -Wno-parentheses-equality -Wno-deprecated-register -Wno-tautological-compare "
add_files -tb $hwdir/host.cpp -cflags "-std=c++0x -D_GLIBCXX_USE_CXX11_ABI=0 -v -I$hwdir -I$hostdir -I$libdir -I$halide_include -I$hls_support -Wno-parentheses-equality -Wno-deprecated-register -Wno-tautological-compare "
open_solution "unit_test_256"
set_part {xczu9eg-ffvb1156-2-i-es2} -tool vivado
create_clock -period 4 -name default
#source "./hls_cnn_db/solution_conf1/directives.tcl"
#csim_design -clean -compiler clang
#csynth_design
#cosim_design -compiler clang -mflags "ExtraCXXFlags=-D_GLIBCXX_USE_CXX11_ABI=0" -trace_level all
export_design -rtl verilog -format ip_catalog
