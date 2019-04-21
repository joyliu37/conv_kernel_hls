// ==============================================================
// File generated by Vivado(TM) HLS - High-Level Synthesis from C, C++ and SystemC
// Version: 2016.4
// Copyright (C) 1986-2017 Xilinx, Inc. All Rights Reserved.
// 
// ==============================================================

// control
// 0x00 : Control signals
//        bit 0  - ap_start (Read/Write/COH)
//        bit 1  - ap_done (Read/COR)
//        bit 2  - ap_idle (Read)
//        bit 3  - ap_ready (Read)
//        bit 7  - auto_restart (Read/Write)
//        others - reserved
// 0x04 : Global Interrupt Enable Register
//        bit 0  - Global Interrupt Enable (Read/Write)
//        others - reserved
// 0x08 : IP Interrupt Enable Register (Read/Write)
//        bit 0  - Channel 0 (ap_done)
//        bit 1  - Channel 1 (ap_ready)
//        others - reserved
// 0x0c : IP Interrupt Status Register (Read/TOW)
//        bit 0  - Channel 0 (ap_done)
//        bit 1  - Channel 1 (ap_ready)
//        others - reserved
// 0x10 : Data signal of Ksz
//        bit 15~0 - Ksz[15:0] (Read/Write)
//        others   - reserved
// 0x14 : reserved
// 0x18 : Data signal of Xsz
//        bit 15~0 - Xsz[15:0] (Read/Write)
//        others   - reserved
// 0x1c : reserved
// 0x20 : Data signal of Ysz
//        bit 15~0 - Ysz[15:0] (Read/Write)
//        others   - reserved
// 0x24 : reserved
// 0x28 : Data signal of X_n
//        bit 15~0 - X_n[15:0] (Read/Write)
//        others   - reserved
// 0x2c : reserved
// 0x30 : Data signal of Y_n
//        bit 15~0 - Y_n[15:0] (Read/Write)
//        others   - reserved
// 0x34 : reserved
// 0x38 : Data signal of Cin_n
//        bit 15~0 - Cin_n[15:0] (Read/Write)
//        others   - reserved
// 0x3c : reserved
// 0x40 : Data signal of Cin_SZ
//        bit 15~0 - Cin_SZ[15:0] (Read/Write)
//        others   - reserved
// 0x44 : reserved
// 0x48 : Data signal of Cout_n
//        bit 15~0 - Cout_n[15:0] (Read/Write)
//        others   - reserved
// 0x4c : reserved
// 0x50 : Data signal of Cout_SZ
//        bit 15~0 - Cout_SZ[15:0] (Read/Write)
//        others   - reserved
// 0x54 : reserved
// 0x58 : Data signal of Stride
//        bit 15~0 - Stride[15:0] (Read/Write)
//        others   - reserved
// 0x5c : reserved
// 0x60 : Data signal of pool
//        bit 0  - pool[0] (Read/Write)
//        others - reserved
// 0x64 : reserved
// (SC = Self Clear, COR = Clear on Read, TOW = Toggle on Write, COH = Clear on Handshake)

#define XHLS_TARGET_CONTROL_ADDR_AP_CTRL      0x00
#define XHLS_TARGET_CONTROL_ADDR_GIE          0x04
#define XHLS_TARGET_CONTROL_ADDR_IER          0x08
#define XHLS_TARGET_CONTROL_ADDR_ISR          0x0c
#define XHLS_TARGET_CONTROL_ADDR_KSZ_DATA     0x10
#define XHLS_TARGET_CONTROL_BITS_KSZ_DATA     16
#define XHLS_TARGET_CONTROL_ADDR_XSZ_DATA     0x18
#define XHLS_TARGET_CONTROL_BITS_XSZ_DATA     16
#define XHLS_TARGET_CONTROL_ADDR_YSZ_DATA     0x20
#define XHLS_TARGET_CONTROL_BITS_YSZ_DATA     16
#define XHLS_TARGET_CONTROL_ADDR_X_N_DATA     0x28
#define XHLS_TARGET_CONTROL_BITS_X_N_DATA     16
#define XHLS_TARGET_CONTROL_ADDR_Y_N_DATA     0x30
#define XHLS_TARGET_CONTROL_BITS_Y_N_DATA     16
#define XHLS_TARGET_CONTROL_ADDR_CIN_N_DATA   0x38
#define XHLS_TARGET_CONTROL_BITS_CIN_N_DATA   16
#define XHLS_TARGET_CONTROL_ADDR_CIN_SZ_DATA  0x40
#define XHLS_TARGET_CONTROL_BITS_CIN_SZ_DATA  16
#define XHLS_TARGET_CONTROL_ADDR_COUT_N_DATA  0x48
#define XHLS_TARGET_CONTROL_BITS_COUT_N_DATA  16
#define XHLS_TARGET_CONTROL_ADDR_COUT_SZ_DATA 0x50
#define XHLS_TARGET_CONTROL_BITS_COUT_SZ_DATA 16
#define XHLS_TARGET_CONTROL_ADDR_STRIDE_DATA  0x58
#define XHLS_TARGET_CONTROL_BITS_STRIDE_DATA  16
#define XHLS_TARGET_CONTROL_ADDR_POOL_DATA    0x60
#define XHLS_TARGET_CONTROL_BITS_POOL_DATA    1

