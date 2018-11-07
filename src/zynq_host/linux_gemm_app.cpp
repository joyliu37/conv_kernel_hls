#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
//#include "Stencil.h"
//#include <time.h>

//#include <asm/cachectl.h>

// The purpose this test is to show that users can get to devices in user
// mode .This is not to say this should replace a kernel driver, but does
// provide some short term solutions sometimes
// or a debug solution that can be helpful.


// This test was derived from devmem2.c.

//#define GEMM_BASE_ADDRESS     0x43C00000
#define GEMM_BASE_ADDRESS     0x00A0000000
#define GPIO_DATA_OFFSET     0
#define GPIO_DIRECTION_OFFSET     4

#define DDR_BASE_ARG0_ADDRESS    0x30000000
#define DDR_BASE_ARG1_ADDRESS    0x10000000
#define DDR_BASE_ARG2_ADDRESS    0x20000000

#define XGPIO_CHAN_OFFSET  8

// config
// 0x0 : Control signals
//       bit 0  - ap_start (Read/Write/COH)
//       bit 1  - ap_done (Read/COR)
//       bit 2  - ap_idle (Read)
//       bit 3  - ap_ready (Read)
//       bit 7  - auto_restart (Read/Write)
//       others - reserved
// 0x4 : Global Interrupt Enable Register
//       bit 0  - Global Interrupt Enable (Read/Write)
//       others - reserved
// 0x8 : IP Interrupt Enable Register (Read/Write)
//       bit 0  - Channel 0 (ap_done)
//       bit 1  - Channel 1 (ap_ready)
//       others - reserved
// 0xc : IP Interrupt Status Register (Read/TOW)
//       bit 0  - Channel 0 (ap_done)
//       bit 1  - Channel 1 (ap_ready)
//       others - reserved
// (SC = Self Clear, COR = Clear on Read, TOW = Toggle on Write, COH = Clear on Handshake)
#define XHLS_TARGET_CONFIG_ADDR_AP_CTRL 0x0
#define XHLS_TARGET_CONFIG_ADDR_GIE     0x4
#define XHLS_TARGET_CONFIG_ADDR_IER     0x8
#define XHLS_TARGET_CONFIG_ADDR_ISR     0xc

#define XHLS_TARGET_CR_RESET_MASK 0x00000004

#define XHLS_TARGET_AP_START_MASK 0x00000001
#define XHLS_TARGET_AP_DONE_MASK 0x00000002
#define XHLS_TARGET_AP_IDLE_MASK 0x00000004

//#define XAXICDMA_CR_OFFSET      0x00000000  /**< Control register */
//#define XAXICDMA_SR_OFFSET      0x00000004  /**< Status register */
//#define XAXICDMA_CDESC_OFFSET   0x00000008  /**< Current descriptor pointer */
//#define XAXICDMA_TDESC_OFFSET   0x00000010  /**< Tail descriptor pointer */
//#define XAXICDMA_SRCADDR_OFFSET 0x00000018  /**< Source address register */
//#define XAXICDMA_DSTADDR_OFFSET 0x00000020  /**< Destination address register */
//#define XAXICDMA_BTT_OFFSET     0x00000028  /**< Bytes to transfer */

/** @name Bitmasks of XAXICDMA_CR_OFFSET register
 * @{
 */
//#define XAXICDMA_CR_RESET_MASK  0x00000004 /**< Reset DMA engine */
//#define XAXICDMA_CR_SGMODE_MASK 0x00000008 /**< Scatter gather mode */

/** @name Bitmask for interrupts
 * These masks are shared by XAXICDMA_CR_OFFSET register and
 * XAXICDMA_SR_OFFSET register
 * @{
 */
//#define XAXICDMA_XR_IRQ_IOC_MASK        0x00001000 /**< Completion interrupt */
//#define XAXICDMA_XR_IRQ_DELAY_MASK      0x00002000 /**< Delay interrupt */
//#define XAXICDMA_XR_IRQ_ERROR_MASK      0x00004000 /**< Error interrupt */
//#define XAXICDMA_XR_IRQ_ALL_MASK        0x00007000 /**< All interrupts */
//#define XAXICDMA_XR_IRQ_SIMPLE_ALL_MASK 0x00005000 /**< All interrupts for
//                                                        simple only mode */
/*@}*/

/** @name Bitmasks of XAXICDMA_SR_OFFSET register
 * This register reports status of a DMA channel, including
 * idle state, errors, and interrupts
 * @{
 */
//#define XAXICDMA_SR_IDLE_MASK         0x00000002  /**< DMA channel idle */
//#define XAXICDMA_SR_SGINCLD_MASK      0x00000008  /**< Hybrid build */
//#define XAXICDMA_SR_ERR_INTERNAL_MASK 0x00000010  /**< Datamover internal err */
//#define XAXICDMA_SR_ERR_SLAVE_MASK    0x00000020  /**< Datamover slave err */
//#define XAXICDMA_SR_ERR_DECODE_MASK   0x00000040  /**< Datamover decode err */
//#define XAXICDMA_SR_ERR_SG_INT_MASK   0x00000100  /**< SG internal err */
//#define XAXICDMA_SR_ERR_SG_SLV_MASK   0x00000200  /**< SG slave err */
//#define XAXICDMA_SR_ERR_SG_DEC_MASK   0x00000400  /**< SG decode err */
//#define XAXICDMA_SR_ERR_ALL_MASK      0x00000770  /**< All errors */
/*@}*/

#define MAP_SIZE 4096UL
#define MAP_MASK (MAP_SIZE - 1)

#define DDR_MAP_SIZE 0x10000000
//#define DDR_MAP_SIZE 0x1000000000
#define DDR_MAP_MASK (DDR_MAP_SIZE - 1)

//#define DDR_WRITE_OFFSET 0x10000000


#define WEIGHT_BYTESIZE 3*3*64*64
#define OUTPUT_BYTESIZE 32*32*128
#define INPUT_BYTESIZE 32*32*32//1048576
#define DATAWIDTH 32

#define P_COUT 8
#define P_CIN 8
//#define BUFFER_BYTESIZE         262144  // Length of the buffers for DMA transfer

/*void clearcache(char* begin, char* end)
{
    const int syscall = 0xf0002;
    __asm__ volatile (
        "mov r0, %0\n"
        "mov r1, %1\n"
        "mov r7, %2\n"
        "mov r2, #0x0\n"
        "svc 0x00000000\n"
        :
        :  "r" (begin), "r" (end), "r" (syscall)
        :  "r0", "r1", "r7");
}*/
typedef int8_t dtype;
typedef uint8_t dtype_u;

    static int row = 32;
    static int col = 32;
    static int iCh = 32;
    static int oCh = 128;
    static int Ksz = 3;

void initial_input(dtype *image){
    for (int c = 0; c < iCh; c++){
        for (int j = 0; j < row; j++){
            for (int i = 0; i < col; i++){
                image[c * (row * col) + j * col + i] = (dtype)(abs(j - i) + c);
            }
        }
    }
}
/*
void image2stencil(dtype *image, PackedStencil<dtype, DATAWIDTH, 1, 1, 1> *image_stencil){
    for (int i = 0; i < row; i++){
        for (int j = 0; j < col; j++){
            for (int c = 0; c < iCh/DATAWIDTH; c++){
                Stencil<dtype, DATAWIDTH, 1, 1, 1> temp;
                for (int i_pack = 0; i_pack < DATAWIDTH; i_pack++){
                    temp(i_pack, 0, 0, 0) = image[i * (col * iCh) + j*iCh + c*DATAWIDTH + i_pack];
                }
                image_stencil[i * (col*iCh/DATAWIDTH) + j*iCh/DATAWIDTH + c] = temp;
            }
        }
    }
}*/

void weight_reshape(dtype *weight, dtype *reshape_weight){
        //dtype reshape_weight[row*col*iCh*oCh];

        for (int coutBlk = 0; coutBlk < oCh / P_COUT; coutBlk ++){
            for (int cinBlk = 0; cinBlk < iCh / P_CIN; cinBlk ++){
                for (int yOff = 0; yOff < Ksz; yOff ++){
                    for (int xOff = 0; xOff < Ksz; xOff ++){
                        for (int ii = 0; ii < P_COUT; ii ++){
                            for (int jj = 0; jj < P_CIN; jj ++){
                                int addr_org = (coutBlk * P_COUT + ii)*Ksz*Ksz*iCh +\
                                                yOff*Ksz*iCh + xOff*iCh +\
                                                cinBlk*P_CIN + jj;
                                int addr_new = coutBlk * Ksz * Ksz * iCh * P_COUT +\
                                                cinBlk * Ksz * Ksz * P_COUT * P_CIN +\
                                                yOff * Ksz * P_CIN * P_COUT +\
                                                xOff * P_CIN * P_COUT +\
                                                ii * P_CIN + jj;
                                reshape_weight[addr_new] = weight[addr_org];
                            }
                        }
                    }
                }
            }
        }


}

void initial_ouput(dtype *image){
    for (int c = 0; c < oCh; c++){
        for (int j = 0; j < row; j++){
            for (int i = 0; i < col; i++){
                image[c * (row * col) + j * col + i] = 0;
            }
        }
    }
}

void initial_weight(dtype *weight){
    for (int idx = 0; idx < Ksz; idx ++){
        for (int idy = 0; idy < Ksz; idy ++){
            for (int idi = 0; idi < iCh; idi ++){
                for (int ido = 0; ido < oCh; ido ++){
                    weight[ido*iCh*Ksz*Ksz + idi*Ksz*Ksz + idy*Ksz + idx] = (dtype)(idx-idy);
                }
            }
        }
    }
}

int check_err(dtype* input, dtype* weight, dtype* output){
    //compute the sw output
    dtype output_sw_tmp[row * col * oCh];
    dtype output_sw[row*col*oCh];
    int err_cnt = 0;
    for (int i = 0; i < row * col * oCh; i++){
        output_sw_tmp[i] = 0;
    }

    for (int k = 0; k <oCh; k++){
        for (int y = 0; y < row; y++){
            for (int x = 0; x <col; x++){
                for(int c = 0; c < iCh; c++){
                    for(int fy = 0; fy < Ksz; fy ++){
                        for(int fx = 0; fx < Ksz; fx ++){
                            if( (y+fy > 0) && (x+fx > 0) && (y+fy < row+1) && (x+fx < col+1) ){
                                output_sw_tmp[y*col*oCh + x*oCh + k] += \
                                    (dtype)(input[(y+fy-1) *col *iCh + (x+fx-1)*iCh + c]\
                                    * weight[k * Ksz * Ksz * iCh + fy*Ksz*iCh + fx*iCh + c]);
                            }
                        }
                    }
                }
                output_sw[y*col*oCh + x*oCh +k] = output_sw_tmp[y*col*oCh + x*oCh +k];
                if(output_sw[y*col*oCh + x*oCh + k] < 0 )
                {
                    //printf("%x enter ReLU.\n", output_sw[y*col*oCh + x*oCh +k]);
                    output_sw[y*col*oCh + x*oCh + k] = 0;
                }
                if(output_sw[y*col*oCh + x*oCh + k] != output[y*col*oCh + x*oCh+ k])
                {
                    err_cnt ++;
                    printf("NO. %d: pos ( y:%d, x:%d, ch:%d) is diff---SW_res = %x---- HW_res = %x \n",
                            err_cnt, y, x, k,output_sw[y*col*oCh + x*oCh +k], output[y*col*oCh + x*oCh + k]);
                }

            }
        }
    }
    //return 0;
    return err_cnt;
}

int main()
{

    printf("enter the program!!\n");
    int memfd;
    void *mapped_base, *mapped_dev_base;
    off_t dev_base = GEMM_BASE_ADDRESS;

    int memfd_0;
    void *mapped_base_0, *mapped_dev_base_0;
    off_t dev_base_0 = DDR_BASE_ARG0_ADDRESS;

    int memfd_1;
    void *mapped_base_1, *mapped_dev_base_1;
    off_t dev_base_1 = DDR_BASE_ARG1_ADDRESS;

    //int memfd_2;
    void *mapped_base_2, *mapped_dev_base_2;
    off_t dev_base_2 = DDR_BASE_ARG2_ADDRESS;

    //unsigned int TimeOut = 5;
    unsigned int ResetMask;
    unsigned int RegValue;
    dtype SrcArray0[INPUT_BYTESIZE ];
    dtype SrcArray1[WEIGHT_BYTESIZE ];
    dtype SrcArray1_reshape[WEIGHT_BYTESIZE];
    dtype DestArray[OUTPUT_BYTESIZE ];
    //PackedStencil<dtype, DATAWIDTH, 1, 1, 1> SrcArray0_packed[INPUT_BYTESIZE/DATAWIDTH];

    //initial the parameter of experiment layer
    /*======================================================================================
     STEP 1 : Initialize the source buffer bytes with a pattern  and clear the Destination
                  location
         ========================================================================================*/
        /*for (Index = 0; Index < BUFFER_BYTESIZE; Index++)
        {
                        SrcArray0[Index] = 3;
                        SrcArray1[Index] = 1;
                        DestArray[Index] = 0;
        }*/
    initial_input(SrcArray0);
    //image2stencil(SrcArray0, SrcArray0_packed);
    initial_weight(SrcArray1);
    weight_reshape(SrcArray1, SrcArray1_reshape);
    initial_ouput(DestArray);
        /*======================================================================================
        STEP 2 : Map the kernel memory location starting from 0x20000000 to the User layer
        ========================================================================================*/
        memfd_1 = open("/dev/mem", O_RDWR | O_SYNC);
    if (memfd_1 == -1)
    {
        printf("Can't open /dev/mem.\n");
        exit(0);
    }
    printf("/dev/mem opened.\n");
    // Map one page of memory into user space such that the device is in that page, but it may not
    // be at the start of the page.

    mapped_base_1 = mmap(0, DDR_MAP_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, memfd_1, dev_base_1 & ~DDR_MAP_MASK);
    if (mapped_base_1 == (void *) -1)
    {
        printf("Can't map the memory to user space.\n");
        exit(0);
    }
    printf("Memory mapped at address %p.\n", mapped_base_1);

    // get the address of the device in user space which will be an offset from the base
    // that was mapped as memory is mapped at the start of a page
     mapped_dev_base_1 = mapped_base_1 + (dev_base_1 & DDR_MAP_MASK);
     /*======================================================================================
     STEP 3 : Copy the Data to the DDR Memory at location 0x20000000
     ========================================================================================*/
    memcpy(mapped_dev_base_1, SrcArray0, (INPUT_BYTESIZE)*sizeof(dtype));
    printf("Src0 set to: %lu\n", *((unsigned long *) mapped_dev_base_1));
    //cacheflush(mapped_dev_base_1, BUFFER_BYTESIZE, DCACHE);
    //clearcache(dev_base_1, dev_base_1+4096);
    /*======================================================================================
     STEP 4 : Un-map the kernel memory from the User layer.
    ========================================================================================*/
    if (munmap(mapped_base_1, DDR_MAP_SIZE) == -1)
    {
        printf("Can't unmap memory from user space.\n");
        exit(0);
    }
    //close(memfd_1);

    // Initialize arg2
    /*memfd_2 = open("/dev/mem", O_RDWR | O_SYNC);
    if (memfd_2 == -1)
    {
        printf("Can't open /dev/mem.\n");
        exit(0);
    }
    printf("/dev/mem opened.\n");
    */
    mapped_base_2 = mmap(0, DDR_MAP_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, memfd_1, dev_base_2 & ~DDR_MAP_MASK);
    if (mapped_base_2 == (void *) -1)
    {
        printf("Can't map the memory to user space.\n");
        exit(0);
    }
    printf("Memory mapped at address %p.\n", mapped_base_2);


    mapped_dev_base_2 = mapped_base_2 + (dev_base_2 & DDR_MAP_MASK);
    memcpy(mapped_dev_base_2, SrcArray1_reshape, (WEIGHT_BYTESIZE)*sizeof(dtype));
    printf("Src1 set to: %lu\n", *((unsigned long *) mapped_dev_base_2));
    //clearcache(dev_base_1, dev_base_1+4096);
    //cacheflush(mapped_dev_base_1, BUFFER_BYTESIZE, DCACHE);

    if (munmap(mapped_base_2, DDR_MAP_SIZE) == -1)
    {
        printf("Can't unmap memory from user space.\n");
        exit(0);
    }
    close(memfd_1);


        /*======================================================================================
        STEP 5 : Map the AXI GEMM Register memory to the User layer
                        Do the Register Setting for DMA transfer
        ========================================================================================*/
    memfd = open("/dev/mem", O_RDWR | O_SYNC);
    if (memfd == -1)
    {
        printf("Can't open /dev/mem.\n");
        exit(0);
    }
      printf("/dev/mem opened.\n");

    // Map one page of memory into user space such that the device is in that page, but it may not
    // be at the start of the page.
    mapped_base = mmap(0, MAP_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, memfd, dev_base & ~MAP_MASK);
    if (mapped_base == (void *) -1)
    {
          printf("Can't map the memory to user space.\n");
          exit(0);
      }
    // get the address of the device in user space which will be an offset from the base
    // that was mapped as memory is mapped at the start of a page
      struct timeval begin, end;
      gettimeofday(&begin, NULL);

    int iter;
    for (iter=0 ; iter < 1000; iter++) {
    //printf("iter: %d\n", iter);
    mapped_dev_base = mapped_base + (dev_base & MAP_MASK);
    //printf("GEMM was: %lu\n", *((unsigned long *) (mapped_dev_base + XHLS_TARGET_CONFIG_ADDR_AP_CTRL)));
    //Reset CDMA
      //do{
                  ResetMask = (unsigned long )XHLS_TARGET_CR_RESET_MASK;
                  *((volatile unsigned long *) (mapped_dev_base + XHLS_TARGET_CONFIG_ADDR_AP_CTRL)) = (unsigned long)ResetMask;
      //printf("Reset was: %lu\n", *((unsigned long *) (mapped_dev_base +  XHLS_TARGET_CONFIG_ADDR_AP_CTRL)));
                        /* If the reset bit is still high, then reset is not done       */
                        //ResetMask = *((volatile unsigned long *) (mapped_dev_base + XHLS_TARGET_CONFIG_ADDR_AP_CTRL));
                        //if(!(ResetMask & XHLS_TARGET_CR_RESET_MASK))
                        //{
                        //        break;
                        //}
                        //TimeOut -= 1;
      //}while (TimeOut);
      //printf("GEMM reset to: %lu\n", *((unsigned long *) (mapped_dev_base)));
        //disable Interrupt
      /*printf("Read was: %lu\n", *((unsigned long *) (mapped_dev_base + XHLS_TARGET_CONFIG_ADDR_IER)));
      RegValue = *((volatile unsigned long *) (mapped_dev_base + XHLS_TARGET_CONFIG_ADDR_GIE));
      //printf("Read was: %lu\n", *((unsigned long *) (mapped_dev_base + XHLS_TARGET_CONFIG_ADDR_GIE)));
      RegValue = (unsigned long)(RegValue & XHLS_TARGET_CR_RESET_MASK );
      *((volatile unsigned long *) (mapped_dev_base + XHLS_TARGET_CONFIG_ADDR_GIE)) = (unsigned long)RegValue;
      //printf("Read was: %lu\n", *((unsigned long *) (mapped_dev_base + XHLS_TARGET_CONFIG_ADDR_GIE)));

      RegValue = *((volatile unsigned long *) (mapped_dev_base + XHLS_TARGET_CONFIG_ADDR_IER));
      RegValue = (unsigned long)(RegValue & XHLS_TARGET_CR_RESET_MASK );
      *((volatile unsigned long *) (mapped_dev_base + XHLS_TARGET_CONFIG_ADDR_IER)) = (unsigned long)RegValue;

      printf("reset done\n");
      */
     // printf("GEMM reset to: %lu\n", *((unsigned long *) (mapped_dev_base)));
      // Checking for the Bus Idle
      /*RegValue = *((volatile unsigned long *) (mapped_dev_base + XHLS_TARGET_CONFIG_ADDR_AP_CTRL));
      if(!(RegValue & XHLS_TARGET_AP_IDLE_MASK))
      {
          printf("BUS IS BUSY Error Condition \n\r");
          return 1;
      }*/

      // Check the DMA Mode and switch it to simple mode
      //clock_t start = clock(), diff;
      //struct timeval begin, end;
      //gettimeofday(&begin, NULL);
      RegValue = *((volatile unsigned long *) (mapped_dev_base + XHLS_TARGET_CONFIG_ADDR_AP_CTRL));
      if(!(RegValue & XHLS_TARGET_AP_START_MASK))
      {
          RegValue = (unsigned long)(RegValue | XHLS_TARGET_AP_START_MASK);
          //printf("Reading \n \r");
          *((volatile unsigned long *) (mapped_dev_base + XHLS_TARGET_CONFIG_ADDR_AP_CTRL)) = (unsigned long)RegValue ;

      }
      //printf("start hw\n");
      //Set the Source Address
      //*((volatile unsigned long *) (mapped_dev_base + XAXICDMA_SRCADDR_OFFSET)) = (unsigned long)DDR_BASE_ADDRESS;
      //Set the Destination Address
      //*((volatile unsigned long *) (mapped_dev_base + XAXICDMA_DSTADDR_OFFSET)) = (unsigned long)DDR_BASE_WRITE_ADDRESS;
      //RegValue = (unsigned long)(BUFFER_BYTESIZE);
      // write Byte to Transfer
      //*((volatile unsigned long *) (mapped_dev_base + XAXICDMA_BTT_OFFSET)) = (unsigned long)RegValue;
        /*======================================================================================
        STEP 6 : Wait for the DMA transfer Status
        ========================================================================================*/
      do
      {
                  RegValue = *((volatile unsigned long *) (mapped_dev_base + XHLS_TARGET_CONFIG_ADDR_AP_CTRL));
      }while(!(RegValue & XHLS_TARGET_AP_DONE_MASK));
      //diff = clock() - start;
      //int usec = diff * 1000000 / CLOCKS_PER_SEC;
      //gettimeofday(&end, NULL);
      //unsigned int usec = end.tv_usec - begin.tv_usec;
      //printf("Time taken %d usec\n", usec);
      //printf("RegValue: %u\n", RegValue);
      }
      gettimeofday(&end, NULL);
      unsigned long usec = end.tv_sec - begin.tv_sec;
      usec *= 1000000;
      usec += end.tv_usec - begin.tv_usec;
      printf("Time taken %lu usec\n", usec);

      //if((RegValue & XAXICDMA_XR_IRQ_IOC_MASK))
      //{
      //    printf("Transfer Completed \n\r ");
      //}
      /*if((RegValue & XAXICDMA_XR_IRQ_DELAY_MASK))
      {
        printf("IRQ Delay Interrupt\n\r ");
      }
      if((RegValue & XAXICDMA_XR_IRQ_ERROR_MASK))
      {
        printf(" Transfer Error Interrupt\n\r ");
      }
      */
      /*======================================================================================
       STEP 7 : Un-map the AXI CDMA memory from the User layer.
      ========================================================================================*/
      if (munmap(mapped_base, MAP_SIZE) == -1)
      {
                printf("Can't unmap memory from user space.\n");
                exit(0);
      }

      close(memfd);

    /*======================================================================================
    STEP 8 : Map the kernel memory location starting from 0x30000000 to the User layer
    ========================================================================================*/
      memfd_0 = open("/dev/mem", O_RDWR | O_SYNC);
       if (memfd_0 == -1)
       {
           printf("Can't open /dev/mem.\n");
           exit(0);
       }
       printf("/dev/mem opened.\n");
       // Map one page of memory into user space such that the device is in that page, but it may not
       // be at the start of the page.
       mapped_base_0 = mmap(0, DDR_MAP_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, memfd_0, dev_base_0 & ~DDR_MAP_MASK);
       if (mapped_base_0 == (void *) -1)
       {
           printf("Can't map the memory to user space.\n");
           exit(0);
       }
       printf("Memory mapped at address %p.\n", mapped_base_0);
        // get the address of the device in user space which will be an offset from the base
        // that was mapped as memory is mapped at the start of a page
        mapped_dev_base_0 = mapped_base_0 + (dev_base_0 & DDR_MAP_MASK);

        /*======================================================================================
        STEP 9 : Copy the Data from DDR Memory location 0x20000000 to Destination Buffer
        ========================================================================================*/
        memcpy(DestArray, mapped_dev_base_0, (OUTPUT_BYTESIZE* sizeof(dtype)));

        printf("Dest set to: %lu\n", *((unsigned long *) mapped_dev_base_0));
        /*======================================================================================
        STEP 10 : Un-map the Kernel memory from the User layer.
        ========================================================================================*/
        if (munmap(mapped_base_0, DDR_MAP_SIZE) == -1)
        {
            printf("Can't unmap memory from user space.\n");
            exit(0);
        }

       close(memfd_0);


       /*======================================================================================
        STEP 11 : Compare Source Buffer with Destination Buffer.
       ========================================================================================*/
       /*for (Index = 0; Index < (BUFFER_BYTESIZE/4); Index++)
       {
           if (SrcArray[Index] != DestArray[Index])
           {
                   printf("Error in the Data comparison \n \r");
                   return 1;
           }
       }*/
       int err_cnt = check_err(SrcArray0, SrcArray1, DestArray);
       if (err_cnt == 0)
           printf("Result verification is Successful \n\r");
       else
           printf("There are %d different!\n\r", err_cnt);

    return 0;
}

