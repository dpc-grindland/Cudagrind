/******************************************************************************
 *
 * Copyright (C) 2012-2013, HLRS, University of Stuttgart
 *
 * This file is part of Cudagrind.
 *
 * Cudagrind is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.

 * Cudagrind is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with Cudagrind.  If not, see <http://www.gnu.org/licenses/>.
 *
 * Author: Thomas Baumann, HLRS
 *
 ******************************************************************************/
 
/******************************************************************************
 *
 * Wrapper for Device->Device memory copy.
 *
 ******************************************************************************/
#include <cudaWrap.h>

// Only checks the allocation status since we do not know whether the memory has been accessed in kernels.
// TODO: Is is erroneous to copy between 'the same' pointer?
CUresult I_WRAP_SONAME_FNNAME_ZZ(libcudaZdsoZa, cuMemcpyDtoD)(CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t ByteCount) {
   OrigFn fn;
   CUresult result;
   CUcontext ctx = NULL;
   long vgErrorAddressDst, vgErrorAddressSrc;
   cgCtxListType *nodeCtx;
   cgMemListType *nodeMemDst, *nodeMemSrc;
   size_t dstSize, srcSize;
   
   VALGRIND_GET_ORIG_FN(fn);
   cgLock();
   
   // Get definedness status for both pointers
   vgErrorAddressDst = VALGRIND_CHECK_MEM_IS_DEFINED(&dstDevice, sizeof(CUdeviceptr));
   vgErrorAddressSrc = VALGRIND_CHECK_MEM_IS_DEFINED(&srcDevice, sizeof(CUdeviceptr));
   
   // Are the pointers defined? -> Only after call to cuMemAlloc
   if (vgErrorAddressDst || vgErrorAddressSrc) {
      VALGRIND_PRINTF("Error:");
      if (vgErrorAddressDst) {
         VALGRIND_PRINTF(" destination");
         if (vgErrorAddressSrc) {
            VALGRIND_PRINTF(" and");
         }
      }
      if (vgErrorAddressSrc) {
         VALGRIND_PRINTF(" source");
      }
      VALGRIND_PRINTF_BACKTRACE(" device pointer in cuMemcpyDtoD undefined.\n");
   } else if (dstDevice == 0 || srcDevice == 0) { // Check for 0 pointer -> cuMemFree'd
      VALGRIND_PRINTF("Error: NULL ");
      if (dstDevice == 0) {
         VALGRIND_PRINTF(" destination");
         if (srcDevice == 0) {
            VALGRIND_PRINTF(" and");
         }
      }
      if (srcDevice == 0) {
         VALGRIND_PRINTF(" source");
      }
      VALGRIND_PRINTF_BACKTRACE(" pointer in cuMemcpyDtoD.\n");
   } else { // Check if allocated src/dst memory is big enough
      nodeCtx = cgFindCtx(ctx);
      nodeMemDst = cgFindMem(nodeCtx, dstDevice);
      nodeMemSrc = cgFindMem(nodeCtx, srcDevice);
      
      if (!nodeMemDst) {
         VALGRIND_PRINTF("Error: Destination memory during device->device copy is not allocated\n");
      } else {
         dstSize = nodeMemDst->size - (dstDevice - nodeMemDst->dptr);
         if (dstSize < ByteCount) {
            VALGRIND_PRINTF("Error: Allocated destination memory too small for device->device copy.\n");
            VALGRIND_PRINTF("       Expected %lu allocated bytes but only found %lu.", ByteCount, dstSize);
         }
      }
      if (!nodeMemSrc) {
         VALGRIND_PRINTF("Error: Source memory during device->device copy is not allocated\n");
      } else {
         srcSize = nodeMemSrc->size - (srcDevice - nodeMemSrc->dptr);
         if (srcSize < ByteCount) {
            VALGRIND_PRINTF("Error: Allocated source memory too small for device->device copy.\n");
            VALGRIND_PRINTF("       Expected %lu allocated bytes but only found %lu.", ByteCount, srcSize);
         }
      }
      if (!nodeMemDst || dstSize < ByteCount || !nodeMemSrc || srcSize < ByteCount) {
         VALGRIND_PRINTF_BACKTRACE("\n");
      }
   }
   
   CALL_FN_W_WWW(result, fn, dstDevice, srcDevice, ByteCount);
   cgUnlock();
   return result;
}