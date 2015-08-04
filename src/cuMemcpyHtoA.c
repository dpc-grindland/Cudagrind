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
 * Wrapper for cuMemcpyHtoA, which copies from device memory to 1D Cuda Array.
 *
 * If being called with 2D source array a
 *    warning is printed if Width is big enough to contain offset+bytecount
 *    error   otherwise
 * While the standard does not mention the first case it still does work.
 *
 ******************************************************************************/
#include <cudaWrap.h>

CUresult I_WRAP_SONAME_FNNAME_ZZ(libcudaZdsoZa, cuMemcpyHtoA)(CUarray dstArray, size_t dstOffset, const void *srcHost, size_t ByteCount) {
   OrigFn fn;
   CUresult result;
   CUcontext ctx = NULL;
   cgCtxListType *nodeCtx;
   cgArrListType *nodeArrDst;
   cgMemListType *nodeMemSrc;
   
   long vgErrorAddress;
   int error = 0;
   
   VALGRIND_GET_ORIG_FN(fn);
   cgLock();
   CALL_FN_W_WWWW(result, fn, dstArray, dstOffset, srcHost, ByteCount);
   
   // Check if actual function parameters are defined
   if (VALGRIND_CHECK_MEM_IS_DEFINED(&dstArray, sizeof(CUarray))) {
      error++;
      VALGRIND_PRINTF("Error: dstArray in call to cuMemcpyAtoD is not defined.\n");
   } else if (!dstArray) {
      error++;
      VALGRIND_PRINTF("Error: dstArray in call to cuMemcpyAtoD is NULL.\n");
   }
   if (VALGRIND_CHECK_MEM_IS_DEFINED(&dstOffset, sizeof(size_t))) {
      error++;
      VALGRIND_PRINTF("Error: dstOffset in call to cuMemcpyAtoD is not defined.\n");
   }
   if (VALGRIND_CHECK_MEM_IS_DEFINED(&srcHost, sizeof(CUdeviceptr))) {
      error++;
      VALGRIND_PRINTF("Error: srcHost in call to cuMemcpyAtoD is not defined.\n");
   } else if (!srcHost) {
      error++;
      VALGRIND_PRINTF("Error: srcDevice in call to cuMemcpyAtoD is NULL");
   }
   if (VALGRIND_CHECK_MEM_IS_DEFINED(&ByteCount, sizeof(size_t))) {
      error++;
      VALGRIND_PRINTF("Error: ByteCount in call to cuMemcpyAtoD is not defined.\n");
   }
   
   cgGetCtx(&ctx);
   
   nodeCtx = cgFindCtx(ctx);
   
   nodeArrDst = cgFindArr(nodeCtx, dstArray);
   
   if (srcHost) {
      vgErrorAddress = VALGRIND_CHECK_MEM_IS_ADDRESSABLE(srcHost, ByteCount);
      // Check if memory referenced by srcHost has been allocated.
      if (vgErrorAddress) {
         error++;
         VALGRIND_PRINTF("Error: Source host memory in cuMemcpyHtoA is not not allocated.\n"
                         "       Expected %l bytes but only found %l.\n",
                         ByteCount, vgErrorAddress - (long)srcHost);
      } else {
         // If allocated, now check if host memory is defined.
         vgErrorAddress = VALGRIND_CHECK_MEM_IS_DEFINED(srcHost, ByteCount);
         if (vgErrorAddress) {
            error++;
            VALGRIND_PRINTF("Error: Source host memory in cuMemcpyHtoA is not defined.\n"
                            "       Expected %l bytes but only found %l.\n",
                            ByteCount, vgErrorAddress - (long)srcHost);
         }
      }
   }
   
   if (nodeArrDst) {
      // Check if array is 1-dimensional or big enough in first dimension
      if (nodeArrDst->desc.Height > 1 || nodeArrDst->desc.Depth > 1) {
         if (nodeArrDst->desc.Width - dstOffset < ByteCount) {
            error++;
            VALGRIND_PRINTF("Error: Destination array in cuMemcpyAtoD is 2-dimensional\n"
                            "       and ByteCount bigger than available width in first dimension.\n");
         } else {
            VALGRIND_PRINTF("Warning: Destination array in cuMemcpyAtoD is 2-dimensional.\n");
         }
      } else if (nodeArrDst->desc.Width - dstOffset < ByteCount) { 
            // If array is 1D, check size.
            VALGRIND_PRINTF("Error: Destination array in cuMemcpyAtoD is too small.\n"
                            "       Expected %l bytes but only found %l.\n", 
                            ByteCount, nodeArrDst->desc.Width - dstOffset);
            error++;
         }
   } else {
      error++;
      VALGRIND_PRINTF("Error: Destination array not allocated in call to cuMemcpyAtoD.\n");
   }
   
   cgUnlock();
   return result;
}