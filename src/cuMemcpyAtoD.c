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
 * Wrapper for cuMemcpyAtoD, which copies from 1D Cuda Array to device memory.
 *
 * If being called with 2D/3D source array a
 *    warning is printed if Width is big enough to contain offset+bytecount
 *    error   otherwise
 * While the standard does not mention the first case it still does work.
 *
 ******************************************************************************/
#include <cudaWrap.h>

CUresult I_WRAP_SONAME_FNNAME_ZZ(libcudaZdsoZa, cuMemcpyAtoD)(CUdeviceptr dstDevice, CUarray srcArray, size_t srcOffset, size_t ByteCount) {
   OrigFn fn;
   CUresult result;
   CUcontext ctx = NULL;
   cgCtxListType *nodeCtx;
   cgArrListType *nodeArrSrc;
   cgMemListType *nodeMemDst;
   
   long vgErrorAddress;
   int error = 0;
   
   VALGRIND_GET_ORIG_FN(fn);
   cgLock();
   CALL_FN_W_WWWW(result, fn, dstDevice, srcArray, srcOffset, ByteCount);
   
   // Check if actual function parameters are defined
   if (VALGRIND_CHECK_MEM_IS_DEFINED(&dstDevice, sizeof(CUdeviceptr))) {
      error++;
      VALGRIND_PRINTF("Error: dstDevice in call to cuMemcpyAtoD is not defined.\n");
   } else if (!dstDevice) {
      error++;
      VALGRIND_PRINTF("Error: dstDevice in call to cuMemcpyAtoD is NULL.\n");
   }
   if (VALGRIND_CHECK_MEM_IS_DEFINED(&srcArray, sizeof(CUarray))) {
      error++;
      VALGRIND_PRINTF("Error: srcArray in call to cuMemcpyAtoD is not defined.\n");
   } else if (!srcArray) {
      error++;
      VALGRIND_PRINTF("Error: srcArray in call to cuMemcpyAtoD is NULL");
   }
   if (VALGRIND_CHECK_MEM_IS_DEFINED(&srcOffset, sizeof(size_t))) {
      error++;
      VALGRIND_PRINTF("Error: srcOffset in call to cuMemcpyAtoD is not defined.\n");
   }
   if (VALGRIND_CHECK_MEM_IS_DEFINED(&ByteCount, sizeof(size_t))) {
      error++;
      VALGRIND_PRINTF("Error: ByteCount in call to cuMemcpyAtoD is not defined.\n");
   }
   
   cgGetCtx(&ctx);
   
   nodeCtx = cgFindCtx(ctx);
   
   nodeArrSrc = cgFindArr(nodeCtx, srcArray);
   nodeMemDst = cgFindMem(nodeCtx, dstDevice);
   
   if (nodeArrSrc) {
      // Check if array is 1-dimensional or big enough in first dimension
      if (nodeArrSrc->desc.Height > 1 || nodeArrSrc->desc.Depth > 1) {
         if (nodeArrSrc->desc.Width - srcOffset < ByteCount) {
            error++;
            VALGRIND_PRINTF("Error: Source array in cuMemcpyAtoD is 2-dimensional\n"
                            "       and ByteCount bigger than available width in first dimension.\n");
         } else {
            VALGRIND_PRINTF("Warning: Source array in cuMemcpyAtoD is 2-dimensional.\n");
         }
      } else if (nodeArrSrc->desc.Width - srcOffset < ByteCount) { 
         // If array is 1D, check size.
         VALGRIND_PRINTF("Error: Source array in cuMemcpyAtoD is too small.\n"
                         "       Expected %l bytes but only found %l.\n", 
                         ByteCount, nodeArrSrc->desc.Width - srcOffset);
         error++;
      }
      if ( ByteCount % cgArrDescBytesPerElement(&(nodeArrSrc->desc)) ) {
         error++;
         VALGRIND_PRINTF("Error: ByteCount not evenly divisible by source array element size.\n");
      }
   } else {
      error++;
      VALGRIND_PRINTF("Error: Source array not allocated in call to cuMemcpyAtoD.\n");
   }
   
   if (nodeMemDst) {
      // Check if allocated memory is big enough for copy operation
      if (nodeMemDst->dptr + nodeMemDst->size < dstDevice + ByteCount) {
         error++;
         VALGRIND_PRINTF("Error: Destination device memory in cuMemcpyAtoD is too small.\n"
                         "       Expeceted %l bytes but only found %l.\n",
                         ByteCount, nodeMemDst->size - (nodeMemDst->dptr - dstDevice));
      }
   } else {
      error++;
      VALGRIND_PRINTF("Error: Destination device memory not allocated in call to cuMemcpyAtoD.\n");
   }
   
   cgUnlock();
   return result;
}