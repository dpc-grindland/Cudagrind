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
 * Wrapper for cuMemcpyAtoA, which copies from 1D Cuda Array to 1D Cuda Array.
 *
 * If being called with 2D arrays a
 *    warning is printed if Width is big enough to contain offset+bytecount
 *    error   otherwise
 * While the standard does not mention the first case it still does work.
 *
 ******************************************************************************/
#include <cudaWrap.h>

CUresult I_WRAP_SONAME_FNNAME_ZZ(libcudaZdsoZa, cuMemcpyAtoA)(CUarray dstArray, size_t dstOffset, CUarray srcArray, size_t srcOffset, size_t ByteCount) {
   OrigFn fn;
   CUresult result;
   CUcontext ctx = NULL;
   cgCtxListType *nodeCtx;
   cgArrListType *nodeArrDst, *nodeArrSrc;
   int error = 0;
   
   VALGRIND_GET_ORIG_FN(fn);
   cgLock();
   
   cgGetCtx(&ctx);
   
   nodeCtx = cgFindCtx(ctx);
   
   nodeArrDst = cgFindArr(nodeCtx, dstArray);
   nodeArrSrc = cgFindArr(nodeCtx, srcArray);
   
   // First check if either pointer is NULL
   if (!dstArray || !srcArray) {
      VALGRIND_PRINTF("Error: ");
      if (!nodeArrDst) {
         VALGRIND_PRINTF("destination ");
         if (!nodeArrSrc) {
            VALGRIND_PRINTF("and ");
         }
      }
      if (!nodeArrSrc) {
         VALGRIND_PRINTF("source ");
      }
      VALGRIND_PRINTF("array in cuMemcpyAtoA is NULL.\n.");
      error++;
   } 
   // Then whether they have been allocated previously
   if (dstArray && !nodeArrDst || srcArray && !nodeArrSrc) {
      VALGRIND_PRINTF("Error: ");
      if (dstArray && !nodeArrDst) {
         VALGRIND_PRINTF("destination ");
         if (srcArray && !nodeArrSrc) {
            VALGRIND_PRINTF("and ");
         }
      }
      if (srcArray && !nodeArrSrc) {
         VALGRIND_PRINTF("source ");
      }
      VALGRIND_PRINTF("array in cuMemcpyAtoA undefined\n.");
      error++;
   }
   // Checks on destination array
   if (dstArray && nodeArrDst) {
      // Check if destination array is 1-dimensional and if there's enough allocated space for the copy
      if (nodeArrDst->desc.Height > 1 || nodeArrDst->desc.Depth > 1) {
         if (nodeArrDst->desc.Width - dstOffset < ByteCount) {
            VALGRIND_PRINTF("Error: Destination array in cuMemcpyAtoA is 2-dimensional\n"
                            "       and ByteCount bigger than available width in first dimension.\n");
            error++;
         } else {
            VALGRIND_PRINTF("Warning: Destination array in cuMemcpyAtoA is 2-dimensional.\n");
         }
      } else {
         if (nodeArrDst->desc.Width - dstOffset < ByteCount) {
            VALGRIND_PRINTF("Error: Destination array in cuMemcpyAtoA is too small.\n"
                            "       Expected %l bytes but only found %l.\n", 
                            ByteCount, nodeArrDst->desc.Width - dstOffset);
            error++;
         }
      }
   }
   // Checks on source array
   if (srcArray && nodeArrSrc) {
      // Check if source array is 1-dimensional and if there's enough allocated space for the copy
      if (nodeArrSrc->desc.Height > 1 || nodeArrSrc->desc.Depth > 1) {
         if (nodeArrSrc->desc.Width - srcOffset < ByteCount) {
            VALGRIND_PRINTF("Error: Destination array in cuMemcpyAtoA is 2-dimensional\n"
                            "       and ByteCount bigger than available width in first dimension.\n");
            error++;
         } else {
            VALGRIND_PRINTF("Warning: source array in cuMemcpyAtoA is 2-dimensional.\n");
         }
      } else {
         if (nodeArrSrc->desc.Width - srcOffset < ByteCount) {
            VALGRIND_PRINTF("Error: Source array in cuMemcpyAtoA is too small.\n"
                            "       Expected %l bytes but only found %l.\n", 
                            ByteCount, nodeArrSrc->desc.Width - srcOffset);
            error++;
         }
      }
   }
   
   if (error) {
      VALGRIND_PRINTF_BACKTRACE("");
   }
   // TODO: Can we do additional checks?
   
   CALL_FN_W_5W(result, fn, dstArray, dstOffset, srcArray, srcOffset, ByteCount);
   cgUnlock();
   return result;
}