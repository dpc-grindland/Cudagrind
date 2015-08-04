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
 * Copy between two pointers on devices that support unified addressing.
 *
 ******************************************************************************/
#include <cudaWrap.h>

//
// Note: We do not currently support unified addressing, so what we do is sensible guessing.
//       First we check if the pointers are registered in our device memory list, 
//       then if they are valid host memory and if neither is true we output an Error.
//
// TODO: Make the function unified addressing aware.
// TODO: Properly test for non-unified cases.
CUresult I_WRAP_SONAME_FNNAME_ZZ(libcudaZdsoZa, cuMemcpy)(CUdeviceptr dst, CUdeviceptr src, size_t ByteCount) {
   OrigFn fn;
   CUresult result;
   CUcontext ctx = NULL;
   
   long vgErrorAddressDst, vgErrorAddressSrc;
   cgCtxListType *nodeCtx;
   cgMemListType *nodeMemDst, *nodeMemSrc;
   size_t dstSize = 0, srcSize = 0;
   
   VALGRIND_GET_ORIG_FN(fn);
   cgLock();
   CALL_FN_W_WWW(result, fn, dst, src, ByteCount);
   
   // Get current context
   cgGetCtx(&ctx);
   nodeCtx = cgFindCtx(ctx);
   
   nodeMemDst = cgFindMem(nodeCtx, dst);
   nodeMemSrc = cgFindMem(nodeCtx, src);
   
   // Get definedness status for both pointers
   vgErrorAddressDst = VALGRIND_CHECK_MEM_IS_DEFINED(&dst, sizeof(CUdeviceptr));
   vgErrorAddressSrc = VALGRIND_CHECK_MEM_IS_DEFINED(&src, sizeof(CUdeviceptr));
   
   // Are the pointers defined? -> Only after call to cuMemAlloc/malloc/..
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
      VALGRIND_PRINTF_BACKTRACE(" pointer in cuMemcpy undefined.\n");
   } else if (dst == 0 || src == 0) { // Check for 0 pointer -> cuMemFree'd/free'd
      VALGRIND_PRINTF("Error: NULL ");
      if (dst == 0) {
         VALGRIND_PRINTF(" destination");
         if (src == 0) {
            VALGRIND_PRINTF(" and");
         }
      }
      if (src == 0) {
         VALGRIND_PRINTF(" source");
      }
      VALGRIND_PRINTF_BACKTRACE(" pointer in cuMemcpy.\n");
   } else { // Check if allocated src/dst memory is big enough
      nodeCtx = cgFindCtx(ctx);
      nodeMemDst = cgFindMem(nodeCtx, dst);
      nodeMemSrc = cgFindMem(nodeCtx, src);
      
      // Test if either memory might be a host pointer (if cgFindMem returns NULL).
      // False positive if the variable actually is a not allocated device pointer.
      if (!nodeMemDst) {
         vgErrorAddressDst = VALGRIND_CHECK_MEM_IS_ADDRESSABLE(dst, ByteCount);
         if (vgErrorAddressDst) {
            VALGRIND_PRINTF("Error: Destination memory during call to cuMemcpy is not allocated.\n");
            VALGRIND_PRINTF("       Expected %lu allocated bytes but only found %lu.", ByteCount, vgErrorAddressDst - (long)dst);
         } 
      } else {
         dstSize = nodeMemDst->size - (dst - nodeMemDst->dptr);
         if (dstSize < ByteCount) { // Check if remaining memory is big enough for the to be copied data
            VALGRIND_PRINTF("Error: Allocated destination memory too small in call to cuMemcpy.\n");
            VALGRIND_PRINTF("       Expected %lu allocated bytes but only found %lu.", ByteCount, dstSize);
         }
      }
      // Same for the source pointer
      if (!nodeMemSrc) {
         vgErrorAddressSrc = VALGRIND_CHECK_MEM_IS_ADDRESSABLE(src, ByteCount);
         if (vgErrorAddressSrc) {
            VALGRIND_PRINTF("Error: Source memory during call to cuMemcpy is not allocated.\n");
            VALGRIND_PRINTF("       Expected %lu allocated bytes but only found %lu.", ByteCount, vgErrorAddressSrc - (long)dst);
         } 
      } else {
         srcSize = nodeMemSrc->size - (src - nodeMemSrc->dptr);
         if (srcSize < ByteCount) {
            VALGRIND_PRINTF("Error: Allocated source memory too small in call to cuMemcpy.\n");
            VALGRIND_PRINTF("       Expected %lu allocated bytes but only found %lu.", ByteCount, srcSize);
         }
      }
      if (!nodeMemDst || dstSize < ByteCount || vgErrorAddressDst || !nodeMemSrc || srcSize < ByteCount || vgErrorAddressSrc) {
         VALGRIND_PRINTF_BACKTRACE("\n");
      }
   }
   
   cgUnlock();
   return result;
}
