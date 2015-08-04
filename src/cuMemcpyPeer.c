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
 * Copy device memory between different contexts.
 *
 ******************************************************************************/
#include <cudaWrap.h>

// TODO: Check if contexts are valid, check allocation status of device memory, ???, profit!
CUresult I_WRAP_SONAME_FNNAME_ZZ(libcudaZdsoZa, cuMemcpyPeer)(CUdeviceptr dstDevice, CUcontext dstContext, CUdeviceptr srcDevice, CUcontext srcContext, size_t ByteCount) {
   OrigFn fn;
   CUresult result;
   
   cgCtxListType *nodeCtxDst, *nodeCtxSrc;
   cgMemListType *nodeMemDst, *nodeMemSrc;
   long vgErrorAddressDst, vgErrorAddressSrc;
   long vgErrorAddressDstCtx, vgErrorAddressSrcCtx;
   size_t dstSize, srcSize;
   int error = 0;
   
   VALGRIND_GET_ORIG_FN(fn);
   cgLock();
   CALL_FN_W_5W(result, fn, dstDevice, dstContext, srcDevice, srcContext, ByteCount);
   
   // Get definedness status for both pointers
   vgErrorAddressDst = VALGRIND_CHECK_MEM_IS_DEFINED(&dstDevice, sizeof(CUdeviceptr));
   vgErrorAddressSrc = VALGRIND_CHECK_MEM_IS_DEFINED(&srcDevice, sizeof(CUdeviceptr));
   
   // Check if device pointers themself are defined
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
      VALGRIND_PRINTF(" pointer in cuMemcpyPeer undefined.\n");
      error++;
   }

   // Check if context function parameters are defined
   vgErrorAddressDstCtx = VALGRIND_CHECK_MEM_IS_DEFINED(&dstContext, sizeof(CUcontext));
   vgErrorAddressSrcCtx = VALGRIND_CHECK_MEM_IS_DEFINED(&srcContext, sizeof(CUcontext));

   if (vgErrorAddressDstCtx || vgErrorAddressSrcCtx) {
      VALGRIND_PRINTF("Error:");
      if (vgErrorAddressDstCtx) {
         VALGRIND_PRINTF(" dstContext");
         if (vgErrorAddressSrcCtx) {
            VALGRIND_PRINTF(" and");
         }
      }
      if (vgErrorAddressSrcCtx) {
         VALGRIND_PRINTF(" srcContext");
      }
      VALGRIND_PRINTF(" in cuMemcpyPeer undefined.\n");
      error++;
   }
   
   // If defined check whether pointers contain NULL
   if (!vgErrorAddressDst && !dstDevice) {
      VALGRIND_PRINTF("Error: NULL destination device pointer in call to cuMemcpyPeer.\n");
      error++;
   }
   if (!vgErrorAddressSrc && !srcDevice) {
      VALGRIND_PRINTF("Error: NULL source device pointer in call to cuMemcpyPeer.\n");
      error++;
   }
   
   // Fetch noes for context and memory from internal list to check allocation status of destination device memory
   if (!vgErrorAddressDst && !vgErrorAddressDstCtx) {
      nodeCtxDst = cgFindCtx(dstContext);
      nodeMemDst = cgFindMem(nodeCtxDst, dstDevice);
      
      if (nodeMemDst) {
         dstSize = nodeMemDst->size - (dstDevice - nodeMemDst->dptr);
         if (dstSize < ByteCount) {
            VALGRIND_PRINTF("Error: Allocated destination memory is too small in call to cuMemcpyPeer.\n"
                            "       Expected %lu allocated bytes but only found %lu.\n",
                            ByteCount, dstSize);
            error++;
         }
      } else {
         VALGRIND_PRINTF("Error: Destination device memory is not allocated in call to cuMemcpyPeer.\n");
         error++;
      }
   }
   
   // Fetch noes for context and memory from internal list to check allocation status of source device memory
   if (!vgErrorAddressSrc && !vgErrorAddressSrcCtx) {
      nodeCtxSrc = cgFindCtx(srcContext);
      nodeMemSrc = cgFindMem(nodeCtxSrc, srcDevice);
      
            
      if (nodeMemSrc) {
         srcSize = nodeMemSrc->size - (srcDevice - nodeMemSrc->dptr);
         if (srcSize < ByteCount) {
            VALGRIND_PRINTF("Error: Allocated source memory is too small in call to cuMemcpyPeer.\n"
                            "       Expected %lu allocated bytes but only found %lu.\n",
                            ByteCount, srcSize);
            error++;
         }
      } else {
         VALGRIND_PRINTF("Error: Source device memory is not allocated in call to cuMemcpyPeer.\n");
         error++;
      }
   }
   
   // Print the backtrace if we encountered any error
   if (error) {
      VALGRIND_PRINTF_BACKTRACE("");
   }
   
   cgUnlock();
   return result;
}