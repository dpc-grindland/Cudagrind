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
 * Wrapper for cuMemsetD16
 *
 ******************************************************************************/
#include <cudaWrap.h>

CUresult I_WRAP_SONAME_FNNAME_ZZ(libcudaZdsoZa, cuMemsetD16)(CUdeviceptr dstDevice, unsigned short us, size_t N) {
   OrigFn fn;
   CUresult result;
   CUcontext ctx = NULL;
   cgMemListType *nodeMemDst;
   
   int error = 0;
   long vgErrorAddress;
   size_t dstSize;

   VALGRIND_GET_ORIG_FN(fn);
   cgLock();
   CALL_FN_W_WWW(result, fn, dstDevice, us, N);
   
   // Check if function parameters are defined.
   // TODO: Warning or error in case of a partially undefined us?
   vgErrorAddress = VALGRIND_CHECK_MEM_IS_DEFINED(&dstDevice, sizeof(CUdeviceptr));
   if (vgErrorAddress) {
      error++;
      VALGRIND_PRINTF("Error: 'dstDevice' in call to cuMemsetD16 not defined.\n");
   }
   vgErrorAddress = VALGRIND_CHECK_MEM_IS_DEFINED(&us, sizeof(us));
   if (vgErrorAddress) {
      error++;
      VALGRIND_PRINTF("Warning: 'us' in call to cuMemsetD16 is not fully defined.\n");
   }
   vgErrorAddress = VALGRIND_CHECK_MEM_IS_DEFINED(&N, sizeof(size_t));
   if (vgErrorAddress) {
      error++;
      VALGRIND_PRINTF("Error: 'N' in call to cuMemsetD16 not defined.\n");
   }
   
   
   // Fetch current context
   cgGetCtx(&ctx);
   nodeMemDst = cgFindMem(cgFindCtx(ctx), dstDevice);
   
   // Check if memory has been allocated
   if (!nodeMemDst) {
      error++;
      VALGRIND_PRINTF("Error: Destination device memory not allocated in call to cuMemsetD16.\n");
   } else {
      // If memory is allocated, check size of available memory
      dstSize = nodeMemDst->size - (dstDevice - nodeMemDst->dptr);
      if (dstSize < sizeof(unsigned short) * N) {
         error++;
         VALGRIND_PRINTF("Error: Allocated device memory too small in call to cuMemsetD16.\n"
                         "       Expected %lu allocated bytes but only found %lu.\n",
                         sizeof(unsigned short) * N, dstSize);
      }
      
      // Check if pointer is properly two byte aligned.
      // TODO: Is this a valid check?
      if (dstDevice % 2) {
         error++;
         VALGRIND_PRINTF("Error: Pointer dstDevice in call to cuMemsetD16 not two byte aligned.\n");
      }
   }
   
   if (error) {
      VALGRIND_PRINTF_BACKTRACE("");
   }
   
   cgUnlock();
   return result;
}