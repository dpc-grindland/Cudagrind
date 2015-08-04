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
 * Wraps the cuArrayGetDescriptor function to compare
 * returned errors with the internal list of 2D arrays.
 *
 ******************************************************************************/
#include <cudaWrap.h>

// TODO: Can we do additional checks?
CUresult I_WRAP_SONAME_FNNAME_ZZ(libcudaZdsoZa, cuArrayGetDescriptor)(CUDA_ARRAY_DESCRIPTOR *pArrayDescriptor, CUarray hArray) {
   OrigFn         fn;
   CUresult       result;
   CUcontext      ctx = NULL;
   cgCtxListType  *ctxNode;
   cgArrListType  *node;
   
   VALGRIND_GET_ORIG_FN(fn);
   cgLock();
   CALL_FN_W_WW(result, fn, pArrayDescriptor, hArray);
   
   // Determine context of current thread ..
   cgGetCtx(&ctx);
   // .. locate the respective ctx node ..
   ctxNode = cgFindCtx(ctx);
   // .. and finally locate the array in the context's list of arrays.
   node = cgFindArr(ctxNode, hArray);
   
   if (result == CUDA_SUCCESS && !node) {
      VALGRIND_PRINTF("cuArrayGetDescriptor returned successfully, but array not found\n");
      VALGRIND_PRINTF_BACKTRACE("   in cudagrind's internal list. Reason: Unknown\n");
   } else if (result != CUDA_SUCCESS && node) {
      VALGRIND_PRINTF("cuArrayGetDescriptor returned with error code: %d,\n", result);
      VALGRIND_PRINTF_BACKTRACE("   but array is found in cudagrind's internal list.\n");
   } else if (result != CUDA_SUCCESS) {
      VALGRIND_PRINTF("cuArrayGetDescriptor returned with error code: %d,\n", result);
      VALGRIND_PRINTF_BACKTRACE("   possible reason: Wrong context or array not previously created.\n");
   }
   
   cgUnlock();
   return result;
}