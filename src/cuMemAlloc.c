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
 * Wrapper for cuMemAlloc that adds the allocated memory to the internal list.
 *
 ******************************************************************************/
#include <cudaWrap.h>

CUresult I_WRAP_SONAME_FNNAME_ZZ(libcudaZdsoZa, cuMemAlloc)(CUdeviceptr *dptr, size_t bytesize) {
   OrigFn fn;
   CUresult result, res;
   CUcontext  ctx = NULL;
   
   VALGRIND_GET_ORIG_FN(fn);
   cgLock();
   CALL_FN_W_WW(result, fn, dptr, bytesize);
   
   // If the allocation was successful we add it to the list of known locations
   if (result == CUDA_SUCCESS) {
      // Determine context of current thread ..
      cgGetCtx(&ctx);
      // .. and add the freshly allocated memory to the list.
      cgAddMem(ctx, *dptr, bytesize);
   } else {
      VALGRIND_PRINTF_BACKTRACE("cuMemAlloc returned with error code: %d\n", result);
   }

   cgUnlock();
   return result;
}