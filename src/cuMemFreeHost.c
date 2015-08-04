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
 * Wrapper for cuMemFreeHost.
 *
 ******************************************************************************/
#include <cudaWrap.h>

// We need to remove the device pointer from the list if the memory had been allocated with the DEVICEMAP flag.
CUresult I_WRAP_SONAME_FNNAME_ZZ(libcudaZdsoZa, cuMemFreeHost)(void *pp) {
   OrigFn fn;
   CUresult result;
   CUdeviceptr dptr;
   CUcontext ctx = NULL;
   
   VALGRIND_GET_ORIG_FN(fn);
   cgLock();
   
   // Only remove the pointer from the list if it has been enabled previously
   if (cuMemHostGetDevicePointer(&dptr, pp, 0) == CUDA_SUCCESS) {
      cgGetCtx(&ctx);
      cgDelMem(ctx, dptr);
   }
   
   CALL_FN_W_W(result, fn, pp);
   cgUnlock();
   return result;
}