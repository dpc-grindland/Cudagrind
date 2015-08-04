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
 * Author: Thomas Baumann, HLRS <baumann@hlrs.de>
 *
 ******************************************************************************/
 
/******************************************************************************
 *
 * Wrapper for cuMemHostAlloc.
 *
 ******************************************************************************/
#include <cudaWrap.h>

/*
 * Wrapper needed to make sure all registered host memory is tracked correctly.
 * -> Two ways to get that, either with this or with the cuMemHostRegister function. Any more?!
 * Reminder: cudaMallocHost calls this function (without CU_MEMHOSTREGISTER_DEVICEMAP being set (always?))
 */

CUresult I_WRAP_SONAME_FNNAME_ZZ(libcudaZdsoZa, cuMemHostAlloc)(void **pp, size_t bytesize, unsigned int Flags) {
   OrigFn fn;
   CUresult result;
   CUdeviceptr dptr;
   CUcontext ctx = NULL;
   // Number of errors found
   int numError;
   // Actual flags after call to cuMemHostAlloc ..
   unsigned int realFlags;
   
   numError = 0;

   // No sanity checks here, since we are only tracking the devicepointer
   VALGRIND_GET_ORIG_FN(fn);
   cgLock();
   CALL_FN_W_WWW(result, fn, pp, bytesize, Flags);

   // cuMemHostAlloc seems to be free to set certain attributes,
   // e.g. the DEVICEMAP flag is set in Cuda 5.0, even if Flags is 0.
   if (cuMemHostGetFlags(&realFlags, *pp) != CUDA_SUCCESS) {
      numError++;
      VALGRIND_PRINTF("Error fetching host pointer flags in cuMemHostAlloc call.\n");
   }
   
   // Device pointer is only tracked if the proper flag is set
   if (realFlags & CU_MEMHOSTREGISTER_DEVICEMAP) {
      if (cuMemHostGetDevicePointer(&dptr, *pp, 0) != CUDA_SUCCESS) {
         numError++;
         VALGRIND_PRINTF("Error fetching device pointer in cuMemHostAlloc call with DEVICEMAP flag set.\n");
      }
      // Determine context and add device pointer to list
      cgGetCtx(&ctx);
      cgAddMem(ctx, dptr, bytesize);
   }
   
   if (numError) {
      VALGRIND_PRINTF_BACKTRACE("");
   }

   cgUnlock();
   return result;
}
