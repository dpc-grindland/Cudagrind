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
 * Wrapper for cuMemHostUnregister that removes registered memory from list.
 *
 ******************************************************************************/
#include <cudaWrap.h>

// Removes the corresponding device pointer from the device memory list
CUresult I_WRAP_SONAME_FNNAME_ZZ(libcudaZdsoZa, cuMemHostUnregister)(void *p) {
   OrigFn fn;
   CUresult result;
   long vgErrorAddress;
   CUcontext ctx = NULL;
   CUdeviceptr dptr;
   
   VALGRIND_GET_ORIG_FN(fn);
   cgLock();
   
   // The pointer should still point to a valid memory region
   vgErrorAddress = VALGRIND_CHECK_MEM_IS_DEFINED(&p, sizeof(void*));
   if (vgErrorAddress) {
      VALGRIND_PRINTF_BACKTRACE("Error: Host pointer not defined in cuMemHostUnregister.\n");
   // TODO: We need the size to do a definedness check -> Fetch from memory list?
   //} else if (vgErrorAddress = VALGRIND_CHECK_MEM_IS_ADDRESSABLE(p, bytesize)) {
   //   VALGRIND_PRINTF_BACKTRACE("Error: Memory not allocated in call to cuMemHostRegister.\n")
   } else {
      // If the pointer ist valid fetch the corresponding device pointer and remove it from the list if it still exists
      if (cuMemHostGetDevicePointer(&dptr, p, 0) == CUDA_SUCCESS) {
         cgGetCtx(&ctx);
         cgDelMem(ctx, dptr);
      }
   }
   
   CALL_FN_W_W(result, fn, p);

   
   cgUnlock();
   return result;
}