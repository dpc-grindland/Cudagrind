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
 * Wrapper for cuMemHostRegister that adds registered memory to internal list.
 *
 ******************************************************************************/
#include <cudaWrap.h>

// Checks the host memory and adds the corresponding device pointer to the list if the DEVICEMAP flag ist set
CUresult I_WRAP_SONAME_FNNAME_ZZ(libcudaZdsoZa, cuMemHostRegister)(void *p, size_t bytesize, unsigned int Flags) {
   OrigFn fn;
   CUresult result;
   long vgErrorAddress;
   CUcontext ctx = NULL;
   CUdeviceptr dptr;
   
   VALGRIND_GET_ORIG_FN(fn);
   cgLock();
   
   // Check if pointer is defined and memory actually addressable
   vgErrorAddress = VALGRIND_CHECK_MEM_IS_DEFINED(&p, sizeof(void*));
   if (vgErrorAddress) {
      VALGRIND_PRINTF_BACKTRACE("Error: Host pointer not defined in cuMemHostRegister.\n");
   } else {
      vgErrorAddress = VALGRIND_CHECK_MEM_IS_ADDRESSABLE(p, bytesize);
      if (vgErrorAddress) {
         VALGRIND_PRINTF_BACKTRACE("Error: Memory not allocated in call to cuMemHostRegister.\n");
      }
   }
   
   CALL_FN_W_WWW(result, fn, p, bytesize, Flags);
   
   // Enter device memory into list if the call to cuMemHostRegister was successful
   if ((result == CUDA_SUCCESS) && (Flags | CU_MEMHOSTREGISTER_DEVICEMAP)) {
      // Fetch device pointer to add it to the list
      if (cuMemHostGetDevicePointer(&dptr, p, 0) == CUDA_SUCCESS) {
         cgGetCtx(&ctx);
         cgAddMem(ctx, dptr, bytesize);
      }
   }

   cgUnlock();
   return result;
}