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
 * Wrapper for cuMemAllocPitch.
 *
 * Performs several checks and adds the allocated memory to the internal list.
 *
 ******************************************************************************/
#include <cudaWrap.h>

CUresult I_WRAP_SONAME_FNNAME_ZZ(libcudaZdsoZa, cuMemAllocPitch)(CUdeviceptr *dptr, size_t *pPitch, size_t WidthInBytes, size_t Height, unsigned int ElementSizeBytes) {
   OrigFn fn;
   CUresult result, res;
   CUcontext  ctx = NULL;
   size_t bytesize;
   int error;
   long vgErrorAddress;
   
   // Fetch and call original function
   VALGRIND_GET_ORIG_FN(fn);
   cgLock();
   CALL_FN_W_5W(result, fn, dptr, pPitch, WidthInBytes, Height, ElementSizeBytes);
   
   // Check if function parameters are properly allocated/defined
   vgErrorAddress = VALGRIND_CHECK_MEM_IS_ADDRESSABLE(dptr, sizeof(CUdeviceptr));
   if (vgErrorAddress) {
      error++;
      VALGRIND_PRINTF("Error: 'dptr' points unallocated memory in cuMemAllocPitch.\n");
   }
   vgErrorAddress = VALGRIND_CHECK_MEM_IS_ADDRESSABLE(pPitch, sizeof(size_t));
   if (vgErrorAddress) {
      error++;
      VALGRIND_PRINTF("Error: 'pPitch' points to unallocated memory in cuMemAllocPitch.\n ");
   }
   vgErrorAddress = VALGRIND_CHECK_MEM_IS_DEFINED(&WidthInBytes, sizeof(size_t));
   if (vgErrorAddress) {
      error++;
      VALGRIND_PRINTF("Error: 'WidthInBytes' undefined in cuMemAllocPitch.\n ");
   } else {
      if (WidthInBytes <= 0) {
         error++;
         VALGRIND_PRINTF("Error: 'WidthInBytes' is %lu instead of positive in cuMemAllocPitch.\n");
      }
   }
   vgErrorAddress = VALGRIND_CHECK_MEM_IS_DEFINED(&Height, sizeof(size_t));
   if (vgErrorAddress) {
      error++;
      VALGRIND_PRINTF("Error: 'Height' undefined in cuMemAllocPitch.\n ");
   } else {
      if (Height <= 0) {
         error++;
         VALGRIND_PRINTF("Error: 'Height' is %lu instead of positive in cuMemAllocPitch.\n");
      }
   }
   vgErrorAddress = VALGRIND_CHECK_MEM_IS_DEFINED(&ElementSizeBytes, sizeof(unsigned int));
   if (vgErrorAddress) {
      error++;
      VALGRIND_PRINTF("Error: 'ElementSizeBytes' undefined in cuMemAllocPitch.\n ");
   }
   
   
   
   // ElementSizeBytes must be 4,8,16
   // TODO: True for CUDA >= 5.0?
   switch (ElementSizeBytes) {
      case 4:
      case 8:
      case 16:
         break;
      default: {
         error++;
         VALGRIND_PRINTF("Error: ElementSizeBytes in cuMemAllocPitch must be 4, 8 or 16.\n");
      }
   }
   
   // If the allocation was successful we add it to the list of known locations
   if (result == CUDA_SUCCESS) {
      // Determine context of current thread ..
      cgGetCtx(&ctx);
      // .. calculate actual size of allocated memory ..
      bytesize = *pPitch * Height;
      // .. and add the freshly allocated memory to the list.
      cgAddMem(ctx, *dptr, bytesize);
   } else {
      VALGRIND_PRINTF_BACKTRACE("cuMemAllocPitch returned with error code: %d\n", result);
   }

   cgUnlock();
   return result;
}