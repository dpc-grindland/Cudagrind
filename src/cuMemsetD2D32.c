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
 * Wrapper for cuMemsetD2D32
 *
 ******************************************************************************/
#include <cudaWrap.h>

CUresult I_WRAP_SONAME_FNNAME_ZZ(libcudaZdsoZa, cuMemsetD2D32)(CUdeviceptr dstDevice, size_t dstPitch, unsigned int ui, size_t Width, size_t Height) {
   OrigFn fn;
   CUresult result;
   CUcontext ctx = NULL;
   cgMemListType *nodeMemDst;
   
   int error = 0;
   long vgErrorAddress;
   size_t dstSize;

   VALGRIND_GET_ORIG_FN(fn);
   cgLock();
   CALL_FN_W_5W(result, fn, dstDevice, dstPitch, ui, Width, Height);
   
   // Check if function parameters are defined.
   // TODO: Warning or error in case of a partially undefined ui?
   vgErrorAddress = VALGRIND_CHECK_MEM_IS_DEFINED(&dstDevice, sizeof(CUdeviceptr));
   if (vgErrorAddress) {
      error++;
      VALGRIND_PRINTF("Error: 'dstDevice' in call to cuMemsetD2D32 not defined.\n");
   }
   vgErrorAddress = VALGRIND_CHECK_MEM_IS_DEFINED(&dstPitch, sizeof(size_t));
   if (vgErrorAddress) {
      error++;
      VALGRIND_PRINTF("Error: 'dstPitch' in call to cuMemsetD2D32 not defined.\n");
   }
   vgErrorAddress = VALGRIND_CHECK_MEM_IS_DEFINED(&ui, sizeof(ui));
   if (vgErrorAddress) {
      error++;
      VALGRIND_PRINTF("Warning: 'ui' in call to cuMemsetD2D32 is not fully defined.\n");
   }
   vgErrorAddress = VALGRIND_CHECK_MEM_IS_DEFINED(&Width, sizeof(size_t));
   if (vgErrorAddress) {
      error++;
      VALGRIND_PRINTF("Error: 'Width' in call to cuMemsetD2D32 not defined.\n");
   }
   vgErrorAddress = VALGRIND_CHECK_MEM_IS_DEFINED(&Height, sizeof(size_t));
   if (vgErrorAddress) {
      error++;
      VALGRIND_PRINTF("Error: 'Height' in call to cuMemsetD2D32 not defined.\n");
   }
   
   // Fetch current context
   cgGetCtx(&ctx);
   nodeMemDst = cgFindMem(cgFindCtx(ctx), dstDevice);
   
   // Check if memory has been allocated
   if (!nodeMemDst) {
      error++;
      VALGRIND_PRINTF("Error: Destination device memory not allocated in call to cuMemsetD2D32.\n");
   } else {
      // If memory is allocated, check size of available memory
      dstSize = nodeMemDst->size - (dstDevice - nodeMemDst->dptr);
      // The whole memory block of dstPitch*Height must fit into memory
      if (dstSize < sizeof(ui) * dstPitch * Height) {
         error++;
         VALGRIND_PRINTF("Error: Allocated device memory too small in call to cuMemsetD2D32.\n"
                         "       Expected %lu allocated bytes but only found %lu.\n",
                         sizeof(ui) * dstPitch * Height, dstSize);
      }
      
      // Check if dstDevice and dstPitch are both properly aligned
      // TODO: Is this a valid check? (see also cuMemsetD32)
      if (dstDevice % 4) {
         error++;
         VALGRIND_PRINTF("Error: Pointer dstDevice in call to cuMemsetD2D32 not four byte aligned.\n");
      }
      if (dstPitch % 4) {
         error++;
         VALGRIND_PRINTF("Error: Destination pitch in call to cuMemsetD2D32 not four byte aligned.\n");
      }
   }
   
   // Make sure pitch is big enough to accommodate asked for Width
   if (dstPitch < Width) {
      error++;
      VALGRIND_PRINTF("Error: dstPitch smaller than Width in call to cuMemsetD2D32.\n");
   }
   
   if (error) {
      VALGRIND_PRINTF_BACKTRACE("");
   }
   
   cgUnlock();
   return result;
}