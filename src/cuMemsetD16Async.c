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
 * Wrapper for cuMemsetD16Async
 *
 * Instead of providing an actual wrapper we simply call the synchronous
 *  version of the function (and hence implicitly it's wrapper).
 *
 ******************************************************************************/
#include <cudaWrap.h>

CUresult I_WRAP_SONAME_FNNAME_ZZ(libcudaZdsoZa, cuMemsetD16Async)(CUdeviceptr dstDevice, unsigned short us, size_t N, CUstream hStream) {
   int error = 0;
   long vgErrorAddress;
   
   vgErrorAddress = VALGRIND_CHECK_MEM_IS_DEFINED(&hStream, sizeof(CUstream));
   if (vgErrorAddress) {
      error++;
      VALGRIND_PRINTF("Error: 'hStream' in call to cuMemsetD16Async not defined.\n");
   }
   
   cgLock();
   
   CUcontext ctx = NULL;
   cgCtxListType *nodeCtx;
   cgMemListType *nodeMemDst;
   
   
   // Get current context ..
   cgGetCtx(&ctx);
   nodeCtx = cgFindCtx(ctx);
   
   // .. and locate memory if we are handling device memory
   nodeMemDst = cgFindMem(nodeCtx, dstDevice);
   
   if (nodeMemDst && nodeMemDst->locked & 2 && nodeMemDst->stream != hStream) {
      error++;
      VALGRIND_PRINTF("Error: Concurrent write and read access by different streams.\n");
   }
   
   if (nodeMemDst) {
      nodeMemDst->locked = nodeMemDst->locked | 2;
      nodeMemDst->stream = hStream;
   }
   
   cgUnlock();
   
   if (error) {
      VALGRIND_PRINTF_BACKTRACE("");
   }
   
   return cuMemsetD16(dstDevice, us, N);
}