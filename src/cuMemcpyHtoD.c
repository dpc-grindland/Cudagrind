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
 * Wrapper for Host->Device memory copy.
 *
 ******************************************************************************/
#include <cudaWrap.h>

// Copy Host->Device
CUresult I_WRAP_SONAME_FNNAME_ZZ(libcudaZdsoZa, cuMemcpyHtoD)(CUdeviceptr dstDevice, const void *srcHost, size_t ByteCount) {
   OrigFn fn;
   CUresult result;
   CUcontext ctx = NULL;
   cgCtxListType *nodeCtx;
   cgMemListType *nodeMem;
   size_t dstSize;
   long vgErrorAddress, vgErrorAddressDstDevice, vgErrorAddressSrcHost;
   
   VALGRIND_GET_ORIG_FN(fn);
   cgLock();

   vgErrorAddressDstDevice = VALGRIND_CHECK_MEM_IS_DEFINED(&dstDevice, sizeof(void*));
   vgErrorAddressSrcHost   = VALGRIND_CHECK_MEM_IS_DEFINED(&srcHost, sizeof(CUdeviceptr));
   // TODO: Currently errors are exclusive .. i.e. with undefined src and NULL
   //       dst pointer, only the undefined pointer is reported.
   if (vgErrorAddressDstDevice || vgErrorAddressSrcHost) {
      VALGRIND_PRINTF("Error:");
      if (vgErrorAddressDstDevice) {
         VALGRIND_PRINTF(" destination device");
         if (vgErrorAddressSrcHost) {
            VALGRIND_PRINTF(" and");
         }
      }
      if (vgErrorAddressSrcHost) {
         VALGRIND_PRINTF(" source host");
      }
      VALGRIND_PRINTF_BACKTRACE(" pointer in cuMemcpyHtoD not defined.\n");
   } else if (dstDevice != 0 && srcHost != NULL) {
      cgGetCtx(&ctx);
      // Check allocation status and available size on device
      nodeCtx = cgFindCtx(ctx);
      nodeMem = cgFindMem(nodeCtx, dstDevice);
      if (!nodeMem) {
         VALGRIND_PRINTF("Error: Device memory during host->device memory copy is not allocated.");
      } else {
         dstSize = nodeMem->size - (dstDevice - nodeMem->dptr);
         if (dstSize < ByteCount) {
            VALGRIND_PRINTF("Error: Allocated device memory too small for host->device memory copy.\n");
            VALGRIND_PRINTF("       Expected %lu allocated bytes but only found %lu.", ByteCount, dstSize);
         }
      }
      if (!nodeMem || dstSize < ByteCount) {
         VALGRIND_PRINTF_BACKTRACE("\n");
      }
      // Check allocation and definedness for host memory
      vgErrorAddress = VALGRIND_CHECK_MEM_IS_ADDRESSABLE(srcHost, ByteCount);
      if (vgErrorAddress) {
         VALGRIND_PRINTF("Error: Host memory during host->device memory copy is not allocated.\n");
         VALGRIND_PRINTF("       Expected %lu allocated bytes but only found %lu.", ByteCount, vgErrorAddress - (long)srcHost);
      } else {
         vgErrorAddress = VALGRIND_CHECK_MEM_IS_DEFINED(srcHost, ByteCount);
         if (vgErrorAddress) {
            VALGRIND_PRINTF("Error: Host memory during host->device memory copy is not defined.\n");
            VALGRIND_PRINTF("       Expected %lu defined bytes but only found %lu.", ByteCount, vgErrorAddress - (long)srcHost);
         }
      }
      if (vgErrorAddress) {
         VALGRIND_PRINTF_BACKTRACE("\n");
      }
   } else {
      VALGRIND_PRINTF("Error: cuMemcpyHtoD called with NULL");
      if (dstDevice == 0) {
	       VALGRIND_PRINTF(" device");
	       if (srcHost == NULL) VALGRIND_PRINTF(" and");
	   }
	   if (srcHost == NULL) {
	      VALGRIND_PRINTF(" host");
	   }
	   VALGRIND_PRINTF_BACKTRACE(" pointer.\n");
   }
   
   CALL_FN_W_WWW(result, fn, dstDevice, srcHost, ByteCount);
   cgUnlock();
   return result;
}