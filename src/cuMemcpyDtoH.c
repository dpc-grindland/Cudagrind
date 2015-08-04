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
 * Wrapper for Device->Host memory copy.
 *
 ******************************************************************************/
#include <cudaWrap.h>

CUresult I_WRAP_SONAME_FNNAME_ZZ(libcudaZdsoZa, cuMemcpyDtoH)(void *dstHost, CUdeviceptr srcDevice, size_t ByteCount) {
   OrigFn fn;
   CUresult result;
   CUcontext ctx = NULL;
   long vgErrorAddress, vgErrorAddressDstHost, vgErrorAddressSrcDevice;
   cgCtxListType *nodeCtx;
   cgMemListType *nodeMem;
   size_t srcSize;
   
   VALGRIND_GET_ORIG_FN(fn);
   cgLock();
   
   vgErrorAddressDstHost   = VALGRIND_CHECK_MEM_IS_DEFINED(&dstHost, sizeof(void*));
   vgErrorAddressSrcDevice = VALGRIND_CHECK_MEM_IS_DEFINED(&srcDevice, sizeof(CUdeviceptr));
   // TODO: Currently errors are exclusive .. i.e. with undefined src and NULL
   //       host pointer, only the undefined pointer is reported.
   if (vgErrorAddressDstHost || vgErrorAddressSrcDevice) {
      VALGRIND_PRINTF("Error:");
      if (vgErrorAddressDstHost) {
         VALGRIND_PRINTF(" destination host");
         if (vgErrorAddressSrcDevice) {
            VALGRIND_PRINTF(" and");
         }
      }
      if (vgErrorAddressSrcDevice) {
         VALGRIND_PRINTF(" source device");
      }
      VALGRIND_PRINTF_BACKTRACE(" pointer in cuMemcpyDtoH undefined.\n");
   } else if (dstHost != NULL && srcDevice != 0) {
      cgGetCtx(&ctx);
      // Check allocation status and size of host memory
      vgErrorAddress = VALGRIND_CHECK_MEM_IS_ADDRESSABLE(dstHost, ByteCount);
      if (vgErrorAddress) {
         VALGRIND_PRINTF("Error: Host memory during device->host memory copy is not allocated.\n");
         VALGRIND_PRINTF("       Expected %lu allocated bytes but only found %lu.", ByteCount, vgErrorAddress - (long)dstHost);
         VALGRIND_PRINTF_BACKTRACE("\n");
      } 
      // Check allocation status and size of device memory
      nodeCtx = cgFindCtx(ctx);
      nodeMem = cgFindMem(nodeCtx, srcDevice);
      if (!nodeMem) {
         VALGRIND_PRINTF("Error: Device memory during device->host memory copy is not allocated.");
      } else {
         srcSize = nodeMem->size - (srcDevice - nodeMem->dptr);
         if (srcSize < ByteCount) {
            VALGRIND_PRINTF("Error: Allocated device memory too small for device->host memory copy.\n");
            VALGRIND_PRINTF("       Expected %lu allocated bytes but only found %lu.", ByteCount, srcSize);
         }
      }
      if (!nodeMem || srcSize < ByteCount) {
         VALGRIND_PRINTF_BACKTRACE("\n");
      }
   } else {
      VALGRIND_PRINTF("Error: cuMemcpyDtoH called with NULL");
      if (srcDevice == 0) {
	       VALGRIND_PRINTF(" device");
	       if (dstHost == NULL) VALGRIND_PRINTF(" and");
	   }
	   if (dstHost == NULL) {
	      VALGRIND_PRINTF(" host");
	   }
	   VALGRIND_PRINTF_BACKTRACE(" pointer.\n");
   }
   CALL_FN_W_WWW(result, fn, dstHost, srcDevice, ByteCount);
   cgUnlock();
   return result;
}