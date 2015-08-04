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
 * Wrapper for cuMemcpy2DAsync
 *
 * Instead of providing an actual wrapper we simply call the synchronous
 *  version of the function (and hence implicitly it's wrapper).
 *
 ******************************************************************************/
#include <cudaWrap.h>

CUresult I_WRAP_SONAME_FNNAME_ZZ(libcudaZdsoZa, cuMemcpy2DAsync)(const CUDA_MEMCPY2D *pCopy, CUstream hStream) {
   int error = 0;
   long vgErrorAddress;
   
   vgErrorAddress = VALGRIND_CHECK_MEM_IS_DEFINED(&hStream, sizeof(CUstream));
   if (vgErrorAddress) {
      error++;
      VALGRIND_PRINTF("Error: 'hStream' in call to cuMemcpy2DAsync not defined.\n");
   }
   
   cgLock();
   
   CUcontext      ctx = NULL;
   cgGetCtx(&ctx);
   
   // Check if destination (device) memory/array is already being written to.
   switch (pCopy->dstMemoryType) {
      case CU_MEMORYTYPE_DEVICE: {
         cgMemListType  *nodeMem;

         nodeMem = cgFindMem(cgFindCtx(ctx), pCopy->dstDevice);
      
         if (nodeMem) {
            // Are we trying to read a memory region that's being written by diffrent stream?
            if (nodeMem->locked & 2 && nodeMem->stream != hStream) {
               error++;
               VALGRIND_PRINTF("Error: Concurrent write and read access by different streams.\n");
            }
            
            nodeMem->locked = nodeMem->locked | 1;
            nodeMem->stream = hStream;
         }
      
         break;
      }
      case CU_MEMORYTYPE_ARRAY: {
         cgArrListType  *nodeArr;
      
         nodeArr = cgFindArr(cgFindCtx(ctx), pCopy->dstArray);
      
         if (nodeArr) {
            // Are we trying to read an array that's being written by different stream?
            if (nodeArr->locked & 2 && nodeArr->stream != hStream) {
               error++;
               VALGRIND_PRINTF("Error: Concurrent write and read access to array by different streams.\n");
            }
            
            nodeArr->locked = nodeArr->locked | 1;
            nodeArr->stream = hStream;
         }
         
         break;
      }
   }
   
   // Check if source (device) memory/array is already being written to/read from.
   switch (pCopy->srcMemoryType) {
      case CU_MEMORYTYPE_DEVICE: {
         cgMemListType  *nodeMem;

         nodeMem = cgFindMem(cgFindCtx(ctx), pCopy->srcDevice);
      
         if (nodeMem) {
            // Are we trying to read a memory region that's being written by diffrent stream?
            if (nodeMem->locked && nodeMem->stream != hStream) {
               error++;
               VALGRIND_PRINTF("Error: Concurrent write and read access by different streams.\n");
            }
            
            nodeMem->locked = nodeMem->locked | 2;
            nodeMem->stream = hStream;
         }
      
         break;
      }
      case CU_MEMORYTYPE_ARRAY: {
         cgArrListType  *nodeArr;
      
         nodeArr = cgFindArr(cgFindCtx(ctx), pCopy->srcArray);
      
         if (nodeArr) {
            // Are we trying to read an array that's being written by different stream?
            if (nodeArr->locked && nodeArr->stream != hStream) {
               error++;
               VALGRIND_PRINTF("Error: Concurrent write and read access to array by different streams.\n");
            }
            
            nodeArr->locked = nodeArr->locked | 2;
            nodeArr->stream = hStream;
         }
         
         break;
      }
   }   

   cgUnlock();
   
   if (error) {
      VALGRIND_PRINTF_BACKTRACE("");
   }
   
   return cuMemcpy2D(pCopy);
}