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
 * Wrapper for cudaMemcpyFromSymbol
 *
 * Symbols are never allocated with cuMemAlloc et al., but are still initialzed
 *  by cuMemcpy through the runtime. So in order to not produce a false 
 *  positive during the copy we add the symbol's device pointer to the memory
 *  list and mark it as a symbol (to avoid reporting false unfreed memory).
 *
 ******************************************************************************/
#include <cudaWrap.h>

cudaError_t I_WRAP_SONAME_FNNAME_ZZ(libcudartZdsoZa, cudaMemcpyToSymbolAsync)(const void *symbol, const void *src, size_t count, size_t offset, enum cudaMemcpyKind kind, cudaStream_t stream) {
   OrigFn      fn;
   cudaError_t result;
   CUcontext   ctx = NULL;
   int         errors = 0;
   
   // Cudagrind internal data structures
   cgMemListType *nodeMemSrc, *nodeMemDst;
   cgCtxListType *nodeCtx;
   
   int error = 0;
   long vgErrorAddress;
   
   VALGRIND_GET_ORIG_FN(fn);
   
   cgLock();
   
   vgErrorAddress = VALGRIND_CHECK_MEM_IS_DEFINED(&stream, sizeof(cudaStream_t));
   if (vgErrorAddress) {
      error++;
      VALGRIND_PRINTF("Error: 'stream' in call to cudaMemcpyToSymbolAsync not defined.\n");
   }
   
   // The temporary entry for the symbol will be filed under the current CUDA context
   cgGetCtx(&ctx);
   // Locate Cudagrind internal node for context
   nodeCtx = cgFindCtx(ctx);
   
   switch (kind) {
      // There's no break after the DtoD case, as there's only additional work
      //  involved in that case (marking the destination memory for being written to).
      case cudaMemcpyDeviceToDevice: {
         nodeMemSrc = cgFindMem(nodeCtx, (CUdeviceptr)src);
         
         if (nodeMemSrc && (nodeMemSrc->locked | CG_STREAM_WRITING) && nodeMemSrc->stream != stream) {
            error++;
            VALGRIND_PRINTF("Error: Concurrent write and read access by different streams.\n");
         }
         
         if (nodeMemSrc) {
            nodeMemSrc->locked = nodeMemSrc->locked | CG_STREAM_READING;
            nodeMemSrc->stream = stream;
         }
         // No 'break;' here on purpose!
      }
      case cudaMemcpyHostToDevice: {
         void     *symbolPtr;
         size_t   symbolSize;
         cudaError_t getPtrError, getSizeError;
      
         getPtrError = cudaGetSymbolAddress(&symbolPtr, symbol);
         getSizeError = cudaGetSymbolSize(&symbolSize, symbol);
      
         // Any other error returned by either GetAddress or GetSize must be from a 
         //  previous, erroneous call to the runtime.
         if (getPtrError == cudaErrorInvalidSymbol || getSizeError == cudaErrorInvalidSymbol) {
            errors++;
            VALGRIND_PRINTF("Error: Invalid symbol in call to cudaMemcpyFromSymbolAsync.\n");
         } else {
            // Get internal node for current context
            nodeCtx = cgFindCtx(ctx);
            // Try to locate symbol memory pointer in list
            nodeMemDst = cgFindMem(nodeCtx, (CUdeviceptr)symbolPtr);
            // Add pointer to list and set isSymbol if this symbol is used for the first time
            if (!nodeMemDst) {
               cgAddMem(ctx, (CUdeviceptr)symbolPtr, symbolSize);
               nodeMemDst = cgFindMem(nodeCtx, (CUdeviceptr)symbolPtr);
               nodeMemDst->isSymbol = 1;
               nodeMemDst->locked = CG_STREAM_WRITING;
               nodeMemDst->stream = (CUstream)stream;
            } else {
               // We only do race condition checks if the memory has not just been added to the list
               if (nodeMemDst && nodeMemDst->locked && nodeMemDst->stream != stream) {
                  error++;
                  VALGRIND_PRINTF("Error: Concurrent write and read access by different streams.\n");
               }

               if (nodeMemDst) {
                  nodeMemDst->locked = nodeMemDst->locked | CG_STREAM_WRITING;
                  nodeMemDst->stream = stream;
               }
            }
            CALL_FN_W_6W(result, fn, symbol, src, count, offset, kind, stream);
         }
         break;
      }
      default: {
         errors++;
         VALGRIND_PRINTF("Error: Invalid kind in call to cudaMemcpyToSymbolAsync.\n");
         VALGRIND_PRINTF("       Expected cudaMemcpyHostToDevice or cudaMemcpyDeviceToDevice.\n");
      }
   }
   
   if (errors) {
      VALGRIND_PRINTF_BACKTRACE("\n");
   }
   
   cgUnlock();
}