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

cudaError_t I_WRAP_SONAME_FNNAME_ZZ(libcudartZdsoZa, cudaMemcpyFromSymbol)(void *dst, const void *symbol, size_t count, size_t offset, enum cudaMemcpyKind kind) {
   OrigFn      fn;
   cudaError_t result;
   CUcontext   ctx = NULL;
   int         errors = 0;
   
   VALGRIND_GET_ORIG_FN(fn);
   
   cgLock();
   
   // The temporary entry for the symbol will be filed under the current CUDA context
   cgGetCtx(&ctx);
   
   switch (kind) {
      case cudaMemcpyDeviceToHost:
      case cudaMemcpyDeviceToDevice: {
         void     *symbolPtr;
         size_t   symbolSize;
         cudaError_t getPtrError, getSizeError;
         // Cudagrind internal data structures
         cgMemListType *nodeMem;
         cgCtxListType *nodeCtx;
      
         getPtrError = cudaGetSymbolAddress(&symbolPtr, symbol);
         getSizeError = cudaGetSymbolSize(&symbolSize, symbol);
      
         // Any other error returned by either GetAddress or GetSize must be from a 
         //  previous, erroneous call to the runtime.
         if (getPtrError == cudaErrorInvalidSymbol || getSizeError == cudaErrorInvalidSymbol) {
            errors++;
            VALGRIND_PRINTF("Error: Invalid symbol in call to cudaMemcpyFromSymbol.\n");
         } else {
            // Get internal node for current context
            nodeCtx = cgFindCtx(ctx);
            // Try to locate symbol memory pointer in list
            nodeMem = cgFindMem(nodeCtx, (CUdeviceptr)symbolPtr);
            // Add pointer to list and set isSymbol if this symbol is used for the first time
            if (!nodeMem) {
               cgAddMem(ctx, (CUdeviceptr)symbolPtr, symbolSize);
               nodeMem = cgFindMem(nodeCtx, (CUdeviceptr)symbolPtr);
               nodeMem->isSymbol = 1;
            }
            CALL_FN_W_5W(result, fn, dst, symbol, count, offset, kind);
         }
         break;
      }
      default: {
         errors++;
         VALGRIND_PRINTF("Error: Invalid kind in call to cudaMemcpyFromSymbol\n");
         VALGRIND_PRINTF("       Expected cudaMemcpyDeviceToHost or cudaMemcpyDeviceToDevice.\n");
      }
   }
   
   if (errors) {
      VALGRIND_PRINTF_BACKTRACE("\n");
   }
   
   cgUnlock();
}