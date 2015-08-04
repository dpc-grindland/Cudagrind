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
 * Wrapper for cudaDeviceSynchronize
 *
 * Clears locks placed by previous calls to wrappers 
 *  for asynchronous memory operations.
 *
 ******************************************************************************/
#include <cudaWrap.h>

cudaError_t  I_WRAP_SONAME_FNNAME_ZZ(libcudartZdsoZa, cudaDeviceSynchronize)() {
   // TODO: Rewrite to use helper functions instead of going through the
   //        whole list manually.
   
   // Pointer to the Cudagrind internal list of registered contexts
   extern cgCtxListType *cgCtxList;
   // Fetch this pointer to work with locally
   cgCtxListType *nodeCtx = cgCtxList;
   
   OrigFn      fn;
   cudaError_t result;
   
   VALGRIND_GET_ORIG_FN(fn);
   cgLock();
   // TODO: Could we call original function before lock to cover more cases?
   CALL_FN_W_v(result, fn);
   
   // We go through the whole list and delete all locks held by any stream.
   while (nodeCtx) {
      cgMemListType *nodeMem = nodeCtx->memory;
      
      while (nodeMem) {
         if (nodeMem->locked) {
            nodeMem->locked = 0;
         }
         nodeMem = nodeMem->next;
      }
      
      cgArrListType *nodeArr = nodeCtx->array;
      
      while (nodeArr) {
         if (nodeArr->locked) {
            nodeArr->locked = 0;
         }
         nodeArr = nodeArr->next;
      }
      
      nodeCtx = nodeCtx->next;
   }
   
   cgUnlock();
}   