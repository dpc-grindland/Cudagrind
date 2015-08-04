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
 * Wrapper for cuStreamSynchronize
 *
 * Clears locks placed by previous calls to wrappers
 *  for asynchronous memory operations.
 *
 ******************************************************************************/
#include <cudaWrap.h>

CUresult I_WRAP_SONAME_FNNAME_ZZ(libcudaZdsoZa, cuStreamSynchronize)(CUstream hStream) {
   // TODO: Rewrite to use helper functions instead of going through the
   //        whole list manually.
   
   // Pointer to the Cudagrind internal list of registered contexts
   extern cgCtxListType *cgCtxList;
   // Fetch this pointer to work with locally
   cgCtxListType *nodeCtx = cgCtxList;
   
   OrigFn      fn;
   CUresult    result;
   
   VALGRIND_GET_ORIG_FN(fn);
   cgLock();
   // TODO: Could we call original function before lock to cover more cases?
   CALL_FN_W_W(result, fn, hStream);
   
   // We go through the whole list and delete
   //  locks wherever hold by stream hStream.
   while (nodeCtx) {
      cgMemListType *nodeMem = nodeCtx->memory;
      
      while (nodeMem) {
         if (nodeMem->locked && nodeMem->stream == hStream) {
            nodeMem->locked = 0;
         }
         nodeMem = nodeMem->next;
      }
      
      cgArrListType *nodeArr = nodeCtx->array;
      
      while (nodeArr) {
         if (nodeArr->locked && nodeArr->stream == hStream) {
            nodeArr->locked = 0;
         }
         nodeArr = nodeArr->next;
      }
      
      nodeCtx = nodeCtx->next;
   }
   
   cgUnlock();
}   