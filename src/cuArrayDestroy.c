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
 * Wrapper for cuArrayDestroy that removes the 
 * to be destroyed array from the context's list.
 *
 ******************************************************************************/
#include <cudaWrap.h>

CUresult I_WRAP_SONAME_FNNAME_ZZ(libcudaZdsoZa, cuArrayDestroy)(CUarray hArray) {
   OrigFn      fn;
   CUresult    result;
   CUcontext   ctx = NULL;
   
   VALGRIND_GET_ORIG_FN(fn);
   cgLock();
   
   if (hArray != 0) {
      cgGetCtx(&ctx);
      cgDelArr(ctx, hArray);
   } else {
      VALGRIND_PRINTF_BACKTRACE("Error: cuArrayDestroy called with invalid NULL pointer.\n");
   }
   CALL_FN_W_W(result, fn, hArray);
   
   cgUnlock();
   return result;
}