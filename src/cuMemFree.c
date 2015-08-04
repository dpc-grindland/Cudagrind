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
 * Wrapper for cuMemFree that removes freed memory from internal lists.
 *
 ******************************************************************************/
#include <cudaWrap.h>

// TODO: It looks like _v2 versions are automatically wrapped .. why?!
CUresult I_WRAP_SONAME_FNNAME_ZZ(libcudaZdsoZa, cuMemFree)(CUdeviceptr dptr) {
   OrigFn fn;
   CUresult result;
   CUcontext ctx = NULL;
   
   VALGRIND_GET_ORIG_FN(fn);
   cgLock();
   
   if (dptr != 0) {
      cgGetCtx(&ctx);
      cgDelMem(ctx, dptr);
   } else {
      VALGRIND_PRINTF_BACKTRACE("Error: cuMemFree called with invalid NULL pointer.\n");
   }
   CALL_FN_W_W(result, fn, dptr);

   cgUnlock();
   return result;
}