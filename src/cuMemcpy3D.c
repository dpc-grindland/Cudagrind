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
 * Wraps the cuMemcpy3D function to check 3D memory copies.
 * The runtime seems to call this instead of cuMemcpy2D (at certain times?).
 *
 * TODO: Finish missing memory types && decide how to handle unified memory
 * TODO: Additional Valgrind IS_DEFINED/ADDRESSABLE checks on pCopy components?
 *
 ******************************************************************************/
 #include <cudaWrap.h>

CUresult I_WRAP_SONAME_FNNAME_ZZ(libcudaZdsoZa, cuMemcpy3D)(const CUDA_MEMCPY3D *pCopy) {
   OrigFn      fn;
   CUresult    result;
   CUcontext  ctx = NULL;
   int error = 0, error_addressable, error_defined;
   long vgErrorAddress = 0, vgErrorAddressDefined = 0;
   
     
   VALGRIND_GET_ORIG_FN(fn);
   cgLock();
   CALL_FN_W_W(result, fn, pCopy);

   // Check if pCopy is null, not allocated or undefined.
   // For obvious reasons we skip the following checks if either condition is true.
   if (!pCopy) {
      error++;
      VALGRIND_PRINTF_BACKTRACE("Error: pCopy in call to cuMemcpy3D is NULL.\n");
      cgUnlock();
      return result;
   } else if ( vgErrorAddress = VALGRIND_CHECK_MEM_IS_ADDRESSABLE(pCopy, sizeof(CUDA_MEMCPY3D)) ) {
      error++;
      VALGRIND_PRINTF_BACKTRACE("Error: pCopy in call to cuMemcpy3D points to unallocated memory.\n");
      cgUnlock();
      return result;
   } // It makes no sense to check _IS_DEFINED on the whole structure, since only part of it is used!
   
   // General checks of constaints imposed by reference manual
   if (pCopy->srcMemoryType != CU_MEMORYTYPE_ARRAY) {
      if (pCopy->srcPitch && pCopy->srcPitch < pCopy->WidthInBytes + pCopy->srcXInBytes) {
         error++;
         VALGRIND_PRINTF("Error: srcPitch < WidthInBytes+srcXInBytes in cuMemcpy3D.\n");
      }
      if (pCopy->srcHeight && pCopy->srcHeight < pCopy->Height + pCopy->srcY) {
         error++;
         VALGRIND_PRINTF("Error: srcHeight < Height+srcY in cuMemcpy3D.\n");
      }
   }
   if (pCopy->dstMemoryType != CU_MEMORYTYPE_ARRAY) {
      if (pCopy->dstPitch && pCopy->dstPitch < pCopy->WidthInBytes + pCopy->dstXInBytes) {
         error++;
         VALGRIND_PRINTF("Error: dstPitch < WidthInBytes+dstXInBytes in cuMemcpy3D.\n");
      }
      if (pCopy->dstHeight && pCopy->dstHeight < pCopy->Height + pCopy->dstY) {
         error++;
         VALGRIND_PRINTF("Error: dstHeight < Height+dstY in cuMemcpy3D.\n");
      }
   }
   switch (pCopy->srcMemoryType) {
      case CU_MEMORYTYPE_UNIFIED:
         // TODO: How do we handle unified memory?
         break;
      case CU_MEMORYTYPE_HOST: {
         void *line;
      
         error_addressable = 0;
         error_defined = 0;
         // TODO: Is Height, Depth > 1, even for 1D/2D copy operations?   
         for (int i = 0 ; i < pCopy->Height ; i++) {
            for (int j = 0 ; j < pCopy->Depth ; j++) {
               line = (void*)(
                        (char*)pCopy->srcHost 
                        + ((pCopy->srcZ + j) * pCopy->srcHeight + (pCopy->srcY + i))*pCopy->srcPitch 
                        + pCopy->srcXInBytes
                     );
               vgErrorAddress = VALGRIND_CHECK_MEM_IS_ADDRESSABLE(line, (size_t)pCopy->WidthInBytes);
               if (vgErrorAddress) {
                  error_addressable++;
               } else {
                  vgErrorAddress = VALGRIND_CHECK_MEM_IS_DEFINED(line, (size_t)pCopy->WidthInBytes);
                  if (vgErrorAddress) {
                     error_defined++;
                  }
               }
            }
         }
         // TODO: Can we give precise information about location of error?
         if (error_addressable) {
            error++;
            VALGRIND_PRINTF("Error: (Part of) source host memory not allocated\n"
                            "       in call to cuMemcpy3D.\n");
         }
         if (error_defined) {
            error++;
            VALGRIND_PRINTF("Error: (Part of) source host memory not defined\n"
                            "       in call to cuMemcpy3D.\n");
         }
         break;
      }
      case CU_MEMORYTYPE_DEVICE: {
         // ptrEnd points to the end of the memory area which pCopy->srcDevice points into
         CUdeviceptr line, ptrEnd;
         cgMemListType *nodeMem;
         
         // TODO: Check if pCopy->srcDevice is defined?
         cgGetCtx(&ctx);
         nodeMem = cgFindMem(cgFindCtx(ctx), pCopy->srcDevice);
         
         // We only track addressable status (whether memory is allocated) for device memory regions
         error_addressable = 0;
         if (nodeMem) {
            ptrEnd = nodeMem->dptr + nodeMem->size;
            /*
            for (int i = 0 ; i < pCopy->Height ; i++) {
               for (int j = 0 ; j < pCopy->Depth ; j++) {
                  line = (CUdeviceptr)(
                           pCopy->srcDevice
                           + ((pCopy->srcZ + j) * pCopy->srcHeight + (pCopy->srcY + i)) * pCopy->srcPitch
                           + pCopy->srcXInBytes
                         );
                  
                  // Is there enough allocated memory left to statisfy the current line?
                  if (ptrEnd - line < pCopy->WidthInBytes) {
                     error_addressable++;
                  }
               }
            }
            */
            
            // Device memory should not be fragmented, so we only check the very last slice of memory
            line = (CUdeviceptr)(
                     pCopy->srcDevice 
                     + (
                        (pCopy->srcZ + pCopy->Depth - 1) * pCopy->srcHeight 
                        + (pCopy->srcY + pCopy->Height - 1)
                       ) * pCopy->srcPitch 
                     + pCopy->srcXInBytes);
            if (ptrEnd - line < pCopy->WidthInBytes) {
               error_addressable++;
            }
         } else {
            error_addressable++;
         }
         
         if (error_addressable) {
            error++;
            VALGRIND_PRINTF("Error: (Part of) source device memory not allocated\n"
                            "       in call to cuMemcpy3D.\n");
         }
         break;
      }
      case CU_MEMORYTYPE_ARRAY: {
         CUDA_ARRAY3D_DESCRIPTOR descriptor;
         int bytesPerElement;
         int widthInBytes;
         
         // Fetch array descriptor ..
         cuArray3DGetDescriptor(&descriptor, pCopy->srcArray);
         bytesPerElement = cgArrDescBytesPerElement(&descriptor);
         if (!bytesPerElement) {
            error++;
            VALGRIND_PRINTF("Error: Unknown Format value in src array descriptor in cuMemcpy3D.\n");
         }
         widthInBytes = bytesPerElement * descriptor.Width;
         // .. and check if dimensions are conform to the ones requested in pCopy
         if (widthInBytes - pCopy->srcXInBytes < pCopy->WidthInBytes) {
            error++;
            VALGRIND_PRINTF("Error: Available width of %u bytes in source array is smaller than\n"
                            "       requested Width of %u bytes in pCopy of cuMemcpy3D.\n", 
                                    widthInBytes - pCopy->srcXInBytes, pCopy->WidthInBytes);
         }
         if (pCopy->Height > 1 && descriptor.Height - pCopy->srcY < pCopy->Height) {
            error++;
            VALGRIND_PRINTF("Error: Available Height of %u in source array is smaller than\n"
                            "       requested Height of %u in pCopy of cuMemcpy3D.\n",
                            descriptor.Height - pCopy->srcY, pCopy->Height);
         }
         if (pCopy->Depth > 1 && descriptor.Depth - pCopy->srcZ < pCopy->Depth) {
            error++;
            VALGRIND_PRINTF("Error: Available Depth of %u in source array is smaller than\n"
                            "       requested Depth of %u in pCopy of cuMemcpy3D.\n",
                            descriptor.Depth - pCopy->srcY, pCopy->Height);
         }
         break;
      }
      default:
         error++;
         VALGRIND_PRINTF("Error: Unknown source memory type %d in cuMemcpy3D\n");
         break;
   }
   
   switch (pCopy->dstMemoryType) {
      case CU_MEMORYTYPE_UNIFIED:
         // TODO: How do we handle unified memory?
         break;
      case CU_MEMORYTYPE_HOST: {
         void *line;
         
         error_addressable = 0;
         error_defined = 0;
         // TODO: Is Height, Depth > 1, even for 1D/2D copy operations?
         for (int i = 0 ; i < pCopy->Height ; i++) {
            for (int j = 0 ; j < pCopy->Depth ; j++) {
               line = (void*)(
                        (char*)pCopy->dstHost 
                        + ((pCopy->dstZ + j) * pCopy->dstHeight + (pCopy->dstY + i))*pCopy->dstPitch 
                        + pCopy->dstXInBytes
                     );
               // Unlike for the source operand we only need to check allocation status here
               vgErrorAddress = VALGRIND_CHECK_MEM_IS_ADDRESSABLE(line, (size_t)pCopy->WidthInBytes);
               if (vgErrorAddress) {
                  error_addressable++;
               }
            }
         }
         // TODO: Can we give precise information about location of error?
         if (error_addressable) {
            error++;
            VALGRIND_PRINTF("Error: (Part of) destination host memory not allocated\n"
                            "       in call to cuMemcpy3D.\n");
         }
         break;
      }
      case CU_MEMORYTYPE_DEVICE: {
         // ptrEnd points to the end of the memory area which pCopy->dstDevice points into
         CUdeviceptr line, ptrEnd;
         cgMemListType *nodeMem;
         
         // TODO: Check if pCopy->dstDevice is defined?
         cgGetCtx(&ctx);
         nodeMem = cgFindMem(cgFindCtx(ctx), pCopy->dstDevice);
         
         // We only track addressable status (whether memory is allocated) for device memory regions
         error_addressable = 0;
         if (nodeMem) {
            ptrEnd = nodeMem->dptr + nodeMem->size;
            /*
            for (int i = 0 ; i < pCopy->Height ; i++) {
               for (int j = 0 ; j < pCopy->Depth ; j++) {
                  line = (CUdeviceptr)(
                           pCopy->dstDevice
                           + ((pCopy->dstZ + j) * pCopy->dstHeight + (pCopy->dstY + i)) * pCopy->dstPitch
                           + pCopy->dstXInBytes
                         );
                  
                  // Is there enough allocated memory left to statisfy the current line?
                  if (ptrEnd - line < pCopy->WidthInBytes) {
                     error_addressable++;
                  }
               }
            }
            */
            
            // Device memory should not be fragmented, so we only check the very last slice of memory
            line = (CUdeviceptr)(
                     pCopy->dstDevice 
                     + (
                        (pCopy->dstZ + pCopy->Depth - 1) * pCopy->dstHeight 
                        + (pCopy->dstY + pCopy->Height - 1)
                       ) * pCopy->dstPitch 
                     + pCopy->dstXInBytes);
            if (ptrEnd - line < pCopy->WidthInBytes) {
               error_addressable++;
            }
         } else {
            error_addressable++;
         }
         
         if (error_addressable) {
            error++;
            VALGRIND_PRINTF("Error: (Part of) destination device memory not allocated\n"
                            "       in call to cuMemcpy3D.\n");
         }
         break;
      }
      case CU_MEMORYTYPE_ARRAY: {
         CUDA_ARRAY3D_DESCRIPTOR descriptor;
         int bytesPerElement;
         int widthInBytes;
         
         // Fetch array descriptor ..
         cuArray3DGetDescriptor(&descriptor, pCopy->dstArray);
         bytesPerElement = cgArrDescBytesPerElement(&descriptor);
         if (!bytesPerElement) {
               error++;
               VALGRIND_PRINTF("Error: Unknown Format value in dst array descriptor in cuMemcpy3D.\n");
         }
         widthInBytes = bytesPerElement * descriptor.Width;
         // .. and check if dimensions are conform to the ones requested in pCopy
         if (widthInBytes - pCopy->dstXInBytes < pCopy->WidthInBytes) {
            error++;
            VALGRIND_PRINTF("Error: Available width of %u bytes in destination array is smaller than\n"
                            "       requested Width of %u bytes in pCopy of cuMemcpy3D.\n", 
                                    widthInBytes - pCopy->dstXInBytes, pCopy->WidthInBytes);
         }
         if (pCopy->Height > 1 && descriptor.Height - pCopy->dstY < pCopy->Height) {
            error++;
            VALGRIND_PRINTF("Error: Available Height of %u in destination array is smaller than\n"
                            "       requested Height of %u in pCopy of cuMemcpy3D.\n",
                            descriptor.Height - pCopy->dstY, pCopy->Height);
         }
         if (pCopy->Depth > 1 && descriptor.Depth - pCopy->dstZ < pCopy->Depth) {
            error++;
            VALGRIND_PRINTF("Error: Available Depth of %u in destination array is smaller than\n"
                            "       requested Depth of %u in pCopy of cuMemcpy3D.\n",
                            descriptor.Depth - pCopy->dstZ, pCopy->Depth);
         }
         break;
      }
      default:
         error++;
         VALGRIND_PRINTF("Error: Unknown destination memory type %d in cuMemcpy3D\n");
         break;
   }
   if (error) {
      VALGRIND_PRINTF_BACKTRACE("   %d errors detected in call to cuMemcpy3D.", error);
   }
   
   cgUnlock();
   return result;
}