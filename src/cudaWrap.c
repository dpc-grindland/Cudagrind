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
 * Provides a set of helper functions and their interface used in the wrapper 
 * functions of the Valgrind based memory transfer checker Cudagrind.
 *
 ******************************************************************************/
 
/******************************************************************************
 *
 * The implemented functionality can be seen in the src/ directory, with one
 * dedictated source file per working wrapper.
 *
 * To run a CUDA program with Cudagrind make sure to PRELOAD the 
 * libcudaWrap.so dynamic library and call the original program with Valgrind.
 *
 *****************************************************************************/
#ifdef __cplusplus
extern "C" {
#endif

#include "cudaWrap.h"
#ifndef CUDAGRIND_NOPTHREAD
#include <pthread.h>
#endif

cgCtxListType *cgCtxList = NULL;

// The gcc constructor facility is used to initialize the POSIX Mutex that makes
//  Cudagrind threadsafe and output a welcome Message when it's being run.
void __attribute__ ((constructor)) cgConstructor() {
#ifndef CUDAGRIND_NOPTHREAD
   // If CUDAGRIND_NOTHREADS is 
   int error;
   // Set to PTHREAD_MUTEX_RECURSIVE to enable recursive calls inside Cudagrind wrappers
   error = pthread_mutexattr_settype(&cgMutexAttr, PTHREAD_MUTEX_RECURSIVE);
   if (error) {
      VALGRIND_PRINTF("Error: Constructor failed to set mutex attribute with error %d.\n", error);
   }
   error = pthread_mutex_init(&cgMutex, &cgMutexAttr);
   if (error) {
      VALGRIND_PRINTF("Error: Constructor failed to set mutex attribute with error %d.\n", error);
   }
#endif
   VALGRIND_PRINTF("\nWelcome to Cudagrind version %d.%d.%d\n\n", CG_VERSION_MAJOR, CG_VERSION_MINOR, CG_VERSION_REVISION);
}

// The gcc destructor facility is used to clear the POSIX Mutex
void __attribute__ ((destructor)) cgDestructor() {
#ifndef CUDAGRIND_NOPTHREAD
   int error;
   error = pthread_mutex_destroy(&cgMutex);
   if (error) {
      VALGRIND_PRINTF("Error: Deconstructor failed to free mutex with error %d.\n", error);
   }
#endif
   
   // Search through whole context and memory/array lists for unfreed memory.
   //  Memory allocated to handle symboles will be ignored.
   cgCtxListType  *nodeCtx = cgCtxList;
   cgMemListType  *nodeMem;
   cgArrListType  *nodeArr;
   int            countMem = 0, countArr = 0;
   size_t         sizeMem  = 0, sizeArr  = 0;
   
   while (nodeCtx) {
      nodeMem = nodeCtx->memory;
      while (nodeMem) {
         if (!(nodeMem->isSymbol)) {
            countMem++;
            sizeMem += nodeMem->size;
         }
         nodeMem = nodeMem->next;
      }
      nodeArr = nodeCtx->array;
      while (nodeArr) {
         countArr++;
         // TODO: Calculate amount of lost memory for arrays
         nodeArr = nodeArr->next;
      }
      nodeCtx = nodeCtx->next;
   }
   if (countMem || countArr) {
      VALGRIND_PRINTF("Cudagrind leak report:\n");
      if (countMem) {
         VALGRIND_PRINTF("   %d device memory region (%lu bytes) not freed.\n", countMem, (unsigned long)sizeMem);
      }
      if (countArr) {
         VALGRIND_PRINTF("   %d device arrays not freed.\n", countArr);
      }
   }
}


// Helper function to lock the Cudagrind mutex
void cgLock() {
#ifndef CUDAGRIND_NOPTHREAD
   pthread_mutex_lock(&cgMutex);
#endif
}

// Helper function to unlock the Cudagrind mutex
void cgUnlock() {
#ifndef CUDAGRIND_NOPTHREAD
   pthread_mutex_unlock(&cgMutex);
#endif
}

#ifdef CUDAGRIND_DEBUG
// Helper function that prints the current ctx/gpu-mem list on the screen
void printCtxList() {
   cgCtxListType *ctxList = cgCtxList;
   cgMemListType *memList;
   cgArrListType *arrList;
   
   if (cgCtxList) {
      while (ctxList) {
         VALGRIND_PRINTF("CTX %lu: ", ctxList->ctx);
         memList = ctxList->memory;
         while (memList) {
            VALGRIND_PRINTF("%lu (%lu) -> ", memList->dptr, memList->size);
            memList = memList->next;
         }
         VALGRIND_PRINTF("\n");
         if (ctxList->array) {
            VALGRIND_PRINTF("   Arr: ");
            arrList = ctxList->array;
            // TODO: More debug output (size) for arrays?
            while (arrList) {
               VALGRIND_PRINTF("%lu -> ", arrList->dptr);
               arrList = arrList->next;
            }
            VALGRIND_PRINTF("\n");
         }
         ctxList = ctxList->next;
      }
   } else {
      VALGRIND_PRINTF("CTX list is currently empty!\n");
   }
}
#endif

// Helper function that fetches the current CUDA context
void cgGetCtx(CUcontext *ctx) {
   CUresult res;
   
   res = cuCtxGetCurrent(ctx);
   if (res != CUDA_SUCCESS) {
      VALGRIND_PRINTF_BACKTRACE("Error: Retrieving CUDA context in VG-wrapper failed.\n"
      );
   } else if (*ctx == NULL) {
      VALGRIND_PRINTF_BACKTRACE("Error: Retrieved NULL context in Valgrind wrapper.\n:"
      );
   }
}

/*
 * Returns the memory segment of the CUDA context referenced by ctx.
 * The entry is created if it is not found in the list.
 *
 * Input:
 *   CUcontext ctx  - context reference
 * Return value:
 *   cgCtxListType* - Pointer to ctx's list entry
*/
cgCtxListType* cgFindCtx(CUcontext ctx) {
   cgCtxListType *list;
   // Create new entry if list is still empty
   if (!cgCtxList) {
      cgCtxList = (cgCtxListType*)malloc(sizeof(cgCtxListType));
      list = cgCtxList;
      list->ctx = ctx;
      list->memory   = NULL;
      list->array    = NULL;
      list->next     = NULL;
   } else {
      list = cgCtxList;
      while (list->next && list->ctx != ctx) {
         list = list->next;
      }
      if (list->ctx != ctx) {
         list->next = (cgCtxListType*)malloc(sizeof(cgCtxListType));
         list = list->next;
         list->ctx = ctx;
         list->memory   = NULL;
         list->array    = NULL;
         list->next     = NULL;
      }
   }
   return list;
}



/*
 * Calculates bytes per array element for given array descriptor.
 *
 * Input:
 *   CUDA_ARRAY3D_DESCRIPTOR *desc - cuda 3D array descriptor
 *                                   Note: also works for 1D/2D arrays
 * Return value:
 *   int - number of bytes for a single array element
 *         0 if desc->Format specifies an unknown format value
 *
 */
int cgArrDescBytesPerElement(CUDA_ARRAY3D_DESCRIPTOR *desc) {
    switch (desc->Format) {
      case CU_AD_FORMAT_UNSIGNED_INT8:
      case CU_AD_FORMAT_SIGNED_INT8:
         return 2 * desc->NumChannels;
      case CU_AD_FORMAT_UNSIGNED_INT16:
      case CU_AD_FORMAT_SIGNED_INT16:
      case CU_AD_FORMAT_HALF:
         return 4 * desc->NumChannels;
      case CU_AD_FORMAT_UNSIGNED_INT32:
      case CU_AD_FORMAT_SIGNED_INT32:
      case CU_AD_FORMAT_FLOAT:
         return 8 * desc->NumChannels;
      default:
         return 0;
   }
}



/*
 * Returns the memory node referenced by the device pointer from the given list
 *
 * Note: Also works for pointers pointing 'inside' an allocate memory region 
 *       in order to support pointer arithmetic with device pointers.
 *
 * Input:
 *   cgCtxListType *node - Reference to the list of memory regions
 *   CUdeviceptr   dptr  - Pointer to the memory on the device
 * Return value:
 *   cgMemListType* - Pointer to the entry in the list, 
 *                    NULL if it could not be found.
 */
cgMemListType* cgFindMem(cgCtxListType *nodeCtx, CUdeviceptr dptr) {
   cgMemListType *node = nodeCtx->memory;
   // while (node && node->dptr != dptr) { // Old version, only checks node->dptr
   // Go through the whole list and find the node dptr points into
   while (node && (dptr < node->dptr || dptr >= node->dptr + node->size)) {
   	node = node->next;
   }
   return node;
}



/*
 * Adds an entry for the device memory referenced by dptr of given size to the given context node ctxNode.
 *
 * cgCtxLisType *ctxNode - Node containing memory information for context
 * CUdeviceptr  dptr     - Device pointer to memory
 * size_t       size     - Size of memory referenced by dptr
 */
void cgCtxAddMem(cgCtxListType *ctxNode, CUdeviceptr dptr, size_t size) {
   cgMemListType *node;
   // Create new entry if list is still empty
   if (!(ctxNode->memory)) {
      ctxNode->memory = (cgMemListType*)malloc(sizeof(cgMemListType));
      node = ctxNode->memory;
      node->dptr = dptr;
      node->isSymbol = 0;
      node->size = size;
      node->locked = 0;
      // Do not have to set node->stream here
      node->next = NULL;
   } else {
      node = ctxNode->memory;
      while (node->next && node->dptr != dptr) {
         node = node->next;
      }
      if (node->dptr != dptr) {
         node->next = (cgMemListType*)malloc(sizeof(cgMemListType));
         node = node->next;
         node->dptr = dptr;
         node->isSymbol = 0;
         node->size = size;
         node->locked = 0;
         // Do not have to set node->stream here
         node->next = NULL;
      } else {
         VALGRIND_PRINTF("Error: Tried to add already existing device pointer in cgCtxAddMem.\n");
         VALGRIND_PRINTF_BACKTRACE("Possible reason: Unknown. This should not have happened!");
      }
   }
}



/*
 * Removes the entry of memory referenced by dptr from ctxNode's memory list.
 *
 * Input:
 *   cgCtxListType *ctxNode - The node from which dptr is to be removed
 *   CUdeviceptr   dptr     - The device pointer of the to be removed memory entry
 */
void cgCtxDelMem(cgCtxListType *ctxNode, CUdeviceptr dptr) {
   cgMemListType *toFree, *node = ctxNode->memory;
   int deleted = 0;
   // Run through list of memory segments and remove it if it's found
   if (node) {
      if (node->dptr == dptr) {
         ctxNode->memory = node->next;
         toFree = node;
         deleted = 1;
      } else {
         while (node->next && node->next->dptr != dptr) {
            node = node->next;
         }
         // If node->next is not NULL it has to contain dptr now
         if (node->next) {
            toFree = node->next;
            node->next = node->next->next;
            deleted = 1;
         }
      }
   }
   // Print error if the to be deletec device pointer can not be found
   if (!deleted) {
      VALGRIND_PRINTF("Error: Tried to delete device pointer that could not be located.\n");
      VALGRIND_PRINTF_BACKTRACE("Possible reason: Wrong CUDA context or double free on device memory pointer.\n");
   } else { // Else free the memory used by the node ..
      free(toFree);
      // TODO: Also remove the context entry if it's empty? And where?
   }
}



/*
 * Adds the memory residing in context ctx and referenced by dptr to the allocated memory list.
 * 
 * Input:
 *   CUcontext   ctx      - CUDA context of memory
 *   CUdeviceptr dptr     - device pointer of memory
 *   size_t      bytesize - size of memory in bytes
 */
void cgAddMem(CUcontext ctx, CUdeviceptr dptr, size_t bytesize) {
   cgCtxListType *ctxNode;
   
   // Locate or create entry for context
   ctxNode = cgFindCtx(ctx);
   cgCtxAddMem(ctxNode, dptr, bytesize);
	#ifdef CUDAGRIND_DEBUG
   VALGRIND_PRINTF("Context list after 'cgAddMem' operation:\n");
   printCtxList();
   #endif
}



/*
 * Removes the entry of the memory residing context ctx and references by dptr from allocated memory list.
 *
 * Input:
 *   CUcontext   ctx  - context
 *   CUdeviceptr dptr - device pointer of to be removed memory
 */
void cgDelMem(CUcontext ctx, CUdeviceptr dptr) {
   cgCtxListType *ctxNode;
   // Locate the entry for the context .. 
   // TODO: What happens if context does not exist anymore (we create it ..
   //         but is that what we really want?!)
   ctxNode = cgFindCtx(ctx);
   cgCtxDelMem(ctxNode, dptr);
	#ifdef CUDAGRIND_DEBUG
   VALGRIND_PRINTF("Context list after 'cgDelMem' operation:\n");
   printCtxList();
   #endif
}



/*
 * Returns the memory node referenced by the array device pointer from the given list
 *
 * Input:
 *   cgCtxListType *node - Reference to the list of memory regions
 *   CUdeviceptr   dptr  - Pointer to the memory on the device
 * Return value:
 *   cgArrListType* - Pointer to the entry in the list, 
 *                      NULL if it could not be found.
 */
cgArrListType* cgFindArr(cgCtxListType *nodeCtx, CUarray dptr) {
   cgArrListType *node = nodeCtx->array;
   while (node && node->dptr != dptr) {
   	node = node->next;
   }
   return node;
}



/*
 * Adds an entry for the array referenced by dptr with given descriptor desc to the given context node ctxNode.
 *
 * cgCtxLisType *ctxNode       - Node containing memory information for context
 * CUarray      dptr           - Device array reference (1D,2D or 3D)
 * CUDA_ARRAY3D_DESCRIPTOR desc- Descripter array referenced by dptr
 */
void cgCtxAddArr(cgCtxListType *ctxNode, CUarray dptr, CUDA_ARRAY3D_DESCRIPTOR desc) {
   cgArrListType *node;
   // Create new entry if list is still empty
   if (!(ctxNode->array)) {
      ctxNode->array = (cgArrListType*)malloc(sizeof(cgArrListType));
      node = ctxNode->array;
      node->dptr = dptr;
      node->desc = desc;
      node->locked = 0;
      // Do not have to set node->stream here
      node->next = NULL;
   } else {
      node = ctxNode->array;
      while (node->next && node->dptr != dptr) {
         node = node->next;
      }
      if (node->dptr != dptr) {
         node->next = (cgArrListType*)malloc(sizeof(cgArrListType));
         node = node->next;
         node->dptr = dptr;
         node->desc = desc;
         node->locked = 0;
         // Do not have to set node->stream here
         node->next = NULL;
      } else {
         VALGRIND_PRINTF("Error: Tried to add already existing array reference in cgCtxAddArr.\n");
         VALGRIND_PRINTF_BACKTRACE("Possible reason: Unknown. This should not have happened!");
      }
   }
}



/*
 * Removes the entry of array referenced by dptr from ctxNode's array list.
 *
 * Input:
 *   cgCtxListType *ctxNode - The node from which dptr is to be removed
 *   CUarray       dptr     - The device pointer of the to be removed array entry
 */
void cgCtxDelArr(cgCtxListType *ctxNode, CUarray dptr) {
   cgArrListType *toFree, *node = ctxNode->array;
   int deleted = 0;
   // Run through list of memory segments and remove it if it's found
   if (node) {
      if (node->dptr == dptr) {
         ctxNode->array = node->next;
         toFree = node;
         deleted = 1;
      } else {
         while (node->next && node->next->dptr != dptr) {
            node = node->next;
         }
         // If node->next is not NULL it has to contain dptr now
         if (node->next) {
            toFree = node->next;
            node->next = node->next->next;
            deleted = 1;
         }
      }
   }
   // Print error if the to be deletec device pointer can not be found
   if (!deleted) {
      VALGRIND_PRINTF("Error: Tried to remove non-existant device array reference in cgCtxDelArr.\n");
      VALGRIND_PRINTF_BACKTRACE("Possible reason: Wrong CUDA context or double free on device array pointer.\n");
   } else { // Else free the memory used by the node ..
      free(toFree);
      // TODO: Also remove the context entry if it's empty? And where?
   }
}



/*
 * Adds the array residing in context ctx and referenced by dptr to the allocated array list.
 * 
 * Input:
 *   CUcontext             ctx  - CUDA context of array
 *   CUarray               dptr - device pointer of array
 */
void cgAddArr(CUcontext ctx, CUarray dptr) {
   cgCtxListType *ctxNode;
   cgArrListType *arrNode;
   // Initialize desc with dummy values
   CUDA_ARRAY3D_DESCRIPTOR desc = {0, 0, CU_AD_FORMAT_FLOAT, 0, 0, 0};
   CUresult error;
   
   // Locate or create entry for context
   ctxNode = cgFindCtx(ctx);
   // Nasty hack: 
   //    If we call cuArray3DGetDescriptor first, its wrapper will
   //    produce a spurious (albeit, at this point, legit) error message,
   //    because the array is not contained in our internal list. But if 
   //    we call cgCtxAddArr first, we are missing the array descriptor!
   //    -> We add the array with a dummy descriptor which in turn allows
   //       us to fetch the original descriptor through the driver without
   //       triggering the checks in the cuArray3DGetDescriptor wrapper.
   //
   // 1. Add array to internal list, including dummy descriptor
   cgCtxAddArr(ctxNode, dptr, desc);
   // 2. Get real descriptor, the wrapper wont complain now
   error = cuArray3DGetDescriptor(&desc, dptr);
   if (error != CUDA_SUCCESS) {
      VALGRIND_PRINTF("Error: cuArray3DGetDescriptor returned with %d in cgAddArr.\n", error);
   }
   // 3. Fetch node of array and set the real descriptor
   arrNode = cgFindArr(ctxNode, dptr);
   arrNode->desc = desc;
   
	#ifdef CUDAGRIND_DEBUG
   VALGRIND_PRINTF("Context list after 'cgAddArr' operation:\n");
   printCtxList();
   #endif
}



/*
 * Removes the entry of the array residing in context ctx and referenced by dptr from allocated array list.
 *
 * Input:
 *   CUcontext ctx  - context
 *   CUarray   dptr - device pointer of to be removed 2D array
 */
void cgDelArr(CUcontext ctx, CUarray dptr) {
   cgCtxListType *ctxNode;
   // Locate the entry for the context .. 
   // TODO: What happens if context does not exist anymore (we create it ..
   //         but is that what we really want?!)
   ctxNode = cgFindCtx(ctx);
   cgCtxDelArr(ctxNode, dptr);
	#ifdef CUDAGRIND_DEBUG
   VALGRIND_PRINTF("Context list after 'cgDelArr' operation:\n");
   printCtxList();
   #endif
}

#ifdef __cplusplus
}
#endif
