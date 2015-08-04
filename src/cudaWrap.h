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
 * Header file for helper functions provided by cudaWrap.c.
 *
 ******************************************************************************/
#ifndef CUDAWRAP_H
#define CUDAWRAP_H

#include <valgrind.h>
#include <memcheck.h> // VALGRIND_CHECK_MEM_IS_DEFINED
#include <cuda.h>
#include <cuda_runtime.h> // Needed for cudaError_t to be visible in wrapeprs
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#ifndef CUDAGRIND_NOPTHREAD
#include <pthread.h> // include pthread.h for POSIX Mutex
#endif

#define CG_STREAM_READING 1
#define CG_STREAM_WRITING 2

// Current version of Cudagrind
#define CG_VERSION_MAJOR   0
#define CG_VERSION_MINOR   9
#define CG_VERSION_REVISION 4

// Memory segments registered to a single context
typedef struct cgMemListType_s {
   CUdeviceptr dptr;
   size_t      size;
   // 1 if dptr points to a CUDA symbol, 0 else
   int         isSymbol;
   // 1 if 'stream' is reading from memory section, 2 if writing, 0 else
   // TODO: For more precise error detection a list of locks should be used.
   int         locked;
   CUstream    stream;
   struct cgMemListType_s *next;
} cgMemListType;

// Single data structure for arrays. Note: We use CUDA_ARRAY3D_DESCRIPTOR for
//  all cases. This works because cuArray3DGetDescriptor also works for arrays
//  of lower dimensions (with Height and/or Depth set to 0).
typedef struct cgArrListType_s {
   CUDA_ARRAY3D_DESCRIPTOR      desc;
   CUarray                    dptr;
   // 1 if 'stream' is reading from memory section, 2 if writing, 0 else
   // TODO: For more precise error detection a list of locks should be used.
   int         locked;
   CUstream    stream;
   struct cgArrListType_s   *next;
} cgArrListType;

// List of nodes each representing a single context and its memory segments
typedef struct cgCtxListType_s {
   // Context id, number of entries in memory and size of memory
   CUcontext              ctx;
   cgMemListType          *memory;
   cgArrListType          *array;
   struct cgCtxListType_s *next;
} cgCtxListType;

extern cgCtxListType *cgCtxList;

// Helper functions defined in cudaWrap.c
#ifdef CUDAGRIND_DEBUG
void printCtxList();
#endif
void cgGetCtx(CUcontext *ctx);
cgCtxListType* cgFindCtx(CUcontext ctx);

// Calculates the bytes per element for a given cuda array descriptor
int cgArrDescBytesPerElement(CUDA_ARRAY3D_DESCRIPTOR *desc);

cgMemListType* cgFindMem(cgCtxListType *nodeCtx, CUdeviceptr dptr);
void cgCtxAddMem(cgCtxListType *ctxNode, CUdeviceptr dptr, size_t size);
void cgCtxDelMem(cgCtxListType *ctxNode, CUdeviceptr dptr);
void cgAddMem(CUcontext ctx, CUdeviceptr dptr, size_t bytesize);
void cgDelMem(CUcontext ctx, CUdeviceptr dptr);

cgArrListType* cgFindArr(cgCtxListType *nodeCtx, CUarray dptr);
void cgCtxAddArr(cgCtxListType *ctxNode, CUarray dptr, const CUDA_ARRAY3D_DESCRIPTOR array);
void cgCtxDelArr(cgCtxListType *ctxNode, CUarray dptr);
void cgAddArr(CUcontext ctx,  CUarray dptr);
void cgDelArr(CUcontext ctx,  CUarray dptr);


// Mutex un-/lock to make Cudagrind threadsafe
//   Both functions are empty if NOPTHREAD is defined.
void cgLock();
void cgUnlock();

#ifndef CUDAGRIND_NOPTHREAD
// POSIX mutex and mutex-attribute variable
pthread_mutex_t cgMutex;
pthread_mutexattr_t cgMutexAttr;
#endif

#endif
