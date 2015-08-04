#!/bin/bash

# Small helper script that extracts the current 
#    version number from cudaWrap.h.

# Depending on the first parameter the different version numbers
#  (major, minor, revision) are printed to stdout.
if [ "$1" == "major" ];
then
   awk '/#define CG_VERSION_MAJOR/{major=$3} END{print major}' src/cudaWrap.h
fi
if [ "$1" == "minor" ];
then
   awk '/#define CG_VERSION_MINOR/{minor=$3} END{print minor}' src/cudaWrap.h
fi
if [ "$1" == "revision" ];
then
   awk '/#define CG_VERSION_REVISION/{revision=$3} END{print revision}' src/cudaWrap.h
fi
