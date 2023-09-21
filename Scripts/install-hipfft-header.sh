#!/bin/sh
# Copyright 2021-2023 UT-Battelle
# See LICENSE.txt in the root of the source distribution for license info.

if (( $# != 3 ))
then
    echo "usage: $(basename $0) source_dir bindir inst_prefix" >&2
    exit 1
fi

srcdir=$1
bindir=$2
instdir=$3

mkdir -p $bindir/include
cp $srcdir/library/include/hipfft/hipfft.h $bindir/include/
mkdir -p $instdir/share/licenses
cp $srcdir/LICENSE.md $instdir/share/licenses/hipfft.h.LICENSE.md


