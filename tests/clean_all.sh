#!/bin/bash

# set the env
#source setUpModules.sh

myList=$(cat list_of_tests)

for test in $myList
do
    echo "cleaning $test ..."
    cd $test
    make -f Makefile realclean > my_clean_output
    rm -f *.e* *.o* my_*_output *_spectrum
    cd ../
done
