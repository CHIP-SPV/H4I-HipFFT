#!/bin/bash

# set the env
#source setUpModules.sh

myList=$(cat list_of_tests)

for test in $myList
do
    echo "entering $test ..."
    cd $test
    vi $test.cpp
    cd ../
done
