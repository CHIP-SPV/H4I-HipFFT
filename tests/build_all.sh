#!/bin/bash

# set the env
#source setUpModules.sh

myList=$(cat list_of_tests)

for test in $myList
do
    echo "building $test ..."
    cd $test
    ./build.sh &>  my_build_output
    cd ../
done
