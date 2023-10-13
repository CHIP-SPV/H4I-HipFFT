#!/bin/bash

# set the env
#source setUpModules.sh

myList=$(cat list_of_tests)

for test in $myList
do
    echo "entering $test ..."
    cd $test

    rm -rf my_output

    ./$test.x >& my_output

    grep "max error" my_output
    cd ../
done
