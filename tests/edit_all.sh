#!/bin/bash

# set the env
#source setUpModules.sh

myList=$(cat list_of_tests)

for test in $myList
do
    echo "entering $test ..."
    cd $test
    #vi $test.cpp
    #vi Makefile
    #vi build.sh
    #git rm setUpModules.sh
    #git rm check_env.sh
    ls -ltr 
    #vi run_it.sh
    #vi submission_script.sh
    #git diff submission_script.sh
    #git diff run_it.sh
    cd ../
done
