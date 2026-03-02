#! /bin/bash

wget https://gitlab.com/MIAOresearch/software/roundingsat/-/archive/master/roundingsat-master.zip
unzip roundingsat-master.zip
cd roundingsat-master
cd build
# wget https://github.com/scipopt/soplex/archive/refs/tags/release-710.tar.gz
# cmake -DCMAKE_BUILD_TYPE=Release -Dsoplex=ON -Dsoplex_pkg=./release-710.tar.gz .. 
if [ "$#" -eq 1 ] && [ "$1" == "soplex" ]; then
        wget https://github.com/scipopt/soplex/archive/refs/tags/release-710.tar.gz
        cmake -DCMAKE_BUILD_TYPE=Release -Dsoplex=ON -Dsoplex_pkg=./soplex-release-710.tar.gz .. 
else
        cmake -DCMAKE_BUILD_TYPE=Release ..
fi
make
cp roundingsat ../../../src/
cd ../..
rm -rf roundingsat-master*

