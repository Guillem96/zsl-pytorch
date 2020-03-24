#!/bin/bash

kaggle datasets download -d gpiosenka/100-bird-species

mkdir -p birds
unzip 100-bird-species -d birds

rm -rf birds/160\ BIRDS
rm -rf birds/150/consolidated
rm birds/150/BIRDS-224-150-97.73.h5 
rm birds/150/error\ list-\ 97.20.txt 
rm 100-bird-species.zip