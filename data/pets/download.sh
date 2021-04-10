#!/usr/bin/env bash

aria2c -m 0 -s 16 -x 16 https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz
aria2c -m 0 -s 16 -x 16 https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz

tar -xf images.tar.gz
tar -xf annotations.tar.gz
