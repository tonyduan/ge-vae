#!/bin/bash
#python3 -m scripts.test_gf --dataset=community
python3 -m scripts.test_gf --dataset=grid
python3 -m scripts.test_gf --dataset=community_big
python3 -m scripts.test_gf --dataset=grid_big
python3 -m scripts.test_gf --dataset=ego
python3 -m scripts.test_gf --dataset=protein
