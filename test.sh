#!/bin/bash

for ((i=0; i<40; i++))
do
  ((epoch=$i*5))
  model_name="resnet18_at_"
  epoch="$epoch"
  model_name=$model_name$epoch
  python run_attack.py -c config/hyper_params.yml -m pgd --dataset cifar10 --model $model_name
done
