#!/bin/bash

# pluck_acc.sh

DIR=$1
cat $DIR/log.txt | fgrep valid_acc | cut -d' ' -f4 > $DIR/valid_acc
cat $DIR/log.txt | fgrep train_acc | cut -d' ' -f4 > $DIR/train_acc