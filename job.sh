#!/bin/sh
python3 write_tfrecords.py
python3 train_main.py
python3 write_tfrecords.py
