# -*- coding: utf-8 -*-

import os
import re
import sys

import serial

import time

import chardet


def decode_bytes(data, start, stop):
    return data[start:stop + 1].decode('latin1')


# configure the serial connections (the parameters differs on the device you are connecting to)
ser = serial.Serial(
    # In linux
    # /dev/ttyUSB1
    port='COM10',
    baudrate=9600,
    bytesize=8,


)
data = ser.read()
ser.flush()

while True:
    print(data)

# Reading the data from the serial port. This will be running in an infinite loop.

# 1 = b'\xd6'
# 0 = b'\xf6'
# a = b'0'
# c = b'N'