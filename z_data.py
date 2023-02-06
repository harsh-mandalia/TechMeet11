import cv2
import numpy as np
import telnetlib
import time
import matplotlib.pyplot as plt
import pandas as pd
from csv import writer
import os
import struct

host = "192.168.4.1"
# host = "192.168.0.141" 
port = "23"

tn = telnetlib.Telnet(host, port)
print("connected")

def add_checksum(cmd):
    checksum=0
    for b in cmd[3:]:
        checksum ^= b
    cmd.append(checksum)


cmd=bytearray([0x24, 0x4d, 0x3c, 0, 0x6d])
add_checksum(cmd)
while(1):
    tn.write(bytes(cmd))
    #out=tn.read_some()
    out=tn.read_some()
    new=out.decode('utf-8', errors='replace')
    numbers = [ord(c) for c in new]
    numbers = list(out)[4:]
    if len(numbers)>0:
        fmt = '!i' + 'h' * (len(numbers) - 1)
        packed_data = struct.pack(fmt, numbers[0], *numbers[1:])
        unpacked_data = struct.unpack(fmt, packed_data)

        print(packed_data)
        print(unpacked_data)

    # print(out)
    # out=tn.read_until(b'\r')
    # z=struct.unpack('<i',out)

    # for i in out:
        # print(ord(i),end =" ")
    # print()
    # print(new)
    # time.sleep(0.01)


    # print(tn.read_until(b'0x3e'))
    # print(1)
    # print(tn.read_very_eager())
    
    # print()

tn.close()