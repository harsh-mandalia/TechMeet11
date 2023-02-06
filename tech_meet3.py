import telnetlib
import time
import functools
from functools import reduce
import struct
import numpy as np
from time import sleep
host = "192.168.4.1"  # Replace with the IP address of your PLUTO drone
port = "23"

tn = telnetlib.Telnet(host, port)

def convert_UINT16(x):
    y = struct.pack('>H',x)
    #print(y)
    return y

def add_checksum(cmd):
    checksum=0
    for b in cmd[3:]:
        checksum ^= b
    cmd.append(checksum)

roll=1500
pitch=1500
throttle=1000
yaw=1500
aux1=901
aux2=901
aux3=901
aux4=900

# cmd1=bytearray([0x24, 0x4d, 0x3c, 16, 0xc8, (roll>>8)&0xFF, roll&0xFF, (pitch>>8)&0xFF, pitch&0xFF, (throttle>>8)&0xFF, throttle&0xFF, (yaw>>8)&0xFF, yaw&0xFF, (aux1>>8)&0xFF, aux1&0xFF, (aux2>>8)&0xFF, aux2&0xFF, (aux3>>8)&0xFF, aux3&0xFF, (aux4>>8)&0xFF, aux4&0xFF])
cmd1=bytearray([0x24, 0x4d, 0x3c, 16, 0xc8, (roll)&0xFF, (roll>>8)&0xFF, (pitch)&0xFF, (pitch>>8)&0xFF, (throttle)&0xFF, (throttle>>8)&0xFF, (yaw)&0xFF, (yaw>>8)&0xFF, (aux1)&0xFF, (aux1>>8)&0xFF, (aux2)&0xFF, (aux2>>8)&0xFF, (aux3)&0xFF, (aux3>>8)&0xFF, (aux4)&0xFF, (aux4>>8)&0xFF])
cmd2=bytearray([0x24, 0x4d, 0x3c, 12, 0xd9, 0, 1])
cmd3="$M<"+chr(12)+chr(0xd9)+chr(0)+chr(1)+chr(12^0xd9^0^1)
cmd4 = bytearray([
    0x24,
    0x4d,
    0x3c,
    16,
    0xc8,
    0x05, #1
    0xdc,
    0x05, #2
    0xdc,
    0x03, #3
    0xe8,
    0x05, #4
    0xdc,
    0x00, #5
    0x00,
    0x00, #6
    0x00,
    0x00, #7
    0x00,
    0x05, #8
    0xdc
])

cmd5 = bytearray([
    0x24,
    0x4d,
    0x3c,
    0x02,
    0xd9,
    0x00,
    0x01
])

# z = convert_UINT16(1)
# cmd=bytearray([np.uint16(int(0x24)), np.uint16(int(0x4d)), np.uint16(int(0x3c)), np.uint16(int(2)), np.uint16(int(0xd9)), np.uint16(int(1))])
# cmd += z
# cmd=bytearray([0x24, 0x4d, 0x3e, 0x06, 0x6c])
#Append checksum to command
#print(checksum)
# cmd.append(np.uint16(int(checksum)))
#checksum_ = convert_UINT16(checksum)
#cmd += checksum_

add_checksum(cmd1)
add_checksum(cmd2)
add_checksum(cmd4)
add_checksum(cmd5)

str__ = '+++AT GPIO12 1'
# str__ = '+++AT'
byt1 = bytes(str__ , 'utf-8')

# tn.write(byt1)

tn.write(bytes(cmd1))
print(bytes(cmd1))
print(tn.read_some())
# while(1):
#     tn.write(bytes(cmd1))
#     print(bytes(cmd1))
#     print(tn.read_some())
#     sleep(0.1)


# tn.write(bytes(cmd2))
# print(bytes(cmd2))

# tn.write(bytes(cmd3, 'utf-8'))
# print(bytes(cmd3, 'utf-8'))
# tn.read_some()

# print(byt1)
print(tn.read_some())
#attitude_data = tn.read_until(byt)
#print(attitude_data)
#roll,pitch,yaw = struct.unpack('<hhh',attitude_data[3:9])
#print("Roll:",roll)  # Print the response from the drone
tn.close()