import telnetlib
import time
import functools
from functools import reduce
import struct
import numpy as np
from time import sleep
# host = "192.168.4.1"  # Replace with the IP address of your PLUTO drone
host1 = "192.168.0.175"
port1 = "23"

# host2 = "192.168.0.175"
# port2 = "9060"

tn = telnetlib.Telnet(host1, port1)
# print(tn.read_some())

def add_checksum(cmd):
    checksum=0
    for b in cmd[3:]:
        checksum ^= b
    cmd.append(checksum)
    
def rc(roll,pitch,throttle, yaw, aux1, aux2, aux3, aux4):
    cmd1=bytearray([0x24, 0x4d, 0x3c, 16, 0xc8, (roll)&0xFF, (roll>>8)&0xFF, (pitch)&0xFF, (pitch>>8)&0xFF, (throttle)&0xFF, (throttle>>8)&0xFF, (yaw)&0xFF, (yaw>>8)&0xFF, (aux1)&0xFF, (aux1>>8)&0xFF, (aux2)&0xFF, (aux2>>8)&0xFF, (aux3)&0xFF, (aux3>>8)&0xFF, (aux4)&0xFF, (aux4>>8)&0xFF])
    add_checksum(cmd1)
    print(bytes(cmd1))
    return bytes(cmd1)

def set(data):
    cmd1=bytearray([0x24, 0x4d, 0x3c, 2, 0xd9, data, 0])
    add_checksum(cmd1)
    print(bytes(cmd1))
    return bytes(cmd1)



# roll=1500
# pitch=1500
# throttle=1000
# yaw=1500
# aux1=901
# aux2=901
# aux3=1500
# aux4=1500

# # cmd1=bytearray([0x24, 0x4d, 0x3c, 16, 0xc8, (roll>>8)&0xFF, roll&0xFF, (pitch>>8)&0xFF, pitch&0xFF, (throttle>>8)&0xFF, throttle&0xFF, (yaw>>8)&0xFF, yaw&0xFF, (aux1>>8)&0xFF, aux1&0xFF, (aux2>>8)&0xFF, aux2&0xFF, (aux3>>8)&0xFF, aux3&0xFF, (aux4>>8)&0xFF, aux4&0xFF])
# cmd1=bytearray([0x24, 0x4d, 0x3c, 16, 0xc8, (roll)&0xFF, (roll>>8)&0xFF, (pitch)&0xFF, (pitch>>8)&0xFF, (throttle)&0xFF, (throttle>>8)&0xFF, (yaw)&0xFF, (yaw>>8)&0xFF, (aux1)&0xFF, (aux1>>8)&0xFF, (aux2)&0xFF, (aux2>>8)&0xFF, (aux3)&0xFF, (aux3>>8)&0xFF, (aux4)&0xFF, (aux4>>8)&0xFF])
# cmd2=bytearray([0x24, 0x4d, 0x3c, 2, 0xd9, 2, 0])

# add_checksum(cmd1)
# add_checksum(cmd2)

str__ = '+++AT GPIO12 1'
# str__ = '+++AT'
byt1 = bytes(str__ , 'utf-8')

# tn.write(byt1)
# tn.write(rc(1500,1500,1500,1500,901,901,1500,1500))
tn.write(rc(1500,1500,1500,1500,901,901,1500,900))
print(tn.read_some())
tn.write(rc(1500,1500,1000,1500,901,901,1500,1500))
print(tn.read_some())

# sleep(1)

# tn.write(rc(1500,1500,1800,1500,901,901,901,1500))

# #tn.write(set(1))

# sleep(1)
# tn.write(rc(1500,1500,1800,1500,901,901,1500,1500))
# sleep(1)


# sleep(1)
# tn.write(rc(1500,1600,1800,1500,901,901,1500,1500))
# sleep(0.5)
# tn.write(rc(1500,1500,1800,1500,901,901,1500,1500))
sleep(1)

tn.write(set(2))
print(tn.read_some())

# sleep(1)

# tn.write(rc(1500,1500,1500,1500,901,901,1500,900))
# print(tn.read_some())

# while(1):
#     tn.write(bytes(cmd1))
#     print(bytes(cmd1))
#     print(tn.read_some())
#     sleep(0.1)

# print(byt1)
# print(tn.read_some())
#attitude_data = tn.read_until(byt)
#print(attitude_data)
#roll,pitch,yaw = struct.unpack('<hhh',attitude_data[3:9])
#print("Roll:",roll)  # Print the response from the drone
tn.close()