import telnetlib
import time
import functools
from functools import reduce
import struct
import numpy as np
host = "192.168.4.1"  # Replace with the IP address of your PLUTO drone
port = "23"

tn = telnetlib.Telnet(host, port)
#Define MSP command to set roll and pitch angle
# data=[1650]*8
# cmd = bytearray([0x24, 0x4D, 0x3C, 0x10, 0xC8, 0x00, 0x00, 0x00, 0x00])

# #Convert roll and pitch values to bytes
# roll = int(1500)
# pitch = int(2000)
# cmd[5] = roll & 0xFF
# cmd[6] = roll >> 8
# cmd[7] = pitch & 0xFF
# cmd[8] = pitch >> 8

def convert_UINT16(x):
    y = struct.pack('>H',x)
    #print(y)
    return y

roll=1500
pitch=1500
throttle=1500
yaw=1500
aux1=1500
aux2=1500
aux3=1500
aux4=1500

# cmd=bytearray([0x24, 0x4d, 0x3c, 16, 0xc8, (roll>>8)&0xFF, roll&0xFF, (pitch>>8)&0xFF, pitch&0xFF, (throttle>>8)&0xFF, throttle&0xFF, (yaw>>8)&0xFF, yaw&0xFF, (aux1>>8)&0xFF, aux1&0xFF, (aux2>>8)&0xFF, aux2&0xFF, (aux3>>8)&0xFF, aux3&0xFF, (aux4>>8)&0xFF, aux4&0xFF])

z = convert_UINT16(1)
cmd=bytearray([np.uint16(int(0x24)), np.uint16(int(0x4d)), np.uint16(int(0x3c)), np.uint16(int(2)), np.uint16(int(0xd9)), np.uint16(int(1))])
# cmd += z
# Calculate checksum
checksum = 0
# cmd=bytearray([0x24, 0x4d, 0x3e, 0x06, 0x6c])
for b in cmd[3:]:
    checksum ^= b
#Append checksum to command
#print(checksum)
cmd.append(np.uint16(int(checksum)))
#checksum_ = convert_UINT16(checksum)
#cmd += checksum_

str__ = '+++AT GPIO12 1'
#str__ = '+++AT'
byt1 = bytes(str__ , 'utf-8')

# tn.write(byt1)
tn.write(bytes(cmd))
print(bytes(cmd))
# print(byt1)
# print(tn.read_all())
#attitude_data = tn.read_until(byt)
#print(attitude_data)
#roll,pitch,yaw = struct.unpack('<hhh',attitude_data[3:9])
#print("Roll:",roll)  # Print the response from the drone
tn.close()