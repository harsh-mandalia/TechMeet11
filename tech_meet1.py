roll=1500
pitch=1500
throttle=1500
yaw=1500
aux1=1500
aux2=1500
aux3=1500
aux4=1500

a=bytearray([0x24, 0x4d, 0x3c, 16, 0xc8, (roll>>8)&0xFF, roll&0xFF, (pitch>>8)&0xFF, pitch&0xFF, (throttle>>8)&0xFF, throttle&0xFF, (yaw>>8)&0xFF, yaw&0xFF, (aux1>>8)&0xFF, aux1&0xFF, (aux2>>8)&0xFF, aux2&0xFF, (aux3>>8)&0xFF, aux3&0xFF, (aux4>>8)&0xFF, aux4&0xFF])

print(a)