import telnetlib
import time
import functools
from functools import reduce
import struct
import numpy as np
from time import sleep
import pygame

host = "192.168.4.1"  # Replace with the IP address of your PLUTO drone
port = "23"

tn = telnetlib.Telnet(host, port)

def add_checksum(cmd):
    checksum=0
    for b in cmd[3:]:
        checksum ^= b
    cmd.append(checksum)
    
def rc(roll,pitch,throttle, yaw, aux1, aux2, aux3, aux4):
    cmd1=bytearray([0x24, 0x4d, 0x3c, 16, 0xc8, (roll)&0xFF, (roll>>8)&0xFF, (pitch)&0xFF, (pitch>>8)&0xFF, (throttle)&0xFF, (throttle>>8)&0xFF, (yaw)&0xFF, (yaw>>8)&0xFF, (aux1)&0xFF, (aux1>>8)&0xFF, (aux2)&0xFF, (aux2>>8)&0xFF, (aux3)&0xFF, (aux3>>8)&0xFF, (aux4)&0xFF, (aux4>>8)&0xFF])
    add_checksum(cmd1)
    #print(bytes(cmd1))
    return bytes(cmd1)

def set(data):
    cmd1=bytearray([0x24, 0x4d, 0x3c, 2, 0xd9, data, 0])
    add_checksum(cmd1)
    print(bytes(cmd1))
    return bytes(cmd1)


#tn.write(rc(1500,1500,1500,1500,901,901,1500,900))
#tn.write(rc(1500,1500,1000,1500,901,901,1500,1500))
#print(tn.read_some())

#sleep(1)

#tn.write(rc(1500,1500,1800,1500,901,901,901,1500))

#tn.write(set(1))

#sleep(1)
#tn.write(rc(1500,1500,1800,1500,901,901,1500,1500))
#sleep(1)


#sleep(1)
#tn.write(rc(1500,1600,1800,1500,901,901,1500,1500))
#sleep(0.5)
#tn.write(rc(1500,1500,1800,1500,901,901,1500,1500))
#sleep(1)

#tn.write(set(2))
#print(tn.read_some())

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

# initialize pygame
pygame.init()

# create the window
screen = pygame.display.set_mode((400, 300))

# set the title of the window
pygame.display.set_caption("8-key Remote Control")

# define the four variables and their limits
roll = 1500
pitch = 1500
yaw = 1500
throttle = 1000
roll_min = 900
roll_max = 2100
pitch_min = 900
pitch_max = 2100
yaw_min = 900
yaw_max = 2100
throttle_min = 900
throttle_max = 2100

aux1 = 901
aux2 = 901
aux3 = 1500
aux4 = 1500

mode_r = ""
mode_l = ""
mode_o = ""

change = 100

# flag to track if key is pressed
key_up_pressed = False
key_down_pressed = False
key_left_pressed = False
key_right_pressed = False
key_w_pressed = False
key_s_pressed = False
key_a_pressed = False
key_d_pressed = False

tn.write(rc(1500,1500,1000,1500,901,901,1500,1500))
tn.write(rc(1500,1500,1500,1500,901,901,1500,900))
tn.write(rc(1500,1500,1000,1500,901,901,1500,1500))
#tn.write(set(1))
print(tn.read_some())
# run the game loop
running = True
while running:
    aux3 = 1500
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        # check for key down events
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP:
                key_up_pressed = True
            if event.key == pygame.K_DOWN:
                key_down_pressed = True
            if event.key == pygame.K_LEFT:
                key_left_pressed = True
            if event.key == pygame.K_RIGHT:
                key_right_pressed = True
            if event.key == pygame.K_w:
                key_w_pressed = True
            if event.key == pygame.K_s:
                key_s_pressed = True
            if event.key == pygame.K_a:
                key_a_pressed = True
            if event.key == pygame.K_d:
                key_d_pressed = True

            if event.unicode == 'r':
                mode_r = 'ON'
                roll = 1500
                pitch = 1500
                yaw = 1500
                throttle = 1500
                tn.write(rc(roll,pitch,throttle, yaw, aux1, aux2, aux3, 1500))
            elif event.unicode == 'l':
                mode_l = 'ON'
                tn.write(set(2))

            elif event.unicode == 'o':
                mode_o = 'Drone OFF'
                tn.write(rc(1500,1500,1500,1500,901,901,1500,900))

        # check for key up events
        if event.type == pygame.KEYUP:
            if event.key == pygame.K_UP:
                key_up_pressed = False
            if event.key == pygame.K_DOWN:
                key_down_pressed = False
            if event.key == pygame.K_LEFT:
                key_left_pressed = False
            if event.key == pygame.K_RIGHT:
                key_right_pressed = False
            if event.key == pygame.K_w:
                key_w_pressed = False
            if event.key == pygame.K_s:
                key_s_pressed = False
            if event.key == pygame.K_a:
                key_a_pressed = False
            if event.key == pygame.K_d:
                key_d_pressed = False

    # update the variables based on key press
    if key_up_pressed:
        pitch = min(pitch + change, pitch_max)
        #tn.write(rc(roll,pitch,throttle, yaw, aux1, aux2, aux3, aux4))
        #aux3 = 900
    elif key_down_pressed:
        pitch = max(pitch - change, pitch_min)
        #tn.write(rc(roll,pitch,throttle, yaw, aux1, aux2, aux3, aux4))
        #aux3 = 900
    else:
        pitch = 1500
        #tn.write(rc(roll,pitch,throttle, yaw, aux1, aux2, aux3, aux4))
    if key_left_pressed:
        roll = max(roll - change, roll_min)
        #tn.write(rc(roll,pitch,throttle, yaw, aux1, aux2, aux3, aux4))
        #aux3 = 900
    elif key_right_pressed:
        roll = min(roll + change, roll_max)
        #tn.write(rc(roll,pitch,throttle, yaw, aux1, aux2, aux3, aux4))
        #aux3 = 900
    else:
        roll = 1500
        #tn.write(rc(roll,pitch,throttle, yaw, aux1, aux2, aux3, aux4))
        #aux3 = 900
    if key_w_pressed:
        throttle = min(throttle + change, throttle_max)
        #aux3 = 900
        #tn.write(rc(roll,pitch,throttle, yaw, aux1, aux2, aux3, aux4))
    elif key_s_pressed:
        throttle = max(throttle - change, throttle_min)
        #aux3 = 900
        #tn.write(rc(roll,pitch,throttle, yaw, aux1, aux2, aux3, aux4))
    # else:
        # throttle = 1500
        #tn.write(rc(roll,pitch,throttle, yaw, aux1, aux2, aux3, aux4))
    if key_a_pressed:
        yaw = max(yaw - change, yaw_min)
        #tn.write(rc(roll,pitch,throttle, yaw, aux1, aux2, aux3, aux4))
        #aux3 = 900
    elif key_d_pressed:
        yaw = min(yaw + change, yaw_max)
        #tn.write(rc(roll,pitch,throttle, yaw, aux1, aux2, aux3, aux4))
        #aux3 = 900
    else:
        yaw = 1500
        #tn.write(rc(roll,pitch,throttle, yaw, aux1, aux2, aux3, aux4))
    tn.write(rc(roll,pitch,throttle, yaw, aux1, aux2, aux3, aux4))
    # clear the screen
    screen.fill((255, 255, 255))

    # display the current values of the variables
    font = pygame.font.Font(None, 30)
    text = font.render("Roll: {}".format(roll), True, (0, 0, 0))
    screen.blit(text, (10, 10))
    text = font.render("Pitch: {}".format(pitch), True, (0, 0, 0))
    screen.blit(text, (10, 40))
    text = font.render("Yaw: {}".format(yaw), True, (0, 0, 0))
    screen.blit(text, (10, 70))
    text = font.render("Throttle: {}".format(throttle), True, (0, 0, 0))
    screen.blit(text, (10, 100))
    text = font.render("AUX3: {}".format(aux3), True, (0, 0, 0))
    screen.blit(text, (200, 100))
    text = font.render("RESET: {}".format(mode_r), True, (0, 0, 0))
    screen.blit(text, (200, 10))
    text = font.render("LAND: {}".format(mode_l), True, (0, 0, 0))
    screen.blit(text, (200, 40))
    text = font.render("ON/OFF: {}".format(mode_o), True, (0, 0, 0))
    screen.blit(text, (200, 70))
    time.sleep(0.1)
    #tn.write(rc(roll,pitch,throttle, yaw, aux1, aux2, aux3, aux4))
    # update the screen
    pygame.display.update()

# quit pygame and telnet
pygame.quit()
tn.close()