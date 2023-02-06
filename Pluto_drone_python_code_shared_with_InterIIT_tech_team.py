#!/usr/bin/env python
import time
import math
import numpy as np

### Disclaimar
# Please note that you are requested to use this code for interIIT compitition only.
# Distribution of this code is strictly not allowed. 
# You can modify based on the requirement.

global flag
global quad_initial
global start_time
global current_time
global quad_pos
global load_pos
global quad_vel
global quad_pos_old
global square_function_start_time 
global square_function_current_time

flag = 0

quad_initial	= [ 0, 0, 0 ]
quad_pos		= [ 0, 0, 0 ]
quad_vel		= [ 0, 0, 0 ]


current_time	= 0
start_time 		= time.time()
square_function_start_time = 0.0
square_function_current_time = 0.0

class send_data():
	"""docstring for request_data"""

	def state_feedback(self,data):
		global quad_initial
		global start_time
		global current_time
		global quad_pos
		global quad_vel
		global quad_pos_old
		global quad_vel_old

		quad_pos[0] = 0 # write a code here to get x position of the drone
		quad_pos[1] = 1 # write a code here to get x position of the drone
		quad_pos[2] = 2 # write a code here to get x position of the drone

		if jj == 0:
				quad_pos_old[0] = quad_pos[0]
				quad_pos_old[1] = quad_pos[1]
				quad_pos_old[2] = quad_pos[2]
				jj = 1
		
		quad_vel[0] = (quad_pos[0] - quad_pos_old[0])/0.01
		quad_vel[1] = (quad_pos[1] - quad_pos_old[1])/0.01
		quad_vel[2] = (quad_pos[2] - quad_pos_old[2])/0.01
				
		quad_vel[0] = round(quad_vel[0])
		quad_vel[1] = round(quad_vel[1])
		quad_vel[2] = round(quad_vel[2])

		quad_pos_old[0]   = quad_pos[0]
		quad_pos_old[1]   = quad_pos[1]
		quad_pos_old[2]   = quad_pos[2]

	def arm(self):
		rcRoll=1500
		rcYaw=1500
		rcPitch =1500
		rcThrottle =1000
		rcAUX4 =1500
		plutoIndex = 0
		rcAUX3 	=  1200
		# write a code here to send the above data
		
	def disarm(self):
		rcThrottle =1000
		rcAUX4 = 1200
		plutoIndex = 0
		rcAUX3 	=  1200
		# write a code here to send the above data
	
	def keyboard_key_identifier(self, msg):
		
		# this funnction is to read the keyboard input to arm, disarm, start and stop mission, or aborting the mission
		if key_value == 0:         
			disarm()
		if key_value == 70:
			disarm()
			arm()
		if key_value == 200:
			position_hold()
		if key_value == 210:
			square_traj()

	def position_hold(self):
		global quad_pos
		global quad_vel
		# Define the desired position of the quadcopter

		X_des   = 0
		Y_des   = 0
		Z_des   = 1500
		
		Kp_x = 0.25
		Kp_y = 0.25
		Kp_z = 1.5 

		Kd_x = 0.2
		Kd_y = 0.2
		Kd_z = 0.8 

		rcPitch_command    = 1500 + Kp_x * (X_des - quad_pos[0]) - Kd_x * quad_vel[0]
		rcRoll_command     = 1500 - Kp_y * (Y_des - quad_pos[1]) + Kd_y * quad_vel[1]
		rcThrottle_command = 1600 + Kp_z * (Z_des - quad_pos[2]) - Kd_z * quad_vel[2]
	
		if rcThrottle_command < 1000:
			rcThrottle_command = 1000
		elif rcThrottle_command > 2000:
			rcThrottle_command = 2000

		upper_sat_lim = 2000
		lower_sat_lim = 1000

		if 	rcRoll_command < lower_sat_lim:
			rcRoll_command = lower_sat_lim

		elif rcRoll_command > upper_sat_lim:
			rcRoll_command = upper_sat_lim
	
		if rcPitch_command < lower_sat_lim:
			rcPitch_command = lower_sat_lim

		elif rcPitch_command > upper_sat_lim:
			rcPitch_command = upper_sat_lim

		rcRoll     =  int(rcRoll_command)
		rcThrottle =  int(rcThrottle_command)
		rcPitch    =  int(rcPitch_command)
		# write a code here to send the above data

	def square_traj(self):
		global quad_pos
		global quad_vel
		global start_time
		global current_time
		global flag
		global square_function_start_time 
		global square_function_current_time

		if flag == 0:
			square_function_start_time = time.time()
			flag = 1
		
		square_function_current_time = time.time() - square_function_start_time

		square_initial_x = 750
		square_initial_y = 750
		square_initial_z = 1500

		X_des = 0
		Y_des = 0
		Z_des = 0

		X_1   = square_initial_x
		Y_1   = square_initial_y
		Z_1   = square_initial_z

		X_2   = square_initial_x - 1500
		Y_2   = square_initial_y 
		Z_2   = square_initial_z 

		X_3   = square_initial_x - 1500
		Y_3   = square_initial_y - 1500
		Z_3   = square_initial_z 

		X_4   = square_initial_x
		Y_4   = square_initial_y - 1500 
		Z_4   = square_initial_z 

		time_duration = 2

		if square_function_current_time <= 0.5:
			X_des = X_1
			Y_des = Y_1
			Z_des = Z_1

		elif square_function_current_time <= 0.5 + time_duration:
			X_des = X_2
			Y_des = Y_2
			Z_des = Z_2

		elif square_function_current_time <= 0.5 + 2 * time_duration:
			X_des = X_3
			Y_des = Y_3
			Z_des = Z_3

		elif square_function_current_time <= 0.5 + 3 * time_duration:
			X_des = X_4
			Y_des = Y_4
			Z_des = Z_4

		elif square_function_current_time <= 0.5 + 4 * time_duration:
			X_des = X_1
			Y_des = Y_1
			Z_des = Z_1

		elif square_function_current_time > 0.5 + 4 * time_duration:
			X_des = X_1
			Y_des = Y_1
			Z_des = Z_1
			square_function_start_time = time.time()

		#Tune this gains as per your required response

		Kp_x = 0.2		
		Kp_y = 0.2		
		Kp_z = 1.5		

		Kd_x = 7		
		Kd_y = 7	
		Kd_z = 5

		rcPitch_command    = 1500 + Kp_x * (X_des - quad_pos[0]) - Kd_x * quad_vel[0]
		rcRoll_command     = 1500 - Kp_y * (Y_des - quad_pos[1]) + Kd_y * quad_vel[1]
		rcThrottle_command = 1500 + Kp_z * (Z_des - quad_pos[2]) - Kd_z * quad_vel[2]

		upper_sat_lim = 2000
		lower_sat_lim = 1000

		if 	rcRoll_command < lower_sat_lim:
			rcRoll_command = lower_sat_lim

		elif rcRoll_command > upper_sat_lim:
			rcRoll_command = upper_sat_lim
	
		if rcPitch_command < lower_sat_lim:
			rcPitch_command = lower_sat_lim

		elif rcPitch_command > upper_sat_lim:
			rcPitch_command = upper_sat_lim

		if rcThrottle_command < 1000:
			rcThrottle_command = 1000
		elif rcThrottle_command > 2000:
			rcThrottle_command = 2000

		rcRoll     =  rcRoll_command
		rcPitch    =  rcPitch_command
		rcThrottle =  rcThrottle_command
		# write a code here to send the above data


if __name__ == '__main__':
	while True:
		test = send_data()