import telnetlib
import struct

# Connect to the Telnet server on the drone
tn = telnetlib.Telnet("192.168.4.1", 23)

# Define the pitch angle in degrees (e.g. 20 degrees)
pitch_angle = 20

# Convert the pitch angle to the range [-90, 90]
pitch_angle = max(min(pitch_angle, 90), -90)

# Convert the pitch angle to the range [1000, 2000]
pitch_angle = int(1000 + (pitch_angle + 90) * 10)

# Create the MSP command to set the pitch angle
# The command format is: [Header, Size, Command, Data, Checksum]
header = b'$M<'
size = b'\x03'
command = b'\xD0'
data = struct.pack("<H", pitch_angle)
checksum = 0
for b in header + size + command + data:
    checksum ^= b
checksum = struct.pack("<B", checksum)

# Send the MSP command over the Telnet connection
tn.write(header + size + command + data + checksum)

# Close the Telnet connection
tn.close()
