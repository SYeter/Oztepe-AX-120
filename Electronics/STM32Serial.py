import serial

import time
# try:
#     serialcomm = serial.Serial('COM8', 9600)
# except:
#     pass
# serialcomm.timeout = 0.5

def STM32Serial(command):

    # i = command
    try:
        serialcomm = serial.Serial('COM4', 9600)
        serialcomm.write(command.encode())
        return 1
    except:
        return 0
    # time.sleep(0.5)

    # print(serialcomm.readline().decode('ascii'))

# serialcomm.close()