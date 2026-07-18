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
        with serial.Serial('COM4', 9600, timeout=0.5, write_timeout=0.5) as serialcomm:
            serialcomm.write(command.encode())
        return 1
    except:
        return 0
    # time.sleep(0.5)

    # print(serialcomm.readline().decode('ascii'))

# serialcomm.close()
