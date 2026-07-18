import serial


LAST_ERROR = ""


def get_last_error():
    return LAST_ERROR


def STM32Serial(command):
    global LAST_ERROR
    try:
        with serial.Serial('COM4', 9600, timeout=0.5, write_timeout=0.5) as serialcomm:
            serialcomm.write(command.encode())
        LAST_ERROR = ""
        return 1
    except Exception as exc:
        LAST_ERROR = f"COM4 seri haberleşme hatası: {exc}"
        return 0
