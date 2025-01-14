import time

from village.devices.sound_device import sound_device, tone_generator


def function1():
    print("Function 1")

def function2():
    start_time = time.time()
    sound = tone_generator(1, 0.05, 600, 0.005, 192000)
    sound_device.load(sound)
    end_time = time.time()
    print("load delay: ", end_time - start_time)

def function33():
    start_time = time.time()
    sound_device.play()
    end_time = time.time()
    print("play delay: ", end_time - start_time)
