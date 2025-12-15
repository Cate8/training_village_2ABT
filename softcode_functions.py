import time

from sound_functions import sound_device, whitenoise_generator
from village.manager import manager
from village.devices.camera import cam_box


def function1():
    print("play sound")
    sound_left = whitenoise_generator(0.5, manager.task.ports.sound_gain_left, 0.005)
    sound_right = whitenoise_generator(0.5, manager.task.ports.sound_gain_right, 0.005)
    sound_device.load(left=sound_left, right=sound_right)
    sound_device.play()

def function2():
    cam_box.annotation = "ON"

def function3():
    cam_box.annotation = ""
    


