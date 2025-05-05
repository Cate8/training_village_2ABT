import time

from sound_functions import sound_device, whitenoise_generator
from village.manager import get_task

task = get_task()


def function1():
    print("play sound")
    # sound = whitenoise_generator(1, 0.1, 0.005)
    # sound_device.load(left=sound, right=sound)
    # sound_device.play()


