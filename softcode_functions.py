import time

from village.devices.sound_device import sound_device, tone_generator
from village.manager import manager


def function1():
    print("Function 1")
    time.sleep(2)
    manager.task.bpod.send_softcode(1)


def function2():
    start_time = time.time()
    sound = tone_generator(1, 0.05, 600, 0.005, 192000)
    sound_device.load(sound)
    end_time = time.time()
    print("load delay: ", end_time - start_time)


def function10():
    # to test overriding outputs
    manager.task.bpod.manual_override_output(("PWM1", 255))  # funciona
    time.sleep(1)
    manager.task.bpod.manual_override_output(("PWM1", 0))  # funciona
    time.sleep(1)
    manager.task.bpod.manual_override_output("Valve1")
    time.sleep(1)
    manager.task.bpod.manual_override_output("Valve1Off")
    time.sleep(1)
    manager.task.bpod.manual_override_output("BNC1High")  # funciona
    time.sleep(1)
    manager.task.bpod.manual_override_output("BNC1Low")  # funciona


def function11():
    # to test overriding inputs
    manager.task.bpod.manual_override_input("Port1In")  # funciona
    time.sleep(1)
    manager.task.bpod.manual_override_input("Port1Out")  # funciona
    time.sleep(1)


def function33():
    start_time = time.time()
    sound_device.play()
    end_time = time.time()
    print("play delay: ", end_time - start_time)
