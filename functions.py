from village.devices.sound_device import sound_device, tone_generator


def function1():
    print("Function 1")


def function33():
    sound = tone_generator(1, 0.01, 600, 0.005, 192000)
    sound_device.load(sound)
    sound_device.play()
