import numpy as np

from village.devices.sound_device import sound_device


# generators of sounds, must return numpy arrays
def tone_generator(
    duration: float,
    gain: float,
    frequency: int,
    ramp_time: float,
) -> np.ndarray:
    """
    Generate a single tone with ramping
    Args:
        duration (float): Duration (seconds)
        gain (float): Tone amplitude
        frequency (int): Tone frequency
        ramp_time (float): Ramp up/down time (seconds)
    Returns:
        np.ndarray: Generated sound
    """

    samplerate = sound_device.samplerate

    time = np.linspace(0, duration, int(samplerate * duration))
    # If no frequency specified, return zero array
    if frequency == 0:
        return np.zeros_like(time)
    # Generate tone
    tone = gain * np.sin(2 * np.pi * frequency * time)
    # Calculate ramp points
    sample_rate = 1 / (time[1] - time[0])
    ramp_points = int(ramp_time * sample_rate)
    # Create ramp arrays
    if ramp_points > 0:
        ramp_up = np.linspace(0, 1, ramp_points)
        ramp_down = np.linspace(1, 0, ramp_points)
        # Apply ramps
        tone[:ramp_points] *= ramp_up
        tone[-ramp_points:] *= ramp_down
    return tone


def whitenoise_generator(
    duration: float,
    gain: float,
    ramp_time: float,
) -> np.ndarray:
    """
    Generate white noise with ramping
    Args:
        duration (float): Duration (seconds)
        gain (float): Noise amplitude
        ramp_time (float): Ramp up/down time (seconds)
    Returns:
        np.ndarray: Generated sound
    """

    samplerate = sound_device.samplerate

    time = np.linspace(0, duration, int(samplerate * duration))
    # Generate noise
    noise = gain * np.random.uniform(-1, 1, int(samplerate * duration))
    # Calculate ramp points
    sample_rate = 1 / (time[1] - time[0])
    ramp_points = int(ramp_time * sample_rate)
    # Create ramp arrays
    if ramp_points > 0:
        ramp_up = np.linspace(0, 1, ramp_points)
        ramp_down = np.linspace(1, 0, ramp_points)
        # Apply ramps
        noise[:ramp_points] *= ramp_up
        noise[-ramp_points:] *= ramp_down
    return noise


# calibration sounds, the only arguments are duration and gain
def whitenoise(duration: float, gain: float) -> np.ndarray:
    return whitenoise_generator(duration=duration, gain=gain, ramp_time=0.005)


def tone_600(duration: float, gainer: float) -> np.ndarray:
    return tone_generator(duration=duration, gain=gainer, frequency=600, ramp_time=0.005)


def tone_1000(duration: float, gain: float) -> np.ndarray:
    return tone_generator(duration=duration, gain=gain, frequency=1000, ramp_time=0.005)


def tone_5000(duration: float, gain: float) -> np.ndarray:
    return tone_generator(duration=duration, gain=gain, frequency=5000, ramp_time=0.005)


def tone_10000(duration: float, gain: float) -> np.ndarray:
    return tone_generator(
        duration=duration, gain=gain, frequency=10000, ramp_time=0.005
    )


def tone_20000(duration: float, gain: float) -> np.ndarray:
    return tone_generator(
        duration=duration, gain=gain, frequency=20000, ramp_time=0.005
    )


sound_calibration_functions = [
    whitenoise,
    tone_600,
    tone_1000,
    tone_5000,
    tone_10000,
    tone_20000,
]
