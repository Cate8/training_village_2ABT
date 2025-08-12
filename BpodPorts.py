from village.classes.task import Event, Output

class BpodPorts:
    __slots__ = (
        "valve_l_time",
        "valve_l_reward",
        "valve_r_time",
        "valve_r_reward",
        "LED_l_on",
        "LED_c_on",
        "LED_r_on",
        "left_poke",
        "center_poke",
        "right_poke"
    )

    def __init__(self, n_box, water_calibration, settings):
        if n_box == "9":
            self.valve_l_time = water_calibration.get_valve_time(port = 2, volume = settings.volume)
            self.valve_l_reward = Output.Valve2

            self.valve_r_time = water_calibration.get_valve_time(port = 5, volume = settings.volume)
            self.valve_r_reward = Output.Valve5

            self.LED_l_on = (Output.PWM2, settings.led_intensity)
            self.LED_c_on = (Output.PWM3, settings.led_intensity)
            self.LED_r_on = (Output.PWM5, settings.led_intensity)

            self.left_poke = Event.Port2In 
            self.center_poke = Event.Port3In
            self.right_poke = Event.Port5In 

        elif n_box == "12":
            self.valve_l_time = water_calibration.get_valve_time(port = 7, volume = settings.volume)
            self.valve_r_time = water_calibration.get_valve_time(port = 1, volume = settings.volume)
           
            self.valve_l_reward = Output.Valve7
            self.valve_r_reward = Output.Valve1 

            self.LED_l_on = (Output.PWM7, settings.led_intensity)
            self.LED_c_on = (Output.PWM4, settings.led_intensity)
            self.LED_r_on = (Output.PWM1, settings.led_intensity)

            self.left_poke = Event.Port7In 
            self.center_poke = Event.Port4In
            self.right_poke = Event.Port1In
        elif n_box == "11":
            self.valve_l_time = water_calibration.get_valve_time(port = 7, volume = settings.volume)
            self.valve_r_time = water_calibration.get_valve_time(port = 1, volume = settings.volume)
           
            self.valve_l_reward = Output.Valve7
            self.valve_r_reward = Output.Valve1 

            self.LED_l_on = (Output.PWM7, settings.led_intensity)
            self.LED_c_on = (Output.PWM4, settings.led_intensity)
            self.LED_r_on = (Output.PWM1, settings.led_intensity)

            self.left_poke = Event.Port7In 
            self.center_poke = Event.Port4In
            self.right_poke = Event.Port1In
        elif n_box == "8":
            self.valve_l_time = water_calibration.get_valve_time(port = 7, volume = settings.volume)
            self.valve_r_time = water_calibration.get_valve_time(port = 1, volume = settings.volume)
           
            self.valve_l_reward = Output.Valve7
            self.valve_r_reward = Output.Valve1 

            self.LED_l_on = (Output.PWM7, settings.led_intensity)
            self.LED_c_on = (Output.PWM4, settings.led_intensity)
            self.LED_r_on = (Output.PWM1, settings.led_intensity)

            self.left_poke = Event.Port7In 
            self.center_poke = Event.Port4In
            self.right_poke = Event.Port1In 