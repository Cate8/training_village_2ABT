from village.classes.task import Event, Output, Task


class Complete(Task):
    def __init__(self):
        super().__init__()

        self.info = """

        Complete Task
        -------------------

        All the events and outputs available in the Bpod.
        """

    def start(self):
        pass

    def create_trial(self):

        self.bpod.set_global_timer(timer_id=2, timer_duration=1.5)
        self.bpod.set_global_counter(
            counter_number=3, target_event=Event.Port1In, threshold=4
        )

        # EVENT: PortIn and OUTPUT: PWM
        self.bpod.add_state(
            state_name="one",
            state_timer=0,
            state_change_conditions={Event.Port1In: "two"},
            output_actions=[(Output.PWM1, 100)],
        )
        # EVENT: PortOut and OUTPUT: Valve
        self.bpod.add_state(
            state_name="two",
            state_timer=0,
            state_change_conditions={Event.Port1Out: "three"},
            output_actions=[Output.Valve1],
        )
        # EVENT: Tup and OUTPUT: SoftCode
        self.bpod.add_state(
            state_name="three",
            state_timer=1,
            state_change_conditions={Event.Tup: "four"},
            output_actions=[Output.SoftCode1],
        )
        # EVENT: SoftCode and OUTPUT: BNCHigh
        self.bpod.add_state(
            state_name="four",
            state_timer=0,
            state_change_conditions={Event.SoftCode1: "five"},
            output_actions=[Output.BNC1High],
        )
        # EVENT: BNCLow and OUTPUT: Timer
        self.bpod.add_state(
            state_name="five",
            state_timer=0,
            state_change_conditions={Event.BNC1Low: "six"},
            output_actions=[Output.GlobalTimer2Trig],
        )
        # EVENT: Timer and OUTPUT: Nothing
        self.bpod.add_state(
            state_name="six",
            state_timer=0,
            state_change_conditions={Event.GlobalTimer2End: "seven"},
            output_actions=[],
        )
        # EVENT: PortIn and OUTPUT: PWM
        self.bpod.add_state(
            state_name="seven",
            state_timer=0,
            state_change_conditions={Event.Port1In: "eight"},
            output_actions=[(Output.PWM1, 100)],
        )
        # EVENT: Tup, Counter and OUTPUT: Nothing
        self.bpod.add_state(
            state_name="eight",
            state_timer=0.5,
            state_change_conditions={
                Event.Tup: "seven",
                Event.GlobalCounter3End: "exit",
            },
            output_actions=[],
        )

    def after_trial(self):
        # create something to register that changes with each trial
        val = len(self.bpod.session.trials)
        print("in after trial")
        print(self.trial_data)
        self.register_value("test", val)
        self.register_value("test2", "hola")
        self.register_value("test3", 3.14)
        self.register_value("test4", [1, 2.3, 3])
        self.register_value("test5", {"a": 1, "b": 2})
        self.register_value("test6", (1, 2, 3))
        self.register_value("test7", True)
        self.register_value("test8", ["hola", 4])
        print("after register value")
        print(self.trial_data)

    def close(self):
        print("in close")
        print(self.trial_data)
        pass
