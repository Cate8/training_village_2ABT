from village.classes.task import Event, Output, Task


class Test_Speed_Data(Task):
    def __init__(self):
        super().__init__()

        self.info = """

        Test speed data
        -------------------
        """

    def start(self):
        pass

    def create_trial(self):

        if self.current_trial < 1000:

            self.bpod.add_state(
                state_name="one",
                state_timer=0.05,
                state_change_conditions={Event.Tup: "two"},
                output_actions=[(Output.PWM1, 100)],
            )

            self.bpod.add_state(
                state_name="two",
                state_timer=0.05,
                state_change_conditions={Event.Tup: "exit"},
                output_actions=[],
            )

        else:
            self.bpod.add_state(
                state_name="one",
                state_timer=0.05,
                state_change_conditions={Event.Tup: "three"},
                output_actions=[(Output.PWM2, 100)],
            )

            self.bpod.add_state(
                state_name="three",
                state_timer=0.05,
                state_change_conditions={Event.Tup: "exit"},
                output_actions=[],
            )

    def after_trial(self):
        val = len(self.bpod.session.trials)
        self.register_value("test", val)
        for i in range(100):
            self.register_value(f"test_{i}", i)

    def close(self):
        pass
