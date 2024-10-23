from village.classes.task import Event, Output, Task


class Simple(Task):
    def __init__(self):
        super().__init__()

        self.info = """

        Simple Task
        -------------------

        5 trials of 2 seconds each.
        """

    def start(self):
        print("start")
        print(self.settings.next_task)

    def create_trial(self):
        print("creating trial")
        self.bpod.add_state(
            state_name="one",
            state_timer=1,
            state_change_conditions={Event.Tup: "two"},
            output_actions=[(Output.PWM1, 255)],
        )

        self.bpod.add_state(
            state_name="two",
            state_timer=0,
            state_change_conditions={Event.Tup: "exit"},
            output_actions=[],
        )

    def after_trial(self):
        print("after trial")

    def close(self):
        print("close")
