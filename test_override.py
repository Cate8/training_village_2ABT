from village.classes.task import Event, Output, Task


class Test_override(Task):
    def __init__(self):
        super().__init__()

        self.info = """

        Test overrides
        -------------------

        All the events and outputs available in the Bpod.
        """

    def start(self):
        pass

    def create_trial(self):

        self.bpod.add_state(
            state_name="one",
            state_timer=0,
            state_change_conditions={Event.Tup: "two"},
            output_actions=[Output.SoftCode10],
        )

        self.bpod.add_state(
            state_name="two",
            state_timer=0,
            state_change_conditions={Event.BNC1Low: "three"},
            output_actions=[],
        )

        self.bpod.add_state(
            state_name="three",
            state_timer=0,
            state_change_conditions={Event.Tup: "four"},
            output_actions=[Output.SoftCode11],
        )

        self.bpod.add_state(
            state_name="four",
            state_timer=0,
            state_change_conditions={Event.Port1In: "five"},
            output_actions=[],
        )

        self.bpod.add_state(
            state_name="five",
            state_timer=0,
            state_change_conditions={Event.Port1Out: "exit"},
            output_actions=[],
        )

    def after_trial(self):
        pass

    def close(self):
        pass
