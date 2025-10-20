import random
from BpodPorts import BpodPorts
from village.custom_classes.task import Event, Output, Task


class Port_test(Task):

    def __init__(self):
        super().__init__()

        self.info = """
Testing ports, photogates and pumps
----------------------------------------------------------------
it's designed to test valves, leds and photogates and to charge 
water in the tubes if it's needed.
"""

    def start(self):
        self.side = random.choice(["left", "right"])

        self.ports = BpodPorts(
            n_box=self.system_name,
            water_calibration=self.water_calibration,
            sound_calibration=self.sound_calibration,
            settings=self.settings
        )
        # counters
        self.trial_count = 0
        # to keep track of the number of trials on the same side
        self.same_side_count = self.settings.trials_with_same_side 
        self.reward_drunk = 0

        #### CREATING STATE MACHINE, ADDING STATES, SENDING AND RUNNING ####
        
    def create_trial(self):
        
        self.bpod.add_state(
            state_name='waiting',
            state_timer=10,
            state_change_conditions={Event.Tup: 'exit', 
                                      self.ports.left_poke: 'left',
                                      self.ports.center_poke: 'center', 
                                      self.ports.right_poke: 'right'
                                    },
            output_actions=[]
            )

        self.bpod.add_state(
            state_name='left',
            state_timer = 0,
            state_change_conditions={self.ports.left_poke_out: 'waiting',
                                     },
            output_actions=[self.ports.LED_l_on, self.ports.valve_l_reward]
            )
        
        self.bpod.add_state(
            state_name='center',
            state_timer = 0,
            state_change_conditions={self.ports.center_poke_out: 'waiting',
                                     },
            output_actions=[self.ports.LED_c_on]
            )

        self.bpod.add_state(
            state_name='right',
            state_timer = 0,
            state_change_conditions={self.ports.right_poke_out: 'waiting'},
            output_actions=[self.ports.LED_r_on, self.ports.valve_r_reward])
    

    def after_trial(self):
        pass

    def close(self):
        pass
    



