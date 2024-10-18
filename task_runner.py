from village.classes.subject import Subject

from follow_the_light import FollowTheLight
from trial_plotter import TrialPlotter
from virtual_mouse import VirtualMouse


def main():
    # Create an instance of the task
    ftl_task = FollowTheLight()

    # Set the number of trials
    ftl_task.maximum_number_of_trials = 100

    # Activate a virtual mouse and let it know about the bpod
    ftl_task.virtual_mouse = VirtualMouse(ftl_task.bpod)

    print(ftl_task.virtual_mouse)
    # Change how fast the mouse learns
    ftl_task.virtual_mouse.learning_rate = 0.005

    # Increase the speed of the task and virtual mouse
    SPEED = 500
    ftl_task.speed = SPEED
    ftl_task.virtual_mouse.speed = SPEED

    # Use an online plotter to display the results
    ftl_task.plotter = TrialPlotter()

    # Create a subject
    subject = Subject("test_subject")

    # Run the task
    ftl_task.run(subject)

    # Close the bpod and save data
    ftl_task.disconnect_and_save()


if __name__ == "__main__":
    main()
