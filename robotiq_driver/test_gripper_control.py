from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface
from robotiq_gripper_control import RobotiqGripper

rtde_c = RTDEControlInterface("192.168.1.254")
rtde_r = RTDEReceiveInterface("192.168.1.254")
gripper = RobotiqGripper(rtde_c)
gripper.activate()  # returns to previous position after activation
gripper.set_force(0)  # from 0 to 100 %
gripper.set_speed(0)  # from 0 to 100 %
gripper.close()
gripper.open()

rtde_c.stopScript()
rtde_c.disconnect()
