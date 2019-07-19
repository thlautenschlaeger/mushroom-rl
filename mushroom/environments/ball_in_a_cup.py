from mushroom.environments.mujoco_raw import MojucoRaw, ObservationType
import numpy as np
import os


class BallInACup(MojucoRaw):

    def __init__(self):
        xml_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "xml", "ball-in-a-cup.xml")
        action_spec = [("wam/base_yaw_joint", -150., 150.), ("wam/shoulder_pitch_joint", -125., 125.),
                       ("wam/shoulder_yaw_joint", -40., 40,), ("wam/elbow_pitch_joint", -60., 60.),
                       ("wam/wrist_yaw_joint", -5., 5.), ("wam/wrist_pitch_joint", -5., 5.),
                       ("wam/palm_yaw_joint", -2., 2.)]

        observation_spec = [("wam/base_yaw_joint", ObservationType.JOINT_POS),
                            ("wam/base_yaw_joint", ObservationType.JOINT_VEL),
                            ("wam/shoulder_pitch_joint", ObservationType.JOINT_POS),
                            ("wam/shoulder_pitch_joint", ObservationType.JOINT_VEL),
                            ("wam/shoulder_yaw_joint", ObservationType.JOINT_POS),
                            ("wam/shoulder_yaw_joint", ObservationType.JOINT_VEL),
                            ("wam/elbow_pitch_joint", ObservationType.JOINT_POS),
                            ("wam/elbow_pitch_joint", ObservationType.JOINT_VEL),
                            ("wam/wrist_yaw_joint", ObservationType.JOINT_POS),
                            ("wam/wrist_yaw_joint", ObservationType.JOINT_VEL),
                            ("wam/wrist_pitch_joint", ObservationType.JOINT_POS),
                            ("wam/wrist_pitch_joint", ObservationType.JOINT_VEL),
                            ("wam/palm_yaw_joint", ObservationType.JOINT_POS),
                            ("wam/palm_yaw_joint", ObservationType.JOINT_VEL),
                            ("ball", ObservationType.BODY_POS),
                            ("ball", ObservationType.BODY_VEL)]

        additional_data_spec = [("ball_pos", "ball", ObservationType.BODY_POS),
                                ("goal_pos", "cup_goal_final", ObservationType.SITE_POS)]

        collision_groups = [("ball", ["ball_geom"]),
                            ("robot", ["cup_geom1", "cup_geom2", "wrist_palm_link_convex_geom",
                                       "wrist_pitch_link_convex_decomposition_p1_geom",
                                       "wrist_pitch_link_convex_decomposition_p2_geom",
                                       "wrist_pitch_link_convex_decomposition_p3_geom",
                                       "wrist_yaw_link_convex_decomposition_p1_geom",
                                       "wrist_yaw_link_convex_decomposition_p2_geom",
                                       "forearm_link_convex_decomposition_p1_geom",
                                       "forearm_link_convex_decomposition_p2_geom"])]

        super().__init__(xml_path, action_spec, observation_spec, 0.9999, 2000, nsubsteps=4,
                         additional_data_spec=additional_data_spec, collision_groups=collision_groups)

        self.init_robot_pos = np.array([0.0, 0.58760536, 0.0, 1.36004913, 0.0, -0.32072943, -1.57])
        self.p_gains = np.array([200, 300, 100, 100, 10, 10, 2.5])
        self.d_gains = np.array([7, 15, 5, 2.5, 0.3, 0.3, 0.05])

    def reward(self, state, action, next_state):
        dist = self.read_data("goal_pos") - self.read_data("ball_pos")
        return 1. if np.linalg.norm(dist) < 0.05 else 0.

    def is_absorbing(self, state):
        dist = self.read_data("goal_pos") - self.read_data("ball_pos")
        return np.linalg.norm(dist) < 0.05 or self.check_collision("ball", "robot")

    def setup(self):
        # Copy the initial position after the reset
        init_pos = self.sim.data.qpos.copy()
        init_vel = np.zeros_like(init_pos)

        # Reset the system and the set the intial robot position
        self.sim.data.qpos[:] = init_pos
        self.sim.data.qvel[:] = init_vel
        self.sim.data.qpos[0:7] = self.init_robot_pos

        # Do one simulation step to compute the new position of the goal_site
        self.sim.step()

        self.sim.data.qpos[:] = init_pos
        self.sim.data.qvel[:] = init_vel
        self.sim.data.qpos[0:7] = self.init_robot_pos
        self.write_data("ball_pos", self.read_data("goal_pos") - np.array([0., 0., 0.329]))

        # Stabilize the system around the initial position using a PD-Controller
        for i in range(0, 500):
            self.sim.data.qpos[7:] = 0.
            self.sim.data.qvel[7:] = 0.
            self.sim.data.qpos[7] = -0.2
            cur_pos = self.sim.data.qpos[0:7].copy()
            cur_vel = self.sim.data.qvel[0:7].copy()
            trq = self.p_gains * (self.init_robot_pos - cur_pos) + self.d_gains * (
                    np.zeros_like(self.init_robot_pos) - cur_vel)
            self.sim.data.qfrc_applied[0:7] = trq
            self.sim.step()

        # Now simulate for more time-steps without resetting the position of the first link of the rope
        for i in range(0, 500):
            cur_pos = self.sim.data.qpos[0:7].copy()
            cur_vel = self.sim.data.qvel[0:7].copy()
            trq = self.p_gains * (self.init_robot_pos - cur_pos) + self.d_gains * (
                    np.zeros_like(self.init_robot_pos) - cur_vel)
            self.sim.data.qfrc_applied[0:7] = trq
            self.sim.step()
