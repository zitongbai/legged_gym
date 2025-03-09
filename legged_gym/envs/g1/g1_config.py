from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class G1Cfg(LeggedRobotCfg):
    class env(LeggedRobotCfg.env):
        num_envs = 4096
        num_observations = 92  # TODO
        num_privileged_obs = 95
        num_actions = 27 # TODO
        # num_observations = 47
        # num_privileged_obs = 50
        # num_actions = 12
        
    class terrain(LeggedRobotCfg.terrain):
        mesh_type = 'plane'
        measure_heights = False
        
    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.8]   # x,y,z [m]
        default_joint_angles = {
            'left_hip_pitch_joint': -0.1,
            'left_hip_roll_joint': 0.0,
            'left_hip_yaw_joint': 0.0,
            'left_knee_joint': 0.3,
            'left_ankle_pitch_joint': -0.2,
            'left_ankle_roll_joint': 0.0,
            'right_hip_pitch_joint': -0.1,
            'right_hip_roll_joint': 0.0,
            'right_hip_yaw_joint': 0.0,
            'right_knee_joint': 0.3,
            'right_ankle_pitch_joint': -0.2,
            'right_ankle_roll_joint': 0.0,
            'waist_yaw_joint': 0.0,
            'left_shoulder_pitch_joint': 0.3,
            'left_shoulder_roll_joint': 0.3,
            'left_shoulder_yaw_joint': 0.0,
            'left_elbow_joint': 0.9,
            'left_wrist_roll_joint': 0.0,
            'left_wrist_pitch_joint': 0.0,
            'left_wrist_yaw_joint': 0.0,
            'right_shoulder_pitch_joint': 0.3,
            'right_shoulder_roll_joint': -0.3,
            'right_shoulder_yaw_joint': 0.0,
            'right_elbow_joint': 0.9,
            'right_wrist_roll_joint': 0.0,
            'right_wrist_pitch_joint': 0.0,
            'right_wrist_yaw_joint': 0.0,
        }
        
    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        control_type = 'P'
          # PD Drive parameters:
        stiffness = {
                        'hip_yaw': 100,
                        'hip_roll': 100,
                        'hip_pitch': 100,
                        'knee': 150,
                        'ankle': 40,
                        'waist_yaw_joint': 100,
                        'shoulder_pitch': 50,
                        'shoulder_roll': 50,
                        'shoulder_yaw': 50,
                        'elbow': 50,
                        'wrist_roll': 20,
                        'wrist_pitch': 20,
                        'wrist_yaw': 20,
                     }  # [N*m/rad]
        damping = {  
                        'hip_yaw': 2,
                        'hip_roll': 2,
                        'hip_pitch': 2,
                        'knee': 4,
                        'ankle': 2,
                        'waist_yaw_joint': 2,
                        'shoulder_pitch': 2,
                        'shoulder_roll': 2,
                        'shoulder_yaw': 2,
                        'elbow': 2,
                        'wrist_roll': 2,
                        'wrist_pitch': 2,
                        'wrist_yaw': 2,
                     }  # [N*m/rad]  # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/g1/g1_29dof_lock_waist_rev_1_0.urdf'
        # file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/g1/g1_12dof.urdf'
        name = "g1"
        foot_name = "ankle_roll"
        penalize_contacts_on = ["hip", "knee"]
        terminate_after_contacts_on = ["pelvis", "torso_link"]
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = False # Some .obj meshes must be flipped from y-up to z-up
        upper_dof_name = ["shoulder", "elbow", "wrist", "waist"]
        hip_dof_name = ["hip_roll", "hip_yaw"]
        leg_dof_name = ["hip", "knee", "ankle"]
    
    class rewards( LeggedRobotCfg.rewards ):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.78
        clearance_height_target = 0.09
        feet_swing_height = 0.08
        only_positive_rewards = True

        class scales( LeggedRobotCfg.rewards.scales ):
            tracking_lin_vel = 1.0
            tracking_ang_vel = 1.5
            lin_vel_z = -2.0
            ang_vel_xy = -0.05
            orientation = -1.0
            base_height = -10.0
            dof_acc = -2.5e-7
            dof_vel = -1e-3
            feet_air_time = 2.0
            collision = 0.0
            action_rate = -0.01
            dof_pos_limits = -5.0
            alive = 0.15
            hip_pos = -1.0
            contact_no_vel = -0.2
            feet_swing_height = -20.0
            contact = 0.18
            upper_dof = -0.1
            
            # termination = -10.0
            # stand_still = -0.4
            # torques = -1e-7
        
    class domain_rand(LeggedRobotCfg.domain_rand):
        randomize_friction = True
        friction_range = [0.1, 1.25]
        randomize_base_mass = True
        added_mass_range = [-1., 3.]
        push_robots = True
        push_interval_s = 5
        max_push_vel_xy = 1.5
        

class G1CfgPPO( LeggedRobotCfgPPO ):
    class policy:
        init_noise_std = 0.8
        actor_hidden_dims = [32]
        critic_hidden_dims = [32]
        activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        # only for 'ActorCriticRecurrent':
        rnn_type = 'lstm'
        rnn_hidden_size = 64
        rnn_num_layers = 1
        
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01
    class runner( LeggedRobotCfgPPO.runner ):
        policy_class_name = "ActorCriticRecurrent"
        num_steps_per_env = 24 # per iteration
        max_iterations = 2000
        run_name = ''
        experiment_name = 'g1'
        save_interval = 100 # check for potential saves every this many iterations
  