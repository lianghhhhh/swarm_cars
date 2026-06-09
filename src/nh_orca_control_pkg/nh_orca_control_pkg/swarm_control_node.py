import math
import numpy as np
import rvo2  # Matches the pyrvo2 wrapper bindings used by NH-ORCA
from rclpy.node import Node
from nav_msgs.msg import Odometry
from tf2_msgs.msg import TFMessage
from geometry_msgs.msg import Twist

GOAL_THRESHOLD   = 0.3
MAX_ANGULAR      = 3.0        # Isaac maxAngularSpeed
MAX_LINEAR       = 1.2        # Isaac maxLinearSpeed

# NH-ORCA Kinematic Configuration parameters (Matching your Turtlebot reference math)
EFFECTIVE_D      = 0.10      # Shifting control point ahead of the non-holonomic axis
WHEEL_L          = 0.235       # Track width / distance between wheels


class SwarmControlNode(Node):
    def __init__(self, num_agents=11):
        super().__init__('swarm_control_node')
        self.num_agents = num_agents
        
        # 1. Initialize RVO2 Simulator
        # Parameter defaults aligned with NH-ORCA turtlebot setup
        # Note: radius assigned as (radius + effective_distance) as per the source defaults
        sim_radius = 0.3 + EFFECTIVE_D
        # Params: timeStep, neighborDist, maxNeighbors, timeHorizon, timeHorizonObst, radius, maxSpeed
        self.rvo_sim = rvo2.PyRVOSimulator(0.05, 5.0, 5, 1.0, 2.0, sim_radius, MAX_LINEAR)
        
        self.agent_rvo_ids = []
        self.goals = [(10.0, 10.0)] * num_agents 
        self.has_calculated_goals = False
        
        self.publishers_ = []
        self.odom_subscribers_ = []
        self.tf_subscribers_ = []
        self.current_poses = {}      # Store latest (x, y, yaw) from odometry
        self.current_twists = {}     # Store latest (linear_x, angular_z) for effective velocity calculation
        self.initial_poses = {}      # Store initial global poses
        
        for i in range(self.num_agents):
            agent_id = self.rvo_sim.addAgent((0.0, 0.0))
            self.agent_rvo_ids.append(agent_id)
            
            pub = self.create_publisher(Twist, f'/car_{i}/cmd_vel', 10)
            self.publishers_.append(pub)
            
            sub = self.create_subscription(Odometry, f'/car_{i}/odom', 
                                           lambda msg, idx=i: self.odom_callback(msg, idx), 10)
            self.odom_subscribers_.append(sub)

            tf_sub = self.create_subscription(TFMessage, f'/car_{i}/tf', 
                                              lambda msg, idx=i: self.tf_callback(msg, idx), 10)
            self.tf_subscribers_.append(tf_sub)

        # Control Loop (runs at 20Hz matching your 0.05s timeStep)
        self.timer = self.create_timer(0.05, self.control_loop)

    def odom_callback(self, msg, agent_index):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        
        q = msg.pose.pose.orientation
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        yaw = yaw + math.pi  # Match Isaac's orientation convention
        yaw = self.normalize_angle(yaw)
        
        self.current_poses[agent_index] = (x, y, yaw)
        # Store live twist to feed back into get_effective_vel tracking
        self.current_twists[agent_index] = (msg.twist.twist.linear.x, msg.twist.twist.angular.z)

    def tf_callback(self, msg, agent_index):
        if agent_index not in self.initial_poses and msg.transforms:
            transform = msg.transforms[0].transform
            x = transform.translation.x
            y = transform.translation.y
            q = transform.rotation
            siny_cosp = 2 * (q.w * q.z + q.x * q.y)
            cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
            yaw = math.atan2(siny_cosp, cosy_cosp)
            yaw = self.normalize_angle(yaw)
            self.initial_poses[agent_index] = (x, y, yaw)

    def calculate_goals(self):
        center_goal_x = 10.0
        center_goal_y = 10.0
        waiting_radius = 4.0 
        
        active_agents = [i for i in range(self.num_agents) if i in self.initial_poses]
        n = len(active_agents)
        if n == 0: return

        for idx, agent_index in enumerate(active_agents):
            x_start, y_start, yaw_start = self.initial_poses[agent_index]

            if idx == 0:
                my_goal_x = center_goal_x
                my_goal_y = center_goal_y
            else:
                angle = (2.0 * math.pi * idx) / n
                my_goal_x = center_goal_x + (waiting_radius * math.cos(angle))
                my_goal_y = center_goal_y + (waiting_radius * math.sin(angle))

            dx = my_goal_x - x_start
            dy = my_goal_y - y_start
            local_goal_x = (dx * math.cos(yaw_start)) + (dy * math.sin(yaw_start))
            local_goal_y = (-dx * math.sin(yaw_start)) + (dy * math.cos(yaw_start))
            
            self.goals[agent_index] = (local_goal_x, local_goal_y)

    def get_effective_pos(self, pos_x, pos_y, theta):
        """Pushes the tracking coordinates forward away from the turning axle axis."""
        eff_x = pos_x + EFFECTIVE_D * math.cos(theta)
        eff_y = pos_y + EFFECTIVE_D * math.sin(theta)
        return (eff_x, eff_y)

    def get_effective_vel(self, theta, linear_v, angular_w):
        """Converts unicycle speeds to current holonomic point space tracking velocities."""
        vr = linear_v + 0.5 * angular_w * WHEEL_L
        vl = 2 * linear_v - vr
        
        x_vel = (0.5 * math.cos(theta) + EFFECTIVE_D * math.sin(theta) / WHEEL_L) * vl + \
                (0.5 * math.cos(theta) - EFFECTIVE_D * math.sin(theta) / WHEEL_L) * vr
        y_vel = (0.5 * math.sin(theta) - EFFECTIVE_D * math.cos(theta) / WHEEL_L) * vl + \
                (0.5 * math.sin(theta) + EFFECTIVE_D * math.cos(theta) / WHEEL_L) * vr
        return (x_vel, y_vel)

    def calculate_effective_cmd(self, theta, pref_vel):
        """Transforms safe simulated velocities back to Twist linear and angular controls[cite: 3]."""
        A = 0.5 * math.cos(theta) + EFFECTIVE_D * math.sin(theta) / WHEEL_L
        B = 0.5 * math.cos(theta) - EFFECTIVE_D * math.sin(theta) / WHEEL_L
        C = 0.5 * math.sin(theta) - EFFECTIVE_D * math.cos(theta) / WHEEL_L
        D_param = 0.5 * math.sin(theta) + EFFECTIVE_D * math.cos(theta) / WHEEL_L

        vx, vy = pref_vel[0], pref_vel[1]
        
        # Solving the linear matrix decoupling inverse space directly
        vr = (vy - (C / A) * vx) / (D_param - (B * C) / A)
        vl = (vx - B * vr) / A

        vel_msg = Twist()
        # Angular Velocity translation
        raw_angular = (vr - vl) / WHEEL_L
        vel_msg.angular.z = float(np.clip(raw_angular, -MAX_ANGULAR, MAX_ANGULAR))
        
        # Linear Velocity translation
        raw_linear = 0.5 * (vl + vr)
        vel_msg.linear.x = float(np.clip(raw_linear, -MAX_LINEAR, MAX_LINEAR))
        
        return vel_msg
        
    def normalize_angle(self, angle):
        while angle > math.pi: angle -= 2 * math.pi
        while angle < -math.pi: angle += 2 * math.pi
        return angle

    def control_loop(self):
        if len(self.current_poses) < self.num_agents or len(self.initial_poses) < self.num_agents:
            self.get_logger().warning(f'Waiting for all agents to report their positions...'
                                      + f'{len(self.current_poses)}/{self.num_agents} odom,'
                                      + f'{len(self.initial_poses)}/{self.num_agents} TF')
            return
        
        if not self.has_calculated_goals:
            self.calculate_goals()
            self.has_calculated_goals = True

        # 1. Update NH-ORCA state positions and target allocations
        for i in range(self.num_agents):
            pos_x, pos_y, theta = self.current_poses[i]
            linear_v, angular_w = self.current_twists.get(i, (0.0, 0.0))
            
            # Use effective positioning parameters
            eff_pos = self.get_effective_pos(pos_x, pos_y, theta)
            eff_vel = self.get_effective_vel(theta, linear_v, angular_w)

            self.rvo_sim.setAgentPosition(self.agent_rvo_ids[i], eff_pos)
            self.rvo_sim.setAgentVelocity(self.agent_rvo_ids[i], eff_vel)

            # Compute normalized trajectory vector to objective targets
            goal_x, goal_y = self.goals[i]
            dist_to_goal = math.sqrt((goal_x - pos_x)**2 + (goal_y - pos_y)**2)
            
            if dist_to_goal < GOAL_THRESHOLD:
                # Tell RVO2 this agent is parked and will NOT help avoid collisions
                self.rvo_sim.setAgentPrefVelocity(self.agent_rvo_ids[i], (0.0, 0.0))
            else:
                # Car is still driving, calculate trajectory vector
                goal_vector = np.array([goal_x - eff_pos[0], goal_y - eff_pos[1]])
                norm_dist = np.linalg.norm(goal_vector)
                
                if norm_dist > 1.0:
                    goal_vector = goal_vector / norm_dist
                    
                self.rvo_sim.setAgentPrefVelocity(self.agent_rvo_ids[i], tuple(MAX_LINEAR * goal_vector))

        # 2. Complete Simulation Cycle Step
        self.rvo_sim.doStep()

        # 3. Formulate Unicycle Control commands and publish safely
        reach_goal_num = 0
        for i in range(self.num_agents):
            pos_x, pos_y, theta = self.current_poses[i]

            dx = self.goals[i][0] - pos_x
            dy = self.goals[i][1] - pos_y
            dist_to_goal = math.sqrt(dx**2 + dy**2)

            # Stop conditions if within arrival radius configurations
            if dist_to_goal < GOAL_THRESHOLD:
                self.publishers_[i].publish(Twist())  # Empty zeroed vector stop command
                reach_goal_num += 1
                continue

            # Fetch safe simulated velocity assigned from step resolution
            sim_velocity = self.rvo_sim.getAgentVelocity(self.agent_rvo_ids[i])
            
            # Run official matrix decoupling calculation down to the vehicle twist values
            twist_msg = self.calculate_effective_cmd(theta, sim_velocity)
            self.publishers_[i].publish(twist_msg)

        if reach_goal_num > 0:
            self.get_logger().info(f'{reach_goal_num}/{self.num_agents} agents have reached their goals.')