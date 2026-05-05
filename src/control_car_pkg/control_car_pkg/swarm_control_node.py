import math
import rvo2
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist

GOAL_THRESHOLD   = 0.3
ANGULAR_GAIN     = 1.2        # softer gain to avoid saturation
MAX_ANGULAR      = 1.5        # Isaac maxAngularSpeed
MAX_LINEAR       = 1.2        # Isaac maxLinearSpeed
# Blend linear speed down smoothly as angle error grows.
# At >= this angle error, linear = 0. At 0 error, linear = full speed.
ANGLE_SCALE      = math.pi / 3   # 60 degrees -> full stop linearly


class SwarmControlNode(Node):
    def __init__(self, num_agents=11):
        super().__init__('swarm_control_node')
        self.num_agents = num_agents
        
        # 1. Initialize RVO2 Simulator
        # Params: timeStep, neighborDist, maxNeighbors, timeHorizon, timeHorizonObst, radius, maxSpeed
        self.rvo_sim = rvo2.PyRVOSimulator(0.05, 5.0, 5, 2.0, 2.0, 0.5, 1.2)
        
        self.agent_rvo_ids = []
        self.goals = [(10.0, 10.0)] * num_agents 
        
        # 2. Setup ROS 2 Publishers and Subscribers for each agent
        self.publishers_ = []
        self.subscribers_ = []
        self.current_poses = {} # Store latest (x,y) from odometry
        
        for i in range(self.num_agents):
            # Add agent to ORCA sim (initially at 0,0)
            agent_id = self.rvo_sim.addAgent((0.0, 0.0))
            self.agent_rvo_ids.append(agent_id)
            
            # Create Cmd_vel Publisher: e.g., /car_0/cmd_vel
            pub = self.create_publisher(Twist, f'/car_{i}/cmd_vel', 10)
            self.publishers_.append(pub)
            
            # Create Odom Subscriber: e.g., /car_0/odom
            sub = self.create_subscription(Odometry, f'/car_{i}/odom', 
                                           lambda msg, idx=i: self.odom_callback(msg, idx), 10)
            self.subscribers_.append(sub)

        # 3. Control Loop (runs at 20Hz)
        self.timer = self.create_timer(0.05, self.control_loop)

    def odom_callback(self, msg, agent_index):
        # Extract position
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        
        # Extract orientation quaternion
        q = msg.pose.pose.orientation
        
        # Convert quaternion to Euler angle (Yaw)
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        yaw = yaw + math.pi  # FIX: Rotate 180 degrees to match Isaac's forward direction
        yaw = self.normalize_angle(yaw)  # Ensure yaw is within [-pi, pi]
        
        # Store x, y, and yaw
        self.current_poses[agent_index] = (x, y, yaw)

    def calculate_vector_to_goal(self, agent_rvo_id, current_pos, goal_pos):
        dx = goal_pos[0] - current_pos[0]
        dy = goal_pos[1] - current_pos[1]
        distance = math.sqrt(dx**2 + dy**2)

        if distance > GOAL_THRESHOLD:
            max_speed = self.rvo_sim.getAgentMaxSpeed(agent_rvo_id)  # FIX: use correct agent
            return (dx / distance * max_speed, dy / distance * max_speed)
        else:
            return (0.0, 0.0)
        
    def normalize_angle(self, angle):
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle

    def control_loop(self):
        if len(self.current_poses) < self.num_agents:
            self.get_logger().warn(
                f'Waiting for agents... {len(self.current_poses)}/{self.num_agents}')
            return

        # 1. Update RVO2 with current positions and preferred velocities
        for i in range(self.num_agents):
            pos_2d = self.current_poses[i][:2]  # FIX: strip yaw before passing to RVO2

            self.rvo_sim.setAgentPosition(self.agent_rvo_ids[i], pos_2d)

            goal_vector = self.calculate_vector_to_goal(
                self.agent_rvo_ids[i], pos_2d, self.goals[i])  # FIX: pass correct agent id
            self.rvo_sim.setAgentPrefVelocity(self.agent_rvo_ids[i], goal_vector)

        # 2. Step ORCA
        self.rvo_sim.doStep()

        # 3. Publish safe velocities
        for i in range(self.num_agents):
            twist_msg = Twist()
            pos_2d = self.current_poses[i][:2]

            dx = self.goals[i][0] - pos_2d[0]
            dy = self.goals[i][1] - pos_2d[1]
            dist_to_goal = math.sqrt(dx**2 + dy**2)

            if dist_to_goal < GOAL_THRESHOLD:
                self.publishers_[i].publish(twist_msg)  # zero twist = stop
                self.get_logger().info(f'Agent {i} reached goal!')
                continue

            safe_velocity = self.rvo_sim.getAgentVelocity(self.agent_rvo_ids[i])
            linear_speed  = math.sqrt(safe_velocity[0]**2 + safe_velocity[1]**2)
            desired_angle = math.atan2(safe_velocity[1], safe_velocity[0])

            current_angle = self.current_poses[i][2]
            angle_diff    = self.normalize_angle(desired_angle - current_angle)

            # Smoothly scale linear speed: full speed when aligned, zero when 60°+ off
            angle_ratio        = max(0.0, 1.0 - abs(angle_diff) / ANGLE_SCALE)
            blended_linear     = linear_speed * angle_ratio

            # FIX 1: Negate linear.x because Isaac's vehicle forward axis is flipped
            twist_msg.linear.x  = blended_linear

            # FIX 2: Clamp angular within Isaac's limit
            twist_msg.angular.z = max(-MAX_ANGULAR, min(MAX_ANGULAR, ANGULAR_GAIN * angle_diff))

            self.get_logger().debug(
                f'Agent {i}: dist={dist_to_goal:.2f}, angle_diff={math.degrees(angle_diff):.1f}°, '
                f'lin={twist_msg.linear.x:.3f}, ang={twist_msg.angular.z:.3f}')

            self.publishers_[i].publish(twist_msg)
