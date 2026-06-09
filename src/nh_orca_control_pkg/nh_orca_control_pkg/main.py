import rclpy
from nh_orca_control_pkg.swarm_control_node import SwarmControlNode

def main():
    rclpy.init()
    node = SwarmControlNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
