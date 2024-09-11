import carla
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from carla_msgs.msg import CarlaEgoVehicleStatus
from cv_bridge import CvBridge
import cv2
import numpy as np
import time

class CarlaRosBridge(Node):
    def __init__(self):
        super().__init__('carla_ros_bridge')

        # Publishers
        self.image_pub = self.create_publisher(Image, '/carla/ego_vehicle/rgb_front/image', 10)
        self.status_pub = self.create_publisher(CarlaEgoVehicleStatus, '/carla/ego_vehicle/vehicle_status', 10)

        self.bridge = CvBridge()

        # Connect to CARLA server
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(10.0)

        try:
            self.world = self.client.load_world('Town04')
        except RuntimeError as e:
            self.get_logger().error("Unable to connect to CARLA server. Please make sure the simulator is running.")
            raise e

        # Set weather conditions
        self.set_weather_conditions()

        # Remove other vehicles
        self.remove_other_vehicles()

        # Spawn the ego vehicle and camera
        self.ego_vehicle = self.spawn_vehicle('vehicle.tesla.model3', 79)
        self.camera = self.spawn_camera(self.ego_vehicle, 'sensor.camera.rgb', carla.Transform(carla.Location(x=1.5, z=2.4)))

        self.camera.listen(lambda image: self.image_callback(image))

        # Initialize the Traffic Manager
        self.traffic_manager = self.client.get_trafficmanager(8000)  # Use a different port if needed
        self.traffic_manager.set_synchronous_mode(True)
        self.traffic_manager.global_percentage_speed_difference(-50.0)  # Slow down all vehicles by 50%

        self.ego_vehicle.set_autopilot(True, self.traffic_manager.get_port())

        # Define maximum speed in m/s (30 km/h ≈ 8.33 m/s)
        self.MAX_SPEED = 8.33

        # Initialize prev_time
        self.prev_time = time.time()

        self.timer = self.create_timer(0.1, self.publish_vehicle_status)  # Timer to call the function at 10 Hz

    def set_weather_conditions(self):
        weather = carla.WeatherParameters(
            cloudiness=0.0,
            precipitation=0.0,
            sun_altitude_angle=10.0,
            sun_azimuth_angle=70.0,
            precipitation_deposits=0.0,
            wind_intensity=0.0,
            fog_density=0.0,
            wetness=0.0,
        )
        self.world.set_weather(weather)

    def remove_other_vehicles(self):
        for actor in self.world.get_actors():
            if 'vehicle' in actor.type_id:
                actor.destroy()

    def spawn_vehicle(self, model, spawn_point_index):
        bp_lib = self.world.get_blueprint_library()
        vehicle_bp = bp_lib.find(model)
        spawn_points = self.world.get_map().get_spawn_points()
        return self.world.try_spawn_actor(vehicle_bp, spawn_points[spawn_point_index])

    def spawn_camera(self, vehicle, sensor_type, transform):
        bp_lib = self.world.get_blueprint_library()
        camera_bp = bp_lib.find(sensor_type)
        return self.world.spawn_actor(camera_bp, transform, attach_to=vehicle)

    def image_callback(self, image):
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]  # Extract RGB channels
        array = cv2.cvtColor(array, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB

        # Get control values and speed
        control = self.ego_vehicle.get_control()
        velocity = self.ego_vehicle.get_velocity()
        speed = np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2) * 3.6  # Convert to km/h

        # Calculate FPS
        # current_time = time.time()
        # fps = 1.0 / (current_time - self.prev_time)
        # self.prev_time = current_time

        # Draw information on the image
        info_text = f'Throttle: {control.throttle:.2f}, Brake: {control.brake:.2f}, Steer: {control.steer:.2f}, Speed: {speed:.2f} km/h' #, FPS: {fps:.2f}'
        cv2.putText(array, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Publish the image with the overlay
        image_message = self.bridge.cv2_to_imgmsg(array, encoding="rgb8")
        image_message.header.stamp = self.get_clock().now().to_msg()
        self.image_pub.publish(image_message)

    def publish_vehicle_status(self):
        control = self.ego_vehicle.get_control()
        velocity = self.ego_vehicle.get_velocity()
        speed = np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2) * 3.6  # Convert to km/h

        status_msg = CarlaEgoVehicleStatus()
        status_msg.header.stamp = self.get_clock().now().to_msg()
        status_msg.control.steer = control.steer
        status_msg.control.throttle = control.throttle
        status_msg.control.brake = control.brake
        status_msg.velocity = speed

        self.status_pub.publish(status_msg)

        if speed > self.MAX_SPEED * 3.6:  # Convert MAX_SPEED to km/h for comparison
            self.ego_vehicle.disable_constant_velocity()
            control.throttle = 0.0
            control.brake = 1.0
        else:
            self.ego_vehicle.enable_constant_velocity(carla.Vector3D(self.MAX_SPEED, 0, 0))

        self.ego_vehicle.apply_control(control)

    def cleanup(self):
        self.camera.stop()
        self.camera.destroy()
        self.ego_vehicle.destroy()


def main(args=None):
    rclpy.init(args=args)

    bridge = CarlaRosBridge()

    try:
        rclpy.spin(bridge)
    except KeyboardInterrupt:
        pass

    bridge.cleanup()
    bridge.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
