import carla
import cv2
import numpy as np
import time
import os

# import matplotlib.pyplot as plt

# Connect to the CARLA server and load the world
client = carla.Client('localhost', 2000)
client.set_timeout(10.0)  # Set a timeout for the client
try:
    world = client.load_world('Town02')
except RuntimeError as e:
    print("Unable to connect to CARLA server. Please make sure the simulator is running.")
    raise e

# Set weather conditions
weather = carla.WeatherParameters(
    cloudiness=0.0, # 0.0
    precipitation=0.0,
    sun_altitude_angle=10.0, #10.0
    sun_azimuth_angle=70.0, # 70.0
    precipitation_deposits=0.0, # 0.0
    wind_intensity=0.0,
    fog_density=0.0,
    wetness=0.0,
)
world.set_weather(weather)

# Remove other vehicles
for actor in world.get_actors():
    if 'vehicle' in actor.type_id:
        actor.destroy()

# Get the blueprint library and spawn points
bp_lib = world.get_blueprint_library()
spawn_points = world.get_map().get_spawn_points()

# Spawn the ego vehicle
vehicle_bp = bp_lib.find('vehicle.tesla.model3')
ego_vehicle = world.try_spawn_actor(vehicle_bp, spawn_points[12])

# Set up the RGB camera
camera_bp = bp_lib.find('sensor.camera.rgb')
camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))  # Adjust the position to be front-facing
camera = world.spawn_actor(camera_bp, camera_transform, attach_to=ego_vehicle)

# Initialize OpenCV window
cv2.namedWindow("CARLA Manual Control", cv2.WINDOW_AUTOSIZE)

# Define maximum speed in m/s (30 km/h ≈ 8.33 m/s)
MAX_SPEED = 8.33

# Variables to calculate FPS
fps = 0
prev_time = time.time()

def process_image(image):
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = np.reshape(array, (image.height, image.width, 4))
    return array

def image_callback(image):
    global prev_time, fps

    # Process the image
    frame = process_image(image)

    # Access current control state
    control = ego_vehicle.get_control()

    # Check and limit the speed
    velocity = ego_vehicle.get_velocity()
    speed = np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2) * 3.6  # Convert to km/h
    if speed > MAX_SPEED * 3.6:  # Convert MAX_SPEED to km/h for comparison
        ego_vehicle.disable_constant_velocity()
        control.throttle = 0.0
        control.brake = 1.0
    else:
        ego_vehicle.enable_constant_velocity(carla.Vector3D(MAX_SPEED, 0, 0))

    # Calculate FPS
    current_time = time.time()
    fps = 1.0 / (current_time - prev_time)
    prev_time = current_time

    # Display information on the frame
    info_text = [
        f"Throttle: {control.throttle:.2f}",
        f"Brake: {control.brake:.2f}",
        f"Steer: {control.steer:.2f}",
        f"Speed: {speed:.2f} km/h",
        f"FPS: {fps:.2f}"
    ]

    # Ensure frame is in the correct format for display
    display_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    
    # Position for each line of text
    y0, dy = 30, 30
    for i, line in enumerate(info_text):
        y = y0 + i*dy
        cv2.putText(display_frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow("CARLA Manual Control", display_frame)

# Attach the callback to the camera
camera.listen(image_callback)

# Enable autopilot
ego_vehicle.set_autopilot(True)

done = False
while not done:
    # Exit condition
    if cv2.waitKey(1) & 0xFF == ord('q'):
        done = True

    world.tick()

# Clean up
camera.stop()
camera.destroy()
ego_vehicle.destroy()
cv2.destroyAllWindows()
