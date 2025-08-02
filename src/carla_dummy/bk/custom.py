import carla
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Lambda
from numba import cuda

# Limpiar la memoria de la GPU
tf.keras.backend.clear_session()
cuda.select_device(0)
cuda.close()

# Limitar el uso de memoria de la GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# Register the custom loss function
@tf.keras.utils.register_keras_serializable()
def custom_loss(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_pred - y_true) * (tf.abs(y_true) + 0.1))

# Load the trained model with safe_mode=False
model_path = "/home/canveo/Projects/notebook/model/jul27_3gof_epoch_120.keras"
model = tf.keras.models.load_model(model_path, custom_objects={'custom_loss': custom_loss}, safe_mode=False)

def process_image(image):
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]  # Extract RGB channels
    return array

def preprocess_for_model(image):
    # Recortar y redimensionar la imagen
    img_cropped_resized = cv2.resize(image[200:-1, :], (200, 66))
    return img_cropped_resized / 127.5 - 1.0

# Conectar al servidor CARLA y cargar el mundo
client = carla.Client('localhost', 2000)
client.set_timeout(10.0)
try:
    world = client.load_world('Town04')
except RuntimeError as e:
    print("Unable to connect to CARLA server. Please make sure the simulator is running.")
    raise e

# Set weather conditions
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
ego_vehicle = world.try_spawn_actor(vehicle_bp, spawn_points[100])

# Set up the RGB camera
camera_bp = bp_lib.find('sensor.camera.rgb')
camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))  # Adjust the position to be front-facing
camera = world.spawn_actor(camera_bp, camera_transform, attach_to=ego_vehicle)

# Initialize OpenCV window
cv2.namedWindow("CARLA Manual Control", cv2.WINDOW_AUTOSIZE)

# Attach the callback to the camera
frame = None
frame_sequence = []

def image_callback(image):
    global frame, frame_sequence
    frame = process_image(image)
    processed_frame = preprocess_for_model(frame)
    
    if len(frame_sequence) < 3:
        frame_sequence.append(processed_frame)
    else:
        frame_sequence.pop(0)
        frame_sequence.append(processed_frame)

camera.listen(lambda image: image_callback(image))

control = carla.VehicleControl()

done = False
while not done:
    if frame is not None and len(frame_sequence) == 3:
        # Preprocess the frame for the model
        model_input = np.expand_dims(frame_sequence, axis=0)  # Add batch dimension
        
        # Predict the steering angle, throttle, and brake
        predictions = model.predict(model_input)
        steer_prediction = predictions[0][0]
        throttle_prediction = 0.16  # predictions[0][1]
        brake_prediction = 0.0  # predictions[0][2]

        # Ajustar el valor del throttle si es necesario
        if throttle_prediction < 0.1:
            throttle_prediction = 0.4  # Valor mínimo para asegurar movimiento

        print(f"Steer: {steer_prediction}, Throttle: {throttle_prediction}, Brake: {brake_prediction}")
        
        # Set the control constants and predicted values
        control.steer = float(steer_prediction)
        control.throttle = float(throttle_prediction)
        control.brake = float(brake_prediction)

        # Display the frame
        cv2.imshow("CARLA Manual Control", frame)

    # Apply the control to the ego vehicle and tick the simulation
    ego_vehicle.apply_control(control)
    world.tick()

    # Exit condition
    if cv2.waitKey(1) & 0xFF == ord('q'):
        done = True

# Clean up
camera.stop()
camera.destroy()
ego_vehicle.destroy()
cv2.destroyAllWindows()
