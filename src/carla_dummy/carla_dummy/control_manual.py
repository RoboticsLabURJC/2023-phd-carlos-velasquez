import carla
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.losses import MeanSquaredError

from numba import cuda

# Limpiar la memoria de la GPU
tf.keras.backend.clear_session()
cuda.select_device(0)
cuda.close()

@tf.keras.utils.register_keras_serializable()
def mse(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))


model_path = "/home/canveo/Projects/notebook/model/jul27_3gof_epoch_120.keras"
model = tf.keras.models.load_model(model_path, custom_objects={'mse': mse}, safe_mode=False)

def process_image(image):
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]  # Extract RGB channels
    return array

def preprocess_for_model(image):
    img_cropped_resized = cv2.resize(image[200:-1, :], (200, 66))
    return img_cropped_resized

client = carla.Client('localhost', 2000)
client.set_timeout(10.0)
try:
    world = client.load_world('Town03')
except RuntimeError as e:
    print("Unable to connect to CARLA server. Please make sure the simulator is running.")
    raise e


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


for actor in world.get_actors():
    if 'vehicle' in actor.type_id:
        actor.destroy()


bp_lib = world.get_blueprint_library()
spawn_points = world.get_map().get_spawn_points()


vehicle_bp = bp_lib.find('vehicle.tesla.model3')
ego_vehicle = world.try_spawn_actor(vehicle_bp, spawn_points[1])


camera_bp = bp_lib.find('sensor.camera.rgb')
camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4)) 
camera = world.spawn_actor(camera_bp, camera_transform, attach_to=ego_vehicle)


cv2.namedWindow("CARLA Manual Control", cv2.WINDOW_AUTOSIZE)


frame = None
def image_callback(image):
    global frame
    frame = process_image(image)

camera.listen(lambda image: image_callback(image))

control = carla.VehicleControl()

done = False
while not done:
    if frame is not None:
     
        model_input = preprocess_for_model(frame)
        model_input = np.expand_dims(model_input, axis=0)  
        
      
        predictions = model.predict(model_input)
        steer_prediction = predictions[0][0]
        throttle_prediction = 0.2 #predictions[0][1]
        brake_prediction = 0.0 #predictions[0][2]

        # Ajustar el valor del throttle si es necesario
        if throttle_prediction < 0.1:
            throttle_prediction = 0.4  # Valor mínimo para asegurar movimiento

        print(f"Steer: {steer_prediction}, Throttle: {throttle_prediction}, Brake: {brake_prediction}")
        

        control.steer = float(steer_prediction)
        control.throttle = float(throttle_prediction)
        control.brake = float(brake_prediction)


        cv2.imshow("CARLA Manual Control", frame)


    ego_vehicle.apply_control(control)
    world.tick()


    if cv2.waitKey(1) & 0xFF == ord('q'):
        done = True


camera.stop()
camera.destroy()
ego_vehicle.destroy()
cv2.destroyAllWindows()
