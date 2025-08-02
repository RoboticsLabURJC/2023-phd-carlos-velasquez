import carla
import random
import sys
import glob


client = carla.Client('localhost', 2000)
client.set_timeout(10.0)
world = client.load_world('Town04')
map = world.get_map()

# Selecciona un punto de inicio y un punto final para los waypoints
start_location = carla.Location(x=402.84, y=-39.13, z=0.0)
end_location = carla.Location(x=11.17, y=-312.36, z=0.0)

# Obtén los waypoints entre los dos puntos
waypoints = map.generate_waypoints(2.0)

# Filtra los waypoints cercanos a los puntos de inicio y fin
start_waypoint = map.get_waypoint(start_location, project_to_road=True)
end_waypoint = map.get_waypoint(end_location, project_to_road=True)

# Genera la ruta
route = []
waypoint = start_waypoint
while waypoint.transform.location.distance(end_waypoint.transform.location) > 2.0:
    route.append(waypoint)
    waypoint = random.choice(waypoint.next(2.0))

# Dibuja los waypoints en el mundo de CARLA
for i, waypoint in enumerate(route):
    world.debug.draw_string(waypoint.transform.location, 'O', draw_shadow=False,
                            color=carla.Color(r=255, g=0, b=0), life_time=120.0, persistent_lines=True)
    if i % 10 == 0:
        world.debug.draw_string(waypoint.transform.location, 'O', draw_shadow=False,
                                color=carla.Color(r=0, g=0, b=255), life_time=120.0, persistent_lines=True)
