import carla
import random

def create_waypoints(town_name, distance=2.0):
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.load_world(town_name)
    map = world.get_map()

    waypoints = map.generate_waypoints(distance)
    for w in waypoints:
        world.debug.draw_string(w.transform.location, 'O', draw_shadow=False,
                                color=carla.Color(r=255, g=0, b=0), life_time=120.0,
                                persistent_lines=True)
    return waypoints

if __name__ == "__main__":
    town_name = 'Town05'
    create_waypoints(town_name)