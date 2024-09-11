import carla
import random

def create_route(world, start_location, end_location):
    start_waypoint = world.get_map().get_waypoint(start_location)
    end_waypoint = world.get_map().get_waypoint(end_location)
    route = []
    current_waypoint = start_waypoint

    while current_waypoint.transform.location.distance(end_waypoint.transform.location) > 2.0:
        route.append(current_waypoint.transform.location)
        current_waypoint = random.choice(current_waypoint.next(2.0))

    route.append(end_waypoint.transform.location)
    return route

def draw_route(world, route, is_straight):
    color = carla.Color(r=0, g=255, b=0) if is_straight else carla.Color(r=255, g=0, b=0)
    for location in route:
        world.debug.draw_string(location, 'O', draw_shadow=False, color=color, life_time=120.0, persistent_lines=True)

# Connect to the CARLA server
client = carla.Client('localhost', 2000)
client.set_timeout(10.0)
world = client.get_world()

# Define start and end locations for short and long routes
start_location = carla.Location(x=230, y=195, z=40)
end_location = carla.Location(x=230, y=250, z=40)

# Create routes
route = create_route(world, start_location, end_location)

# Check if the route is straight or curved (example logic)
is_straight = True  # Replace this with actual logic to determine if the route is straight or curved

# Draw the route
draw_route(world, route, is_straight)

# Draw the vehicle position
vehicle = world.get_actors().filter('vehicle.*')[0]
world.debug.draw_string(vehicle.get_location(), 'X', draw_shadow=False, color=carla.Color(r=0, g=0, b=255), life_time=0.1, persistent_lines=True)

print("Route drawn. Check the CARLA window.")
