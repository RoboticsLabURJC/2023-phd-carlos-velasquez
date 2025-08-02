import carla

def main():
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()

    tm = client.get_trafficmanager(8010)

    ego_vehicle = None
    for actor in world.get_actors():
        if actor.type_id.startswith('vehicle.') and actor.attributes.get('role_name') == 'ego_vehicle':
            ego_vehicle = actor
            break

    if ego_vehicle:
        print("🚗 Ego vehicle encontrado. Aplicando configuración TM...")
        tm.ignore_lights_percentage(ego_vehicle, 100.0)
        tm.ignore_signs_percentage(ego_vehicle, 100.0)
        tm.auto_lane_change(ego_vehicle, False)
        tm.distance_to_leading_vehicle(ego_vehicle, 0.0)
        print("✅ Listo.")
    else:
        print("⚠️ No se encontró un vehículo con role_name='ego_vehicle'.")

if __name__ == '__main__':
    main()
