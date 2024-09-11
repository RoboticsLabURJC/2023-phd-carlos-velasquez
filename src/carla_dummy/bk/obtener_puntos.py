# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# Guardar los puntos en listas
x_coords = []
y_coords = []
z_coords = []

for point in spawn_points:
    x_coords.append(point.location.x)
    y_coords.append(point.location.y)
    z_coords.append(point.location.z)

# Opcional: Guardar en un archivo CSV
with open('spawn_points.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['x', 'y', 'z'])
    for x, y, z in zip(x_coords, y_coords, z_coords):
        writer.writerow([x, y, z])

# Plotear los puntos en 2D (x, y)
plt.figure(figsize=(10, 8))
plt.scatter(x_coords, y_coords, c='blue', marker='o')
plt.title('Puntos de inicio en el mapa (2D)')
plt.xlabel('X')
plt.ylabel('Y')
plt.grid(True)
plt.show()

# Plotear los puntos en 3D (x, y, z)
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x_coords, y_coords, z_coords, c='red', marker='o')
ax.set_title('Puntos de inicio en el mapa (3D)')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::