import pybullet as p

urdf_file = "fsae/resources/models/cone_blue/cone_blue.urdf"
# Connect to the physics server
physics_client = p.connect(p.GUI)  # Use p.GUI for graphical visualization or p.DIRECT for non-graphical

# Load URDF file and perform operations as needed
obj_id = p.loadURDF(urdf_file)

# Perform operations such as getting bounding box dimensions
# For example:
aabb_min, aabb_max = p.getAABB(obj_id)
bounding_box_dimensions = [aabb_max[i] - aabb_min[i] for i in range(3)]

print("Bounding Box Dimensions:", bounding_box_dimensions)

# Disconnect from the physics server when done
p.disconnect()