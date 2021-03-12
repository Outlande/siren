import open3d as o3d

mesh = o3d.io.read_triangle_mesh('/home/yan/Desktop/codes/siren/logs/test_1/test.ply')
mesh.compute_vertex_normals()
o3d.visualization.draw_geometries([mesh])