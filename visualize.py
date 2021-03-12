import open3d as o3d

mesh = o3d.io.read_triangle_mesh('./logs/test_result/test.ply')
mesh.compute_vertex_normals()
o3d.visualization.draw_geometries([mesh])