import open3d as o3d

mesh = o3d.io.read_triangle_mesh('./logs/model/ICL2_reservoir_union_pos/10/test.ply')
mesh.compute_vertex_normals()
o3d.visualization.draw_geometries([mesh])
