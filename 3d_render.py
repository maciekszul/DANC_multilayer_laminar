import numpy as np
import trimesh
import open3d as o3d


def rotation_matrix(theta1, theta2, theta3):
    c1 = np.cos(theta1 * np.pi / 180)
    s1 = np.sin(theta1 * np.pi / 180)
    c2 = np.cos(theta2 * np.pi / 180)
    s2 = np.sin(theta2 * np.pi / 180)
    c3 = np.cos(theta3 * np.pi / 180)
    s3 = np.sin(theta3 * np.pi / 180)
    matrix=np.array([
        [c2*c3, -c2*s3, s2], 
        [c1*s3+c3*s1*s2, c1*c3-s1*s2*s3, -c2*s1], 
        [s1*s3-c1*c3*s2, c3*s1+c1*s2*s3, c1*c2]
    ])
    return matrix


def custom_draw_geometry(mesh, filename="render.png", visible=True, wh=[960, 960], save=True):
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=wh[0], height=wh[1], visible=visible)
    vis.add_geometry(mesh)
    vis.get_render_option().mesh_show_back_face=True
    vis.run()
    if save:
        vis.capture_screen_image(filename, do_render=True)
    vis.destroy_window()


brain = nb.load("/path/to/3d_brain.gii")
vertices, faces = brain.agg_data()
mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False, validate=False)
mesh = mesh.as_open3d
mesh.compute_vertex_normals(normalized=True)
colour_map = np.repeat(np.array([[1., 1., 1,]]), vertices.shape[0], axis=0)
mesh.vertex_colors = o3d.utility.Vector3dVector(colour_map)
mesh.rotate(rotation_matrix(-130, 60, 0))
filename = "path/to/the/render.png"
custom_draw_geometry(mesh, filename, save=False)