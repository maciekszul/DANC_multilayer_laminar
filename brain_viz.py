import nibabel as nb
import numpy as np
import trimesh
import open3d as o3d
import open3d.visualization.rendering as rendering
import open3d.visualization.gui as gui
import pickle




def custom_draw_geometry(pcd):
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.get_render_option().mesh_show_back_face=True
    vis.run()
    vis.capture_screen_image("test.png", do_render=True)
    vis.destroy_window()
    


if __name__ == "__main__":
    brain = nb.load("/home/common/bonaiuto/multiburst/derivatives/processed/sub-001/multilayer_11/pial.ds.inflated.nodeep.gii")
    vertices, faces = brain.agg_data()
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False, validate=False)
    mesh = mesh.as_open3d
    mesh.compute_vertex_normals(normalized=True)

    custom_draw_geometry(mesh)


