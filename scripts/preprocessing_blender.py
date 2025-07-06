import pandas as pd
import bpy
import os
import numpy as np
from bpy_extras.io_utils import axis_conversion

# --- CONFIGURATION ---
BASE_PATH = './data/Truebone_Z-OO/'
CSV_FILENAME = "TrueboneZ-OO.csv"
ARMATURE_NAME = "Armature"
MESH_NAME = "U3DMesh"


def get_bvh_root_name(filepath):
    """Reads a BVH file to find the name of the ROOT joint."""
    try:
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith("ROOT"):
                    return line.split()[1]
    except FileNotFoundError:
        print(f"WARNING: BVH file not found at {filepath}")
    return None

def get_bone_hierarchy(start_bone):
    """Recursively traverses the bone hierarchy to get a flat list of bones."""
    bones_to_export = [start_bone]
    for child in start_bone.children:
        bones_to_export.extend(get_bone_hierarchy(child))
    return bones_to_export

def find_fbx_file(folder_path, animation_name_part):
    """Finds the first FBX file in a folder matching a part of its name."""
    for filename in os.listdir(folder_path):
        if animation_name_part in filename.upper() and ".FBX" in filename.upper():
            return filename
    return None

def clean_scene():
    """Removes all objects from the current scene."""
    if bpy.context.object and bpy.context.object.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()


def export_animation_to_npz(start_frame, end_frame, output_npz_path, root_joint_name):
    """Exports frame-by-frame animation data (joints and vertices) to a .npz file."""
    print("\n--- Starting Animation Export ---")
    print(f"Exporting frames {start_frame}-{end_frame} to: {output_npz_path}")

    try:
        armature = bpy.data.objects[ARMATURE_NAME]
        mesh_obj = bpy.data.objects[MESH_NAME]
    except KeyError as e:
        print(f"ERROR: Cannot find object {e}. Check names in CONFIGURATION.")
        return

    start_bone = armature.pose.bones.get(root_joint_name)
    if not start_bone:
        print(f"ERROR: Root joint '{root_joint_name}' not found. Check BVH file and armature.")
        # Fallback to exporting all bones if root is not found
        bones_to_process = armature.pose.bones
    else:
        bones_to_process = get_bone_hierarchy(start_bone)
    
    print(f"Found {len(bones_to_process)} bones to process in hierarchy.")

    scene = bpy.context.scene
    mat_conversion = axis_conversion(from_forward='Y', from_up='Z', to_forward='-Z', to_up='Y').to_4x4()
    data_for_npz = {}
    joint_names_list = [bone.name for bone in bones_to_process]

    for frame in range(start_frame, end_frame + 1):
        scene.frame_set(frame)

        # Export joint positions
        joints_pos_list = [(mat_conversion @ (armature.matrix_world @ bone.head)) for bone in bones_to_process]
        data_for_npz[f'frame_{frame}_joints'] = np.array(joints_pos_list, dtype=np.float32)

        # Export vertex positions
        # Create a temporary copy of the mesh to apply the modifier
        bpy.ops.object.select_all(action='DESELECT')
        mesh_obj.select_set(True)
        bpy.context.view_layer.objects.active = mesh_obj
        bpy.ops.object.duplicate()
        mesh_copy = bpy.context.active_object

        try:
            bpy.ops.object.modifier_apply(modifier="Armature")
            
            vertex_count = len(mesh_copy.data.vertices)
            vertices_local_co = np.empty(vertex_count * 3, dtype=np.float32)
            mesh_copy.data.vertices.foreach_get('co', vertices_local_co)
            vertices_local_co.shape = (vertex_count, 3)

            vertices_4d = np.ones((vertex_count, 4), dtype=np.float32)
            vertices_4d[:, :3] = vertices_local_co

            # Efficient matrix multiplication with numpy
            final_transform = np.array(mat_conversion @ mesh_copy.matrix_world)
            transformed_vertices = (final_transform @ vertices_4d.T).T
            data_for_npz[f'frame_{frame}_vertices'] = transformed_vertices[:, :3].astype(np.float32)

        finally:
            bpy.data.objects.remove(mesh_copy, do_unlink=True)

    data_for_npz['joint_names'] = np.array(joint_names_list)
    np.savez_compressed(output_npz_path, **data_for_npz)
    print(f"SUCCESS: Animation data saved to {output_npz_path}")

def export_skinning_data_to_npz(output_npz_path, root_joint_name):
    """Exports static skinning data (weights, rest vertices, faces) to a .npz file."""
    print("\n--- Starting Skinning Data Export ---")
    print(f"Exporting skinning data for root '{root_joint_name}' to: {output_npz_path}")

    try:
        mesh_obj = bpy.data.objects[MESH_NAME]
        armature = bpy.data.objects[ARMATURE_NAME]
    except KeyError as e:
        print(f"ERROR: Cannot find object {e}. Check names in CONFIGURATION.")
        return

    start_bone = armature.pose.bones.get(root_joint_name)
    if not start_bone:
        print(f"ERROR: Root joint '{root_joint_name}' not found.")
        return

    bones_in_hierarchy = get_bone_hierarchy(start_bone)
    filtered_joint_names = [bone.name for bone in bones_in_hierarchy]
    name_to_filtered_index = {name: i for i, name in enumerate(filtered_joint_names)}
    print(f"Found {len(bones_in_hierarchy)} joints in hierarchy for skinning.")

    # Get vertices in rest pose
    num_vertices = len(mesh_obj.data.vertices)
    rest_pose_vertices = np.empty(num_vertices * 3, dtype=np.float32)
    mesh_obj.data.vertices.foreach_get('co', rest_pose_vertices)
    rest_pose_vertices = rest_pose_vertices.reshape(num_vertices, 3)

    # Get mesh topology (faces)
    faces = [p.vertices[:] for p in mesh_obj.data.polygons]

    # Get skinning weights for the filtered joints
    skin_weights = np.zeros((num_vertices, len(filtered_joint_names)), dtype=np.float32)
    for vertex in mesh_obj.data.vertices:
        for group in vertex.groups:
            vg_name = mesh_obj.vertex_groups[group.group].name
            if vg_name in name_to_filtered_index:
                joint_idx = name_to_filtered_index[vg_name]
                skin_weights[vertex.index, joint_idx] = group.weight
    
    # Save all data
    np.savez_compressed(
        output_npz_path,
        weights=skin_weights,
        joint_names=np.array(filtered_joint_names),
        rest_pose_vertices=rest_pose_vertices,
        faces=np.array(faces, dtype=object) # Use dtype=object for variable-length lists
    )
    print(f"SUCCESS: Skinning data saved to {output_npz_path}")


def main():
    """Main function to drive the preprocessing workflow."""
    clean_scene()

    # Load and prepare the CSV file
    csv_path = os.path.join(BASE_PATH, CSV_FILENAME)
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"CRITICAL ERROR: CSV file not found at {csv_path}")
        bpy.ops.wm.quit_blender()
        return

    df = df.dropna()
    df = df[df['Group'] != 'Group']
    df.columns = df.columns.str.strip()

    processed_skin_models = set()

    # Loop through each animation entry in the CSV
    for index, row in df.iterrows():
        print("\n" + "="*50)
        print(f"Processing row {index}: Group '{row['Group']}', File '{row['File']}'")

        clean_scene()

        folder = row["Group"].strip()
        animation_name_part = row["File"].strip().replace("__", "").replace(".BVH", "")
        folder_path = os.path.join(BASE_PATH, folder)
        
        # Find the corresponding FBX file
        fbx_filename = find_fbx_file(folder_path, animation_name_part)
        if not fbx_filename:
            print(f"WARNING: No matching FBX found for '{animation_name_part}' in folder '{folder}'. Skipping.")
            continue
            
        fbx_path = os.path.join(folder_path, fbx_filename)
        bpy.ops.import_scene.fbx(filepath=fbx_path)

        # Get root joint from the corresponding BVH file
        bvh_file_path = os.path.join(folder_path, row["File"].strip())
        root_joint_name = get_bvh_root_name(bvh_file_path)
        if not root_joint_name:
            print(f"ERROR: Could not find root joint for '{bvh_file_path}'. Skipping this entry.")
            continue

        # --- 1. Export Skinning Data ---
        model_id = folder
        if model_id not in processed_skin_models:
            skin_output_path = os.path.join(folder_path, "skin_data.npz")
            export_skinning_data_to_npz(skin_output_path, root_joint_name)
            processed_skin_models.add(model_id)
        else:
            print(f"\nSkinning data for model '{model_id}' already exported. Skipping.")

        # --- 2. Export Animation Data ---
        start_frame = 0
        end_frame = int(row["Frames"])
        anim_output_path = fbx_path.replace(".FBX", ".npz").replace(".fbx", ".npz")
        export_animation_to_npz(start_frame, end_frame, anim_output_path, root_joint_name)

    print("\n" + "="*50)
    print("ALL PROCESSING COMPLETED SUCCESSFULLY!")


if __name__ == "__main__":
    main()