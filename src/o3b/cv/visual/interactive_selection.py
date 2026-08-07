import logging
import json
import numpy as np
import torch
from pathlib import Path
from typing import List, Union

logger = logging.getLogger(__name__)


def build_geometries_from_pts3d_and_meshes(
    pts3d: List[torch.Tensor] = None,
    pts3d_names: List[str] = None,
    pts3d_colors: List = None,
    meshes_list: List = None,
    meshes_names: List[str] = None,
):
    """
    Helper function to build geometries list from point clouds and meshes.

    This makes it easy to prepare data for interactive selection.

    Args:
        pts3d: List of Nx3 torch tensors (point clouds)
        pts3d_names: Names for each point cloud
        pts3d_colors: Colors for each point cloud (uniform or per-point)
        meshes_list: List of Open3D TriangleMesh objects or od3d Meshes
        meshes_names: Names for each mesh

    Returns:
        List of geometry dicts with 'name' and 'geometry' keys
    """
    import open3d as o3d
    geometries = []

    # Add point clouds
    if pts3d is not None:
        for i, pts3d_i in enumerate(pts3d):
            pts3d_engine = o3d.geometry.PointCloud()
            pts3d_engine.points = o3d.utility.Vector3dVector(
                pts3d_i.detach().cpu().numpy()
            )

            # Get color
            if pts3d_colors is not None and i < len(pts3d_colors):
                color = pts3d_colors[i]
                if isinstance(color, torch.Tensor):
                    if color.dim() == 1:
                        # Uniform color
                        pts3d_engine.paint_uniform_color(color.detach().cpu().numpy()[:3])
                    else:
                        # Per-point colors
                        pts3d_engine.colors = o3d.utility.Vector3dVector(
                            color.detach().cpu().numpy()
                        )
                elif isinstance(color, (list, tuple)):
                    pts3d_engine.paint_uniform_color(color[:3])

            # Get name
            name = pts3d_names[i] if pts3d_names and i < len(pts3d_names) else f"pts3d_{i}"

            geometries.append({"name": name, "geometry": pts3d_engine})

    # Add meshes
    if meshes_list is not None:
        for i, mesh in enumerate(meshes_list):
            # Handle od3d Meshes class
            if hasattr(mesh, 'to_o3d'):
                mesh = mesh.to_o3d()

            name = meshes_names[i] if meshes_names and i < len(meshes_names) else f"mesh_{i}"
            geometries.append({"name": name, "geometry": mesh})

    return geometries


def show_scene_with_selection(
    geometries: List[dict],
    selection_save_path: Path = None,
    pts3d_size: float = 3.0,
):
    """
    Interactive Open3D viewer for object selection.

    Click on objects to select them (they will turn green when selected).
    After closing the window, you'll be prompted to enter the numbers of objects you want to keep.

    Args:
        geometries: List of dicts with 'name' and 'geometry' keys from show_scene
        selection_save_path: Path to save selected object names (default: selected_objects.json)
        pts3d_size: Point size for visualization

    Returns:
        List of selected object names
    """
    import open3d as o3d
    if selection_save_path is None:
        selection_save_path = Path("selected_objects.json")

    # Show instructions
    logger.info("\n" + "="*60)
    logger.info("=== Interactive Selection Mode ===")
    logger.info("="*60)
    logger.info("\nAvailable objects:")
    for i, geom_dict in enumerate(geometries):
        logger.info(f"  {i}: {geom_dict['name']}")
    logger.info("\nInstructions:")
    logger.info("  1. Rotate and explore the scene")
    logger.info("  2. Close the window when you're done viewing")
    logger.info("  3. You'll then enter the numbers of objects you WANT TO KEEP")
    logger.info("="*60 + "\n")

    # Create visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Interactive Selection - Explore scene, then close to select objects")

    # Set render options
    opt = vis.get_render_option()
    opt.point_size = pts3d_size
    opt.mesh_show_back_face = True

    # Add all geometries
    for geom_dict in geometries:
        if geom_dict["geometry"] is not None:
            vis.add_geometry(geom_dict["geometry"])

    # Run visualizer
    vis.run()
    vis.destroy_window()

    # After window closes, ask for selection
    logger.info("\n" + "="*60)
    logger.info("=== Selection Input ===")
    logger.info("="*60)
    logger.info("\nAvailable objects:")
    for i, geom_dict in enumerate(geometries):
        logger.info(f"  {i}: {geom_dict['name']}")
    logger.info("\nEnter the numbers of objects you WANT TO KEEP (comma-separated):")
    logger.info("Example: 0,2,5  or  1,3,4,7")

    try:
        user_input = input("Object numbers to KEEP: ")

        if not user_input.strip():
            logger.warning("No selection made. Saving empty selection.")
            selected_indices = []
        else:
            selected_indices = [int(x.strip()) for x in user_input.split(",") if x.strip()]

        # Validate indices
        valid_indices = [i for i in selected_indices if 0 <= i < len(geometries)]
        if len(valid_indices) != len(selected_indices):
            logger.warning(f"Some indices were out of range. Using valid indices: {valid_indices}")
            selected_indices = valid_indices

        selected_names = [geometries[i]["name"] for i in selected_indices]

        # Save to JSON
        selection_data = {
            "selected_objects": selected_names,
            "selected_indices": selected_indices,
            "total_objects": len(geometries),
            "all_objects": [g["name"] for g in geometries]
        }

        with open(selection_save_path, 'w') as f:
            json.dump(selection_data, f, indent=2)

        logger.info("\n" + "="*60)
        logger.info(f"Selection saved to: {selection_save_path}")
        logger.info(f"Selected {len(selected_names)} out of {len(geometries)} objects:")
        for name in selected_names:
            logger.info(f"  ✓ {name}")
        logger.info("="*60 + "\n")

        return selected_names

    except Exception as e:
        logger.error(f"Error processing selection: {e}")
        return []


def show_scene_with_click_selection(
    geometries: List[dict],
    selection_save_path: Path = None,
    pts3d_size: float = 3.0,
):
    """
    Interactive Open3D viewer with visual object selection.

    Press number keys (0-9) to toggle selection of objects.
    Selected objects turn GREEN. Press 'S' to save.

    Args:
        geometries: List of dicts with 'name' and 'geometry' keys from show_scene
        selection_save_path: Path to save selected object names (default: selected_objects.json)
        pts3d_size: Point size for visualization

    Returns:
        List of selected object names
    """
    import open3d as o3d
    if selection_save_path is None:
        selection_save_path = Path("selected_objects.json")

    # Track selection state and original colors
    selected_objects = set()
    original_colors = {}

    # Store original colors
    for i, geom_dict in enumerate(geometries):
        geom = geom_dict["geometry"]
        if geom is None:
            continue

        if isinstance(geom, o3d.geometry.PointCloud):
            if geom.has_colors():
                original_colors[i] = np.asarray(geom.colors).copy()
            else:
                original_colors[i] = None
        elif isinstance(geom, o3d.geometry.TriangleMesh):
            if geom.has_vertex_colors():
                original_colors[i] = np.asarray(geom.vertex_colors).copy()
            else:
                original_colors[i] = None
        elif isinstance(geom, o3d.geometry.LineSet):
            if geom.has_colors():
                original_colors[i] = np.asarray(geom.colors).copy()
            else:
                original_colors[i] = None

    # Create visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Interactive Selection - Press 0-9 to select, S to save, Q to quit")

    # Set render options
    opt = vis.get_render_option()
    opt.point_size = pts3d_size
    opt.mesh_show_back_face = True

    # Add all geometries
    for geom_dict in geometries:
        if geom_dict["geometry"] is not None:
            vis.add_geometry(geom_dict["geometry"])

    def toggle_selection(vis, i):
        """Toggle selection state of a geometry"""
        import open3d as o3d
        if i >= len(geometries):
            logger.warning(f"Index {i} out of range (max: {len(geometries)-1})")
            return False

        geom_dict = geometries[i]
        geom = geom_dict["geometry"]
        name = geom_dict["name"]

        if geom is None:
            return False

        if i in selected_objects:
            # Deselect: restore original color
            selected_objects.remove(i)
            if original_colors.get(i) is not None:
                if isinstance(geom, o3d.geometry.PointCloud):
                    geom.colors = o3d.utility.Vector3dVector(original_colors[i])
                elif isinstance(geom, o3d.geometry.TriangleMesh):
                    geom.vertex_colors = o3d.utility.Vector3dVector(original_colors[i])
                elif isinstance(geom, o3d.geometry.LineSet):
                    geom.colors = o3d.utility.Vector3dVector(original_colors[i])
            logger.info(f"❌ Deselected [{i}]: {name}")
        else:
            # Select: highlight in green
            selected_objects.add(i)
            if isinstance(geom, o3d.geometry.PointCloud):
                num_points = len(geom.points)
                geom.colors = o3d.utility.Vector3dVector(np.tile([0, 1, 0], (num_points, 1)))
            elif isinstance(geom, o3d.geometry.TriangleMesh):
                num_verts = len(geom.vertices)
                geom.vertex_colors = o3d.utility.Vector3dVector(np.tile([0, 1, 0], (num_verts, 1)))
            elif isinstance(geom, o3d.geometry.LineSet):
                num_lines = len(geom.lines)
                geom.colors = o3d.utility.Vector3dVector(np.tile([0, 1, 0], (num_lines, 1)))
            logger.info(f"✅ Selected [{i}]: {name}")

        vis.update_geometry(geom)
        return False

    def save_selection(vis):
        """Save current selection to JSON"""
        selected_indices = sorted(list(selected_objects))
        selected_names = [geometries[i]["name"] for i in selected_indices]

        selection_data = {
            "selected_objects": selected_names,
            "selected_indices": selected_indices,
            "total_objects": len(geometries),
            "all_objects": [g["name"] for g in geometries]
        }

        with open(selection_save_path, 'w') as f:
            json.dump(selection_data, f, indent=2)

        logger.info("\n" + "="*60)
        logger.info(f"💾 Selection saved to: {selection_save_path}")
        logger.info(f"Selected {len(selected_names)} out of {len(geometries)} objects:")
        for name in selected_names:
            logger.info(f"  ✓ {name}")
        logger.info("="*60 + "\n")

        return False

    # Register key callbacks for 0-9 and S
    for i in range(min(10, len(geometries))):
        key = ord(str(i))
        vis.register_key_callback(key, lambda v, idx=i: toggle_selection(v, idx))

    # Register 'S' for save
    vis.register_key_callback(ord('S'), save_selection)

    # Show instructions
    logger.info("\n" + "="*60)
    logger.info("=== Interactive Selection Mode ===")
    logger.info("="*60)
    logger.info("\nAvailable objects (press number to toggle):")
    for i, geom_dict in enumerate(geometries):
        if i < 10:
            logger.info(f"  [{i}] {geom_dict['name']}")
        else:
            logger.info(f"  [?] {geom_dict['name']} (index > 9, use manual mode)")
    logger.info("\nControls:")
    logger.info("  0-9: Toggle selection of object")
    logger.info("  S:   Save selection to JSON")
    logger.info("  Q:   Quit (or close window)")
    logger.info("\nSelected objects will turn GREEN")
    logger.info("="*60 + "\n")

    # Run visualizer
    vis.run()
    vis.destroy_window()

    # Return selected names
    selected_indices = sorted(list(selected_objects))
    return [geometries[i]["name"] for i in selected_indices]
