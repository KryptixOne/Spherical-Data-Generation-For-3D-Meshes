import copy
import json
import trimesh
import numpy as np

import os
import pandas as pd
from os.path import isfile, join
import open3d as o3d
import h5py
from acronym_tools import load_mesh, load_grasps, create_gripper_marker
from copy import deepcopy
from scipy.spatial.transform import Rotation as rot
from matplotlib import pyplot as plt
from shutil import rmtree
from scipy.spatial import Delaunay
from sklearn.preprocessing import normalize
import logging
import math as m

logger = logging.getLogger("trimesh")
logger.setLevel(logging.ERROR)
from collections import Counter


def create_random_rotation_vectors(num_of_vectors, random=True, degree=None):
    """Create an array of different rotation vectors to be applied on Nx3 datapoints as scipy rotational object
    :param num_of_vectors: Number of vectors to be produced
    :param random: default = True. Sets method to create random rotations
    :param degree: Only used if random = False. creats a number of vectors such that and entire horizontal
                    rotation is done at a certain degree
    :return rotationVectorArray: Array of rotation vectors
    """
    if random is True:
        rotationVectorArray = [rot.from_rotvec([[np.pi * round(2 * np.random.rand() - 1, 2),
                                                 np.pi * round(2 * np.random.rand() - 1, 2),
                                                 np.pi * round(2 * np.random.rand() - 1, 2)]]) for i in
                               range(num_of_vectors)]
        rotationVectorArray = np.asarray(rotationVectorArray)
    elif (random is False) and (degree is not None):
        num_of_vectors = int(360 / degree)
        if num_of_vectors < 1:
            raise ValueError("number of rotational vectors to be created is less than 1. Check input 'degree' argument")
        rotationVectorArray = [rot.from_rotvec([[0, np.pi * (degree / 180) * i, 0]]) for i in range(num_of_vectors)]
        rotationVectorArray = np.asarray(rotationVectorArray)
        pass

    return rotationVectorArray, num_of_vectors


def create_hemisphere(cx, cy, cz, r, resolution=180):
    """Creates a list of points in the shape of a hemisphere
    :param cx: Sphere center, X coord
    :param cy: Sphere center, Y coord
    :param cz: Sphere center, Z coord
    :param r: Sphere Radius
    :param resolution: hemisphere sampling resolution
    :return Array of [x,y,z] coordinates in form of hemisphere :
    """

    phi = np.linspace(0, 2 * np.pi, resolution)
    theta = np.linspace(0, 0.5 * np.pi, resolution)
    theta, phi = np.meshgrid(theta, phi)

    r_xy = r * np.sin(theta)
    x = cx + np.cos(phi) * r_xy
    y = cy + np.sin(phi) * r_xy
    z = cz + r * np.cos(theta)
    x = x.flatten()
    y = y.flatten()
    z = z.flatten()

    return np.transpose(np.stack([x, y, z]))


def in_hull(p, hull):
    """
    Test if points in `p` are in `hull`

    `p` should be a `NxK` coordinates of `N` points in `K` dimensions
    `hull` is either a scipy.spatial.Delaunay object or the `MxK` array of the
    coordinates of `M` points in `K`dimensions for which Delaunay triangulation
    will be computed
    """
    if not isinstance(hull, Delaunay):
        hull = Delaunay(hull)

    return hull.find_simplex(p) >= 0


def grasps_in_hemisphere(grasps_list_o3d, hemis_radius=1, hemis_resolution=60, rotation=None):
    """

    :param grasps_list_o3d: list of open3d objects for successful grasps
    :param hemis_radius: original radius of hemisphere
    :param hemis_resolution: original resolution for hemisphere
    :param rotation: scipy rotation object
    :return grasps_in_hull: List of grasps that are within the hemisphere.
    :return hemisphere_points: List of vertices used as origin points of the hemisphere
    """
    # creates Hemisphere and apply rotation to points if there is a rotation available
    hemisphere_points = create_hemisphere(0, 0, 0, hemis_radius + 1, hemis_resolution)  # create hemisphere for rays
    if rotation is not None:
        hemisphere_points = rotation.apply(hemisphere_points)

    grasps_COM = np.asarray([l.get_center() for l in grasps_list_o3d])

    inHullList = (in_hull(grasps_COM, hemisphere_points))
    grasps_in_hull_COM = grasps_COM[np.where(inHullList == True)]

    Existing_grasps_list_o3d = np.asarray(grasps_list_o3d)
    Existing_grasps_list_o3d = Existing_grasps_list_o3d[np.where(inHullList == True)]

    return grasps_in_hull_COM, hemisphere_points, Existing_grasps_list_o3d


def grasps_to_spherical(grasps_in_hull, rayOrigins, Existing_grasps_list_o3d, use_Two_Vectors=False,
                        use_spherical_coords=True):
    """Method takes as input grasp Centers, the origins for ray casting, and the original meshes used for obtaining the
    grasps centers.
    The original meshes are used to obtain the vertices used for creating the orientation vectors.
    The output of this method is two-fold. One img which contains purely positional markers, and a 6-channel orientation
    img which is used to define the orientation of the positional markers.

    :param grasps_in_hull: grasps in the hemisphere
    :param rayOrigins: ray origins of that hemisphere
    :param Existing_grasps_list_o3d: numpy array full of complete grasp objects
    :return: rays which contain the grasps position. Will be either 1 or 0,
    :return: 6channel orientation img of those grasps at each ray
    """

    def find_angle(v1, v2, vn):
        if v1.shape[0] == 1:
            x1 = v1[0]
            y1 = v1[1]
            z1 = v1[2]

            x2 = v2[0]
            y2 = v2[1]
            z2 = v2[2]

            xn = vn[0]
            yn = vn[1]
            zn = vn[2]
            dot = x1 * x2 + y1 * y2 + z1 * z2
            det = x1 * y2 * zn + x2 * yn * z1 + xn * y1 * z2 - z1 * y2 * xn - z2 * yn * x1 - zn * y1 * x2
            angle = m.atan2(det, dot) * 180 / np.pi
            angle = np.array([angle])
        else:
            elementWiseConcat = np.asarray(list((zip(v1, v2, vn))))
            dot = np.einsum('ij, ij->i', v1, v2)
            det = np.linalg.det(elementWiseConcat)
            angle = np.arctan2(det, dot) * 180 / np.pi

        return angle

    def get_Rejection_Vector(a, b):  # calculate rejection vector of a from b
        """rv = unitVect_rays --> a
            gv = unitVect_gripper--> b"""
        if a.shape[0] == 1:
            rej = a - np.dot((np.dot(a, b) / np.dot(b, b)), b)
        else:
            x = (np.einsum('ij, ij->i', a, b)) / (np.einsum('ij, ij->i', b, b))
            x_new = np.expand_dims(x, axis=0)
            proj = np.multiply(np.transpose(x_new), b)
            rej = a - proj

        return rej

    def cart2sph(Vectors):
        x_sq = np.square(Vectors)
        x = Vectors[:, 0]
        y = Vectors[:, 1]
        z = Vectors[:, 2]
        r = np.sqrt((x_sq[:, 0] + x_sq[:, 1] + x_sq[:, 2]))
        phi = np.arccos(z / r) * 180 / np.pi  # phi to degrees
        theta = np.arctan2(y, x) * 180 / np.pi  # theta
        return np.transpose(np.vstack((r, theta, phi)))

    def d(p1, q1, rs1):
        x = p1 - q1
        return np.linalg.norm(
            np.outer(np.dot(rs1 - q1, x) / np.dot(x, x), x) + q1 - rs1,
            axis=1)

    distList = []
    for endpoint in rayOrigins:
        p = np.array([0, 0, 0])  # p and q can have shape (n,) for any
        q = endpoint  # n>0, and rs can have shape (m,n)
        rs = grasps_in_hull

        distList.append(d(p, q, rs))

    dists = np.asarray(distList)
    minDistLoc = np.argmin(dists, axis=0)  # location of ray number
    distDict = dict(Counter(minDistLoc))

    spherical_positions = np.zeros((1, rayOrigins.shape[0]))
    spherical_positions[0][np.asarray(list(distDict.keys()))] = list(distDict.values())
    # spherical_positions[0][minDistLoc] = 1
    res = int(np.sqrt(rayOrigins.shape[0]))
    spherical_positions = spherical_positions.reshape((res, res))

    spherical_positions = np.expand_dims(spherical_positions, axis=0)

    if use_Two_Vectors == True:
        # Creates vectors that allow us to determine orientation of the grasp
        # vertices[45] amd vertices[32] are the midpoint of the end points of P1 and P2 of the gripper.

        left = np.asarray([i.vertices[45] for i in Existing_grasps_list_o3d])
        right = np.asarray([i.vertices[32] for i in Existing_grasps_list_o3d])

        left_vects = grasps_in_hull - left
        right_vects = grasps_in_hull - right
        Orientation_img_left = np.zeros((3, 1, rayOrigins.shape[0]))
        Orientation_img_right = np.zeros((3, 1, rayOrigins.shape[0]))

        Orientation_img_left[0][0][minDistLoc] = left_vects[:, 0]
        Orientation_img_left[1][0][minDistLoc] = left_vects[:, 1]
        Orientation_img_left[2][0][minDistLoc] = left_vects[:, 2]
        Orientation_img_right[0][0][minDistLoc] = right_vects[:, 0]
        Orientation_img_right[1][0][minDistLoc] = right_vects[:, 1]
        Orientation_img_right[2][0][minDistLoc] = right_vects[:, 2]

        Orientation_img_left = Orientation_img_left.reshape((3, res, res))
        Orientation_img_right = Orientation_img_right.reshape((3, res, res))
        Orientation_img = np.concatenate((Orientation_img_left, Orientation_img_right), axis=0)

    if use_spherical_coords == True:
        # obtain unit vectors of Rays used for Spherical Positions
        v = copy.deepcopy(rayOrigins[minDistLoc])
        unitVect_rays = normalize(v, norm="l2", axis=1)

        # obtain unit vectors of gripper orientation
        gripper_endpoint = np.asarray([i.vertices[1] for i in Existing_grasps_list_o3d])
        gripper_vect = grasps_in_hull - gripper_endpoint
        unitVect_gripper = normalize(gripper_vect, norm='l2', axis=1)

        # convert unitRays and unitGrippers to spherical coords
        spherical_rays = cart2sph(unitVect_rays)  # r, theta, phi
        spherical_gripper = cart2sph(unitVect_gripper)

        # ray_theta - gripper_theta = theta_diff
        # hence, ray_theta - theta_diff = gripper_theta
        theta_diff = spherical_rays[:, 1] - spherical_gripper[:, 1]
        phi_diff = spherical_rays[:, 2] - spherical_gripper[:, 2]

        # obtain rejection vectors which are orthogonal to unit gripper and show direction toward rayVect
        rejection_Vectors = get_Rejection_Vector(unitVect_rays, unitVect_gripper)

        # obtain unitvector of com-->p1, Then obtain rejection vectors
        left = np.asarray([i.vertices[45] for i in Existing_grasps_list_o3d])
        left_vects = grasps_in_hull - left
        left_vects_normalized = normalize(left_vects, norm='l2', axis=1)
        orient_rejection = get_Rejection_Vector(left_vects_normalized, unitVect_gripper)
        rotation_angles = find_angle(rejection_Vectors, orient_rejection, unitVect_gripper)

        Orientation_img = np.zeros((3, 1, rayOrigins.shape[0]))
        Orientation_img[0][0][minDistLoc] = theta_diff
        Orientation_img[1][0][minDistLoc] = phi_diff
        Orientation_img[2][0][minDistLoc] = rotation_angles
        Orientation_img = Orientation_img.reshape((3, res, res))

    return spherical_positions, Orientation_img


def create_spherical_depth_data(mesh, resolution=60, sampling_radius=1, rotations=None):
    """ Create a spherical depth image by means of ray-casting onto a 3-D mesh and recording hit distances
    :param mesh: open3d mesh model that has been centered around the origin and has unit vector length for all points
    :param resolution: resolution that is being used to sample the mesh
    :param sampling_radius: radius used to create size of hemisphere used for sampling the mesh
    :param rotations: (optional) scipy rotational object
    :return spherical_depth_data_img: spherical depth image in numpy format
    """

    rayOrigins = create_hemisphere(0, 0, 0, sampling_radius, resolution)  # create hemisphere for rays
    if rotations is not None:
        rayOrigins = rotations.apply(rayOrigins)

    rayDirections = rayOrigins * (-1)
    rayCastingVectors = np.concatenate((rayOrigins, rayDirections), axis=1)
    rays = o3d.core.Tensor(rayCastingVectors, dtype=o3d.core.Dtype.Float32)

    scene = o3d.t.geometry.RaycastingScene()
    t_mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)

    mesh_id = scene.add_triangles(t_mesh)

    ans = scene.cast_rays(rays)

    hit_distance = ans['t_hit'].numpy()
    reshaped_dist = np.reshape(hit_distance, (resolution, resolution))
    df = pd.DataFrame(reshaped_dist)
    df.replace(np.inf, np.nan, inplace=True)
    df = df.interpolate(limit_direction='both')
    spherical_depth_data_img = df.to_numpy()
    spherical_depth_data_img = np.expand_dims(spherical_depth_data_img, axis=0)

    return spherical_depth_data_img


# No longer used
def load_o3d_mesh(filename, mesh_root_dir, name_of_meshes_available=None, scale=None, orig_center=None):
    """Load a mesh from a JSON or HDF5 file from the grasp dataset. The mesh will be scaled accordingly.
    :param filename: JSON or HDF5 file name
    :param mesh_root_dir: directory holding 'meshes' folder
    :param name_of_meshes_available: (optional) list of available meshes
    :param scale: scale (float, optional): If specified, use this as scale instead of value from the file. Default None.
    :return obj_mesh_o3d: Mesh of the loaded object, scaled according to grasp data
    :return obj_mesh_o3d_normalized: Mesh of loaded object, scaled to unit vector length and centered around the origin
    """

    if filename.endswith(".json"):
        data = json.load(open(filename, "r"))
        mesh_fname = data["object"].decode('utf-8')
        mesh_scale = data["object_scale"] if scale is None else scale

    elif filename.endswith(".h5"):
        data = h5py.File(os.path.join(mesh_root_dir, filename), 'r')
        mesh_fname = data["object/file"][()].decode('utf-8')
        # For use with shapenet. Data file not organized by ID name.
        mesh_fname = 'meshes/' + os.path.basename(mesh_fname)
        mesh_scale = data["object/scale"][()] if scale is None else scale

    else:
        raise RuntimeError("Unknown file ending:", filename)

    # Mesh according to input dataa from Dataset
    obj_mesh_o3d = o3d.io.read_triangle_mesh(os.path.join(mesh_root_dir, mesh_fname))
    obj_mesh_o3d.compute_vertex_normals()
    obj_mesh_o3d.paint_uniform_color([1, 0.706, 0])
    obj_mesh_o3d.scale(mesh_scale, center=obj_mesh_o3d.get_center())

    # Mesh normalized to unit vector length and centered around 0
    obj_mesh_o3d_normalized = deepcopy(obj_mesh_o3d)
    obj_mesh_o3d_normalized.translate((0, 0, 0), relative=False)
    scaler = 1 / (np.amax(np.linalg.norm(np.asarray(obj_mesh_o3d.vertices), axis=1)))
    obj_mesh_o3d_normalized.scale(scaler, center=obj_mesh_o3d_normalized.get_center())

    return obj_mesh_o3d, obj_mesh_o3d_normalized, scaler


def trimesh_to_o3d(trimesh_mesh, grasp_mesh_trimesh, mesh_path):
    """Converts Trimesh_Object mesh and Trimesh Grasps mesh to o3d mesh through a trivial write-read method.

    :param trimesh_mesh:
    :param grasp_mesh_trimesh:
    :param mesh_path:
    :return:
    """
    # remove directory and files if it already exists
    grasps_o3d = []
    obj_mesh_o3d = []
    try:
        rmtree(mesh_path)
    except FileNotFoundError:
        None

    os.mkdir(mesh_path)  # create temp directory housing the file

    for meshNumber in range(len(grasp_mesh_trimesh)):
        filename = 'grasp' + str(meshNumber) + '.obj'
        grasp_mesh_trimesh[meshNumber].export(os.path.join(mesh_path, filename))
        temp_o3d = o3d.io.read_triangle_mesh(os.path.join(mesh_path, filename))

        temp_o3d.compute_vertex_normals()
        temp_o3d.paint_uniform_color([1, 0.706, 0])
        grasps_o3d.append(temp_o3d)

    meshFilename = 'mesh.obj'
    trimesh_mesh.export(os.path.join(mesh_path, meshFilename))
    mesh_o3d = o3d.io.read_triangle_mesh(os.path.join(mesh_path, meshFilename))
    mesh_o3d.compute_vertex_normals()
    mesh_o3d.paint_uniform_color([1, 0.706, 0])
    obj_mesh_o3d.append(mesh_o3d)

    return grasps_o3d, obj_mesh_o3d


def as_mesh(scene_or_mesh):
    """
    Convert a possible scene to a mesh.

    If conversion occurs, the returned mesh has only vertex and face data.
    """
    if isinstance(scene_or_mesh, trimesh.Scene):
        if len(scene_or_mesh.geometry) == 0:
            mesh = None  # empty scene
        else:
            # we lose texture information here
            mesh = trimesh.util.concatenate(
                tuple(trimesh.Trimesh(vertices=g.vertices, faces=g.faces)
                      for g in scene_or_mesh.geometry.values()))
    else:
        assert (isinstance(mesh, trimesh.Trimesh))
        mesh = scene_or_mesh

    return mesh


def main():
    import time
    t0 = time.time()
    num_grasps_to_display = 40
    num_of_rotations = 10
    randomized_rotation = False
    rotational_degree = 60  # degree of rotation for hemisphere
    directoryForMeshes = 'D:/Thesis/Data/OrigData'
    directoryForGrasps = 'D:/Thesis/Data/OrigData/grasps'
    directoryHoldingMeshFiles = 'D:/Thesis/Data/OrigData/meshes'
    directoryTemp = r'D:/Thesis/Thesis Code/temp'  # DO NOT CHANGE. IF WRONG DIRECTORY, LOSS OF FILES WILL OCCUR
    l = 0  # break test counter

    graspFiles = ['grasps/' + f for f in os.listdir(directoryForGrasps) if isfile(join(directoryForGrasps, f))]
    # meshFiles = [m for m in os.listdir(directoryHoldingMeshFiles) if isfile(join(directoryHoldingMeshFiles, m))]

    img_collection = {}
    for f in graspFiles:
        # load object mesh
        obj_mesh = load_mesh(f, mesh_root_dir=directoryForMeshes)

        # get transformations and quality of all simulated grasps
        T, success = load_grasps(os.path.join(directoryForMeshes, f))

        # create visual markers for grasps
        successful_grasps = [
            create_gripper_marker(color=[0, 255, 0]).apply_transform(t)
            for t in T[np.random.choice(np.where(success == 1)[0], num_grasps_to_display)]
        ]

        # Centering around origin and scaling to unit vector max length

        k = obj_mesh.centroid

        translation_matrix = np.array([[1, 0, 0, -1 * k[0]],
                                       [0, 1, 0, -1 * k[1]],
                                       [0, 0, 1, -1 * k[2]],
                                       [0, 0, 0, 1]])
        # move all grasps such that object geometric center lies at origin

        obj_mesh.apply_transform(translation_matrix)

        scaler = 1 / (np.amax(np.linalg.norm(np.asarray(obj_mesh.vertices), axis=1)))
        obj_mesh.apply_scale(scaler)
        successful_grasps = [(t.apply_transform(translation_matrix)).apply_scale(scaler) for t in successful_grasps]

        # Now grasps and obj_mesh are centered around origin and have been scaled in unit vector length.
        # Convert Trimesh objects to O3D for use with previously created algorithm.

        successful_grasps_o3d, obj_mesh_o3d_normalized = trimesh_to_o3d(trimesh_mesh=obj_mesh,
                                                                        grasp_mesh_trimesh=successful_grasps,
                                                                        mesh_path=directoryTemp
                                                                        )

        # o3d.visualization.draw_geometries(obj_mesh_o3d_normalized + successful_grasps_o3d, mesh_show_back_face=True)

        # rotate hemisphere around object to sample depth in different ways.
        if num_of_rotations is not None:
            img_list = []

            rotationsArray, num_of_rotations = create_random_rotation_vectors(num_of_rotations,
                                                                              random=randomized_rotation,
                                                                              degree=rotational_degree)
            for i in range(num_of_rotations):
                # create spherical Data Img
                spherical_data_img = create_spherical_depth_data(obj_mesh_o3d_normalized[0],
                                                                 rotations=rotationsArray[i][0])
                # img_list.append(spherical_data_img)

                # Create Positional and orient img
                available_grasps, positionalHemisphere, Existing_grasps_list_o3d = grasps_in_hemisphere(
                    successful_grasps_o3d, rotation=rotationsArray[i][0])
                positional_img, orentiation_img = grasps_to_spherical(available_grasps, positionalHemisphere,
                                                                      Existing_grasps_list_o3d)

                # For visualization purposes
                """
                plt.imshow(spherical_data_img.squeeze(0), interpolation='nearest')
                plt.show()
                plt.imshow(positional_img.squeeze(0), interpolation='nearest')
                plt.show()
                plt.imshow(orentiation_img[0,:,:], interpolation='nearest')
                plt.show()
                plt.imshow(orentiation_img[1,:,:], interpolation='nearest')
                plt.show()
                plt.imshow(orentiation_img[2,:,:], interpolation='nearest')
                plt.show()
                """

                complete_img_data = np.concatenate((spherical_data_img, positional_img, orentiation_img),
                                                   axis=0)  # (8xResxRes)
                img_list.append(complete_img_data)
            img_collection[f] = img_list


        else:

            spherical_data_img = create_spherical_depth_data(obj_mesh_o3d_normalized)

            available_grasps, positionalHemisphere, Existing_grasps_list_o3d = grasps_in_hemisphere(
                successful_grasps_o3d, rotation=rotationsArray[i][0])

            positional_img, Orentiation_img = grasps_to_spherical(available_grasps, positionalHemisphere,
                                                                  Existing_grasps_list_o3d)

            complete_img_data = np.concatenate((spherical_data_img, positional_img, orentiation_img),
                                               axis=0)  # (8xResxRes)
            img_collection[f] = complete_img_data

        l = l + 1
        print(l - 1)
        if l == 10:
            t1 = time.time()
            print(t1 - t0)
            break


if __name__ == "__main__":
    main()
