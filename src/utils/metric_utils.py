import numpy as np
import os
import cv2
import torch
import os.path as osp
from loguru import logger
from time import time
from scipy import spatial
from src.utils.sample_points_on_cad import load_points_from_cad, model_diameter_from_bbox
from .colmap.read_write_model import qvec2rotmat
from .colmap.eval_helper import quaternion_from_matrix

import open3d as o3d
import json
from wis3d import Wis3D

def convert_pose2T(pose):
    # pose: [R: 3*3, t: 3]
    R, t = pose
    return np.concatenate(
        [np.concatenate([R, t[:, None]], axis=1), [[0, 0, 0, 1]]], axis=0
    )  # 4*4

def angle_error_vec(v1, v2):
    n = np.linalg.norm(v1) * np.linalg.norm(v2)
    return np.rad2deg(np.arccos(np.clip(np.dot(v1, v2) / n, -1.0, 1.0)))


def angle_error_mat(R1, R2):
    cos = (np.trace(np.dot(R1.T, R2)) - 1) / 2
    cos = np.clip(cos, -1.0, 1.0)  # numercial errors can make it out of bounds
    return np.rad2deg(np.abs(np.arccos(cos)))

def projection_2d_error(model_3D_pts, pose_pred, pose_targets, K):
    def project(xyz, K, RT):
        """
        NOTE: need to use original K
        xyz: [N, 3]
        K: [3, 3]
        RT: [3, 4]
        """
        xyz = np.dot(xyz, RT[:, :3].T) + RT[:, 3:].T
        xyz = np.dot(xyz, K.T)
        xy = xyz[:, :2] / xyz[:, 2:]
        return xy

    # Dim check:
    if pose_pred.shape[0] == 4:
        pose_pred = pose_pred[:3]
    if pose_targets.shape[0] == 4:
        pose_targets = pose_targets[:3]

    model_2d_pred = project(model_3D_pts, K, pose_pred) # pose_pred: 3*4
    model_2d_targets = project(model_3D_pts, K, pose_targets)
    proj_mean_diff = np.mean(np.linalg.norm(model_2d_pred - model_2d_targets, axis=-1))
    return proj_mean_diff

def add_metric(model_3D_pts, diameter, pose_pred, pose_target, percentage=0.1, syn=False, model_unit='m'):
    # Dim check:
    if pose_pred.shape[0] == 4:
        pose_pred = pose_pred[:3]
    if pose_target.shape[0] == 4:
        pose_target = pose_target[:3]
    
    # if model_unit == 'm':
    #     model_3D_pts *= 1000
    #     diameter *= 1000
    #     pose_pred[:,3] *= 1000
    #     pose_target[:,3] *= 1000
        
    #     max_model_coord = np.max(model_3D_pts, axis=0)
    #     min_model_coord = np.min(model_3D_pts, axis=0)
    #     diameter_from_model = np.linalg.norm(max_model_coord - min_model_coord)
    # elif model_unit == 'mm':
    #     pass

    diameter_thres = diameter * percentage
    model_pred = np.dot(model_3D_pts, pose_pred[:, :3].T) + pose_pred[:, 3]
    model_target = np.dot(model_3D_pts, pose_target[:, :3].T) + pose_target[:, 3]
    
    if syn:
        mean_dist_index = spatial.cKDTree(model_pred)
        mean_dist, _ = mean_dist_index.query(model_target, k=1)
        mean_dist = np.mean(mean_dist)
    else:
        mean_dist = np.mean(np.linalg.norm(model_pred - model_target, axis=-1))
    if mean_dist < diameter_thres:
        return True
    else:
        return False


# Evaluate query pose errors
def query_pose_error(pose_pred, pose_gt, unit='m'):
    """
    Input:
    -----------
    pose_pred: np.array 3*4 or 4*4
    pose_gt: np.array 3*4 or 4*4
    """
    # Dim check:
    if pose_pred.shape[0] == 4:
        pose_pred = pose_pred[:3]
    if pose_gt.shape[0] == 4:
        pose_gt = pose_gt[:3]

    # Convert results' unit to cm
    if unit == 'm':
        translation_distance = np.linalg.norm(pose_pred[:, 3] - pose_gt[:, 3]) * 100
    elif unit == 'cm':
        translation_distance = np.linalg.norm(pose_pred[:, 3] - pose_gt[:, 3])
    elif unit == 'mm':
        translation_distance = np.linalg.norm(pose_pred[:, 3] - pose_gt[:, 3]) / 10
    else:
        raise NotImplementedError

    rotation_diff = np.dot(pose_pred[:, :3], pose_gt[:, :3].T)
    trace = np.trace(rotation_diff)
    trace = trace if trace <= 3 else 3
    angular_distance = np.rad2deg(np.arccos((trace - 1.0) / 2.0))
    return angular_distance, translation_distance


def ransac_PnP(
    K,
    pts_2d,
    pts_3d,
    scale=1,
    pnp_reprojection_error=5,
    img_hw=None,
    use_pycolmap_ransac=False,
):
    """ solve pnp """
    try:
        import pycolmap
    except:
        logger.warning(f"pycolmap is not installed, use opencv ransacPnP instead")
        use_pycolmap_ransac = False

    if use_pycolmap_ransac:
        import pycolmap

        assert img_hw is not None and len(img_hw) == 2

        pts_2d = list(np.ascontiguousarray(pts_2d.astype(np.float64))[..., None]) # List(2*1)
        pts_3d = list(np.ascontiguousarray(pts_3d.astype(np.float64))[..., None]) # List(3*1)
        K = K.astype(np.float64)
        # Colmap pnp with non-linear refinement
        focal_length = K[0, 0]
        cx = K[0, 2]
        cy = K[1, 2]
        cfg = {
            "model": "SIMPLE_PINHOLE",
            "width": int(img_hw[1]),
            "height": int(img_hw[0]),
            "params": [focal_length, cx, cy],
        }

        ret = pycolmap.absolute_pose_estimation(
            pts_2d, pts_3d, cfg, max_error_px=float(pnp_reprojection_error)
        )
        qvec = ret["qvec"]
        tvec = ret["tvec"]
        pose_homo = convert_pose2T([qvec2rotmat(qvec), tvec])
        # Make inliers:
        inliers = ret['inliers']
        if len(inliers) == 0:
            inliers = np.array([]).astype(np.bool)
        else:
            index = np.arange(0, len(pts_3d))
            inliers = index[inliers]

        return pose_homo[:3], pose_homo, inliers, True
    else:
        dist_coeffs = np.zeros(shape=[8, 1], dtype="float64")

        pts_2d = np.ascontiguousarray(pts_2d.astype(np.float64))
        pts_3d = np.ascontiguousarray(pts_3d.astype(np.float64))
        K = K.astype(np.float64)

        pts_3d *= scale
        state = None
        try:
            _, rvec, tvec, inliers = cv2.solvePnPRansac(
                pts_3d,
                pts_2d,
                K,
                dist_coeffs,
                reprojectionError=pnp_reprojection_error,
                iterationsCount=10000,
                flags=cv2.SOLVEPNP_EPNP,
            )

            rotation = cv2.Rodrigues(rvec)[0]

            tvec /= scale
            pose = np.concatenate([rotation, tvec], axis=-1)
            pose_homo = np.concatenate([pose, np.array([[0, 0, 0, 1]])], axis=0)

            if inliers is None:
                inliers = np.array([]).astype(np.bool)
            state = True

            return pose, pose_homo, inliers, state
        except cv2.error:
            state = False
            return np.eye(4)[:3], np.eye(4), np.array([]).astype(np.bool), state


@torch.no_grad()
def compute_query_pose_errors(
    data, configs, training=False
):
    """
    Update:
        data(dict):{
            "R_errs": []
            "t_errs": []
            "inliers": []
        }
    """
    model_unit = configs['model_unit'] if 'model_unit' in configs else 'm'

    m_bids = data["m_bids"].cpu().numpy()
    mkpts_3d = data["mkpts_3d_db"].cpu().numpy()
    mkpts_query = data["mkpts_query_f"].cpu().numpy()
    img_orig_size = (
        torch.tensor(data["q_hw_i"]).numpy() * data["query_image_scale"].cpu().numpy()
    )  # B*2
    query_K = data["query_intrinsic"].cpu().numpy()
    query_pose_gt = data["query_pose_gt"].cpu().numpy()  # B*4*4

    data.update({"R_errs": [], "t_errs": [], "inliers": []})
    data.update({"R_errs_c": [], "t_errs_c": [], "inliers_c": []})

    # Prepare query model for eval ADD metric
    if 'eval_ADD_metric' in configs:
        if configs['eval_ADD_metric'] and not training:
            image_path = data['query_image_path']
            adds = True if ('0810-' in image_path) or ('0811-' in image_path) else False # Symmetric object in LINEMOD
            query_K_origin = data["query_intrinsic_origin"].cpu().numpy()
            model_path = osp.join(image_path.rsplit('/', 3)[0], 'model_eval.ply')
            if not osp.exists(model_path):
                model_path = osp.join(image_path.rsplit('/', 3)[0], 'model.ply')
            diameter_file_path = osp.join(image_path.rsplit('/', 3)[0], 'diameter.txt')
            if not osp.exists(model_path):
                logger.error(f'want to eval add metric, however model_eval.ply path:{model_path} not exists!')
            else:
                # Load model:
                model_vertices, bbox = load_points_from_cad(model_path) # N*3
                # Load diameter:
                if osp.exists(diameter_file_path):
                    diameter = np.loadtxt(diameter_file_path)
                else:
                    diameter = model_diameter_from_bbox(bbox)
                
                data.update({"ADD":[], "proj2D":[]})

    pose_pred = []
    for bs in range(query_K.shape[0]):
        mask = m_bids == bs

        mkpts_query_f = mkpts_query[mask]
        query_pose_pred, query_pose_pred_homo, inliers, state = ransac_PnP(
            query_K[bs],
            mkpts_query_f,
            mkpts_3d[mask],
            scale=configs["point_cloud_rescale"],
            img_hw=img_orig_size[bs].tolist(),
            pnp_reprojection_error=configs["pnp_reprojection_error"],
            use_pycolmap_ransac=configs["use_pycolmap_ransac"],
        )
        pose_pred.append(query_pose_pred_homo)

        if query_pose_pred is None:
            data["R_errs"].append(np.inf)
            data["t_errs"].append(np.inf)
            data["inliers"].append(np.array([]).astype(np.bool))
            if "ADD" in data:
                data['ADD'].append(False)
        else:
            R_err, t_err = query_pose_error(query_pose_pred, query_pose_gt[bs], unit=model_unit)
            data["R_errs"].append(R_err)
            data["t_errs"].append(t_err)
            data["inliers"].append(inliers)

            if "ADD" in data:
                add_result = add_metric(model_vertices, diameter, pose_pred=query_pose_pred, pose_target=query_pose_gt[bs], syn=adds)
                data["ADD"].append(add_result)

                proj2d_result = projection_2d_error(model_vertices, pose_pred=query_pose_pred, pose_targets=query_pose_gt[bs], K=query_K_origin[bs])
                data['proj2D'].append(proj2d_result)

    if 'wis3d' in configs and configs['wis3d']:
        wis3d(data, mkpts_3d, mkpts_query_f, inliers.reshape(-1)) 

    pose_pred = np.stack(pose_pred)  # [B*4*4]

    if 'visualize' in configs and configs['visualize'] == 'LINEMOD':
        pass

    elif 'visualize' in configs and configs['visualize'] == 'ONEPOSE':
        from src.utils.vis_utils import save_demo_image, save_demo_image_GT
        K = data['query_intrinsic_origin'][0].cpu()
        image_path = data['query_image_path']
        
        box3d = np.loadtxt(osp.join(image_path.rsplit('/', 3)[0], "box3d_corners.txt"))
        
        tmp = image_path.rsplit('/', 6)
        save_path = osp.join(tmp[0], f"output/{tmp[2]}/{tmp[3]}/{tmp[-1]}")

        query_pose_gt = data['query_pose_gt'].cpu()

        ## pred
        save_demo_image(pose_pred[0], K, image_path, box3d, save_path=save_path)
        ## GT
        save_demo_image_GT(query_pose_gt[0].numpy(), K, save_path, box3d, save_path=save_path)
        ## put Text
        img = cv2.imread(save_path)

        cv2.putText(img, f"{data['R_errs'][0]}", (400, 400), 5, 1,(0, 0, 255), 1, cv2.LINE_AA, False)
        cv2.putText(img, f"{data['t_errs'][0]}", (400,450), 5, 1,(0, 0, 255), 1, cv2.LINE_AA, False)
        cv2.imwrite(save_path, img)

    data.update({"pose_pred": pose_pred})


def aggregate_metrics(metrics, pose_thres=[1, 3, 5], proj2d_thres=5):
    """ Aggregate metrics for the whole dataset:
    (This method should be called once per dataset)
    """
    R_errs = metrics["R_errs"]
    t_errs = metrics["t_errs"]

    agg_metric = {}
    for pose_threshold in pose_thres:
        agg_metric[f"{pose_threshold}cm@{pose_threshold}degree"] = np.mean(
            (np.array(R_errs) < pose_threshold) & (np.array(t_errs) < pose_threshold)
        )

    if "ADD_metric" in metrics:
        ADD_metric = metrics['ADD_metric']
        agg_metric["ADD metric"] = np.mean(ADD_metric)

        proj2D_metric = metrics['proj2D_metric']
        agg_metric["proj2D metric"] = np.mean(np.array(proj2D_metric) < proj2d_thres)

    return agg_metric

def wis3d(data, mkpts_3d, mkpts_query_f, inliers):
    def render2(ply_path, K, P, output):
        pcd = o3d.io.read_point_cloud(ply_path)

        fx = K[0,0]
        fy = K[1,1]
        cx = K[0,2]
        cy = K[1,2]

        render = o3d.visualization.rendering.OffscreenRenderer(512, 512)
        render.scene.set_background([1,1, 1, 1])



        mat = o3d.visualization.rendering.MaterialRecord()
        mat.point_size = 3.0
        mat.shader = 'defaultUnlit'

        render.scene.add_geometry("pcd", pcd, mat)

        render.setup_camera(K, P, 512, 512)

        render.scene.view.set_post_processing(False)
        img = render.render_to_image()
        img = np.asarray(img)

        cv2.imwrite(output, cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    def proj_3d(mkpts_3d, K, P, output, inlier=False):
        if inlier:
            h = np.ones((mkpts_3d.shape[0], 1))
            mkpts_3d = np.hstack((mkpts_3d, h))

            mkpts_proj = ((K@P)@mkpts_3d.transpose(1,0)).transpose(1,0)

            mkpts_proj = mkpts_proj[:,:2]/mkpts_proj[:,[-1]]

            np.savetxt(output, mkpts_proj)

        else:
            h = np.ones((mkpts_3d.shape[0], 1))
            mkpts_3d = np.hstack((mkpts_3d, h))

            mkpts_proj = ((K@P)@mkpts_3d.transpose(1,0)).transpose(1,0)

            mkpts_proj = mkpts_proj[:,:2]/mkpts_proj[:,[-1]]

            np.savetxt(output, mkpts_proj)

    def make_wis3d_json(mkpts_3d_proj_path, mkpts_2d, output, inlier=False):
        if inlier:
            dic = {}
            mkpts_3d = np.loadtxt(mkpts_3d_proj_path)


            ## calculate l2 distance
            dist = []
            for i in range(len(mkpts_2d)):
                dist.append(np.linalg.norm(mkpts_3d[i] - mkpts_2d[i]))

            dic['kpts0'] = mkpts_3d.tolist()
            dic['kpts1'] = mkpts_2d.tolist()
            dic['l2_dist'] = dist

            with open(output, 'w') as fp:
                json.dump(dic, fp)
        else:
            dic = {}
            mkpts_3d = np.loadtxt(mkpts_3d_proj_path)


            ## calculate l2 distance
            dist = []
            for i in range(len(mkpts_2d)):
                dist.append(np.linalg.norm(mkpts_3d[i] - mkpts_2d[i]))

            dic['kpts0'] = mkpts_3d.tolist()
            dic['kpts1'] = mkpts_2d.tolist()
            dic['l2_dist'] = dist

            with open(output, 'w') as fp:
                json.dump(dic, fp)

    def vis(index, rendered_image_path, image_path, keypoint_correspondences_path, visual_path, inlier=False):
        if inlier:
            vis_dir = os.path.abspath(visual_path)
            wis3d = Wis3D(vis_dir, index+'_inlier')

            img3d_path = os.path.abspath(rendered_image_path)
            img2d_path = os.path.abspath(image_path)
            keypoints_path = os.path.abspath(keypoint_correspondences_path)
            with open(keypoints_path, 'r') as f:
                keypoints_data = json.load(f)

            wis3d.add_keypoint_correspondences(img3d_path,
                                            img2d_path,
                                            kpts0 = keypoints_data["kpts0"],
                                            kpts1 = keypoints_data["kpts1"],
                                            metrics={"l2_dist":keypoints_data['l2_dist']}
                                            )            
        else:
            vis_dir = os.path.abspath(visual_path)
            wis3d = Wis3D(vis_dir, index)

            img3d_path = os.path.abspath(rendered_image_path)
            img2d_path = os.path.abspath(image_path)
            keypoints_path = os.path.abspath(keypoint_correspondences_path)
            with open(keypoints_path, 'r') as f:
                keypoints_data = json.load(f)

            wis3d.add_keypoint_correspondences(img3d_path,
                                            img2d_path,
                                            kpts0 = keypoints_data["kpts0"],
                                            kpts1 = keypoints_data["kpts1"],
                                            metrics={"l2_dist":keypoints_data['l2_dist']}
                                            )
        
    obj = data['query_image_path'].split('/')[3]
    index = data['query_image_path'].split('/')[-1].split('.')[0]
    mkpts_3d = mkpts_3d
    mkpts_2d = mkpts_query_f
    K = data['query_intrinsic'][0].cpu().numpy()
    P = data['query_pose_gt'][0].cpu().numpy()
    image_path = data['query_image_path']
    # ply_path = f'/home/chanho/6dof/onepose_plus/data/datasets/sfm_output/outputs_softmax_loftr_loftr/vis3d/{obj}/00000/point_clouds/filtered_pointcloud.ply'
    ply_path = f'/home/chanho/6dof/opp_part/data/datasets/sfm_output/outputs_softmax_loftr_loftr/{obj}/tkl_model/tl-5.ply'

    

    base_path = '/home/chanho/desktop/tool/matching_vis/' + obj + '/'
    os.makedirs(base_path, exist_ok=True)

    os.makedirs(base_path+'rendered_image/', exist_ok=True)
    os.makedirs(base_path + 'mkpts_3d_proj/', exist_ok=True)
    os.makedirs(base_path + 'keypoint_correspondences/', exist_ok=True)
    os.makedirs(base_path + 'visual/', exist_ok=True)

    os.makedirs(base_path + 'mkpts_3d_proj_inlier/', exist_ok=True)
    os.makedirs(base_path + 'keypoint_correspondences_inlier/', exist_ok=True)



    rendered_image_path = base_path+'rendered_image/' + f'{index}.png'
    # mkpts_3d_path = base_path + 'mkpts_3d/' + f'{index}.txt'
    # mkpts_2d_path = base_path + 'mkpts_2d/' + f'{index}.txt'
    mkpts_3d_proj_path = base_path + 'mkpts_3d_proj/'+f'{index}.txt'
    mkpts_3d_proj_inlier_path = base_path + 'mkpts_3d_proj_inlier/'+f'{index}.txt'
    keypoint_correspondences_path = base_path + 'keypoint_correspondences/' + f'{index}.json'
    keypoint_correspondences_inlier_path = base_path + 'keypoint_correspondences_inlier/' + f'{index}.json'
    visual_path = base_path + 'visual/'






    render2(ply_path, K, P, rendered_image_path)
    proj_3d(mkpts_3d, K, P[:3], mkpts_3d_proj_path)
    make_wis3d_json(mkpts_3d_proj_path, mkpts_2d, keypoint_correspondences_path)
    vis(index, rendered_image_path, image_path, keypoint_correspondences_path, visual_path)

    #inlier
    proj_3d(mkpts_3d[inliers], K, P[:3], mkpts_3d_proj_inlier_path, True)
    make_wis3d_json(mkpts_3d_proj_inlier_path, mkpts_2d[inliers], keypoint_correspondences_inlier_path, True)
    vis(index, rendered_image_path, image_path, keypoint_correspondences_inlier_path, visual_path, True)