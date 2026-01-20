import numpy as np
try:
    import cv2
except Exception:  # pragma: no cover - runtime dependency check
    cv2 = None

from .constants import MATCHER_TYPES


def _ensure_cv2():
    if cv2 is None:
        raise RuntimeError("opencv-python is required for feature alignment. Please install opencv-python.")


def _create_detector(matcher_type, feature_count):
    if matcher_type == "orb":
        return cv2.ORB_create(nfeatures=feature_count), cv2.NORM_HAMMING
    if matcher_type == "akaze":
        return cv2.AKAZE_create(), cv2.NORM_HAMMING
    if matcher_type == "sift":
        if not hasattr(cv2, "SIFT_create"):
            return None, None
        return cv2.SIFT_create(nfeatures=feature_count), cv2.NORM_L2
    return None, None


def detect_and_match(
    background_np,
    overlay_np,
    bg_mask_np,
    ov_mask_np,
    feature_count,
    min_matches,
    good_match_percent,
    matcher_type,
    color_mode,
    lab_channels,
):
    _ensure_cv2()

    if lab_channels not in ("l", "lab"):
        lab_channels = "lab"

    if matcher_type not in MATCHER_TYPES:
        return None, None, f"Unsupported matcher: {matcher_type}"

    if background_np.shape[2] != 3 or overlay_np.shape[2] not in (3, 4):
        return None, None, "Only 3-channel background and 3/4-channel overlay images are supported."

    if color_mode == "lab":
        bg_lab = cv2.cvtColor(background_np[:, :, :3], cv2.COLOR_RGB2LAB)
        ov_lab = cv2.cvtColor(overlay_np[:, :, :3], cv2.COLOR_RGB2LAB)
        if lab_channels == "l":
            bg_gray = bg_lab[:, :, 0]
            ov_gray = ov_lab[:, :, 0]
        else:
            bg_gray = bg_lab
            ov_gray = ov_lab
    elif color_mode == "lab_l":
        bg_gray = cv2.cvtColor(background_np[:, :, :3], cv2.COLOR_RGB2LAB)[:, :, 0]
        ov_gray = cv2.cvtColor(overlay_np[:, :, :3], cv2.COLOR_RGB2LAB)[:, :, 0]
    else:
        bg_gray = cv2.cvtColor(background_np[:, :, :3], cv2.COLOR_RGB2GRAY)
        ov_gray = cv2.cvtColor(overlay_np[:, :, :3], cv2.COLOR_RGB2GRAY)

    detector, norm_type = _create_detector(matcher_type, feature_count)
    if detector is None:
        return None, None, f"Matcher {matcher_type} is not available in this OpenCV build."

    ov_keypoints, ov_desc = detector.detectAndCompute(ov_gray, ov_mask_np)
    bg_keypoints, bg_desc = detector.detectAndCompute(bg_gray, bg_mask_np)

    if ov_desc is None or bg_desc is None or len(ov_keypoints) < min_matches or len(bg_keypoints) < min_matches:
        return None, None, "Not enough keypoints for alignment."

    if matcher_type in ("sift",):
        matcher = cv2.BFMatcher(norm_type)
        knn_matches = matcher.knnMatch(ov_desc, bg_desc, k=2)
        ratio = 0.75
        matches = [m for m, n in knn_matches if m.distance < ratio * n.distance]
        if len(matches) < min_matches:
            return None, None, "Not enough matches for alignment."
    else:
        matcher = cv2.BFMatcher(norm_type, crossCheck=True)
        matches = matcher.match(ov_desc, bg_desc)
        if len(matches) < min_matches:
            return None, None, "Not enough matches for alignment."

    matches = sorted(matches, key=lambda match: match.distance)
    keep = max(min_matches, int(len(matches) * good_match_percent))
    matches = matches[:keep]

    ov_points = np.float32([ov_keypoints[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    bg_points = np.float32([bg_keypoints[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    return ov_points, bg_points, "ok"


def estimate_affine(ov_points, bg_points, ransac_thresh, scale_mode, min_inliers):
    _ensure_cv2()
    if scale_mode == "preserve_aspect":
        matrix, inliers = cv2.estimateAffinePartial2D(
            ov_points,
            bg_points,
            method=cv2.RANSAC,
            ransacReprojThreshold=ransac_thresh,
        )
    else:
        matrix, inliers = cv2.estimateAffine2D(
            ov_points,
            bg_points,
            method=cv2.RANSAC,
            ransacReprojThreshold=ransac_thresh,
        )
    if matrix is None:
        return None, "Could not estimate affine transform."
    inlier_count = int(inliers.sum()) if inliers is not None else 0
    if inlier_count < min_inliers:
        return None, f"Not enough inliers for alignment ({inlier_count} < {min_inliers})."
    return matrix, "ok"
