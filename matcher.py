import numpy as np
try:
    import cv2
except Exception:  # pragma: no cover - runtime dependency check
    cv2 = None

from .constants import MATCHER_TYPES, MIN_MATCHES


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
    if matcher_type == "surf":
        if not hasattr(cv2, "xfeatures2d") or not hasattr(cv2.xfeatures2d, "SURF_create"):
            return None, None
        return cv2.xfeatures2d.SURF_create(hessianThreshold=400), cv2.NORM_L2
    return None, None


def detect_and_match(
    background_np,
    overlay_np,
    bg_mask_np,
    ov_mask_np,
    feature_count,
    good_match_percent,
    matcher_type,
):
    _ensure_cv2()

    if matcher_type not in MATCHER_TYPES:
        return None, None, f"Unsupported matcher: {matcher_type}"

    if background_np.shape[2] != 3 or overlay_np.shape[2] not in (3, 4):
        return None, None, "Only 3-channel background and 3/4-channel overlay images are supported."

    bg_gray = cv2.cvtColor(background_np[:, :, :3], cv2.COLOR_RGB2GRAY)
    ov_gray = cv2.cvtColor(overlay_np[:, :, :3], cv2.COLOR_RGB2GRAY)

    detector, norm_type = _create_detector(matcher_type, feature_count)
    if detector is None:
        return None, None, f"Matcher {matcher_type} is not available in this OpenCV build."

    ov_keypoints, ov_desc = detector.detectAndCompute(ov_gray, ov_mask_np)
    bg_keypoints, bg_desc = detector.detectAndCompute(bg_gray, bg_mask_np)

    if ov_desc is None or bg_desc is None or len(ov_keypoints) < MIN_MATCHES or len(bg_keypoints) < MIN_MATCHES:
        return None, None, "Not enough keypoints for alignment."

    matcher = cv2.BFMatcher(norm_type, crossCheck=True)
    matches = matcher.match(ov_desc, bg_desc)
    if len(matches) < MIN_MATCHES:
        return None, None, "Not enough matches for alignment."

    matches = sorted(matches, key=lambda match: match.distance)
    keep = max(MIN_MATCHES, int(len(matches) * good_match_percent))
    matches = matches[:keep]

    ov_points = np.float32([ov_keypoints[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    bg_points = np.float32([bg_keypoints[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    return ov_points, bg_points, "ok"


def estimate_affine(ov_points, bg_points, ransac_thresh):
    _ensure_cv2()
    matrix, _inliers = cv2.estimateAffinePartial2D(
        ov_points,
        bg_points,
        method=cv2.RANSAC,
        ransacReprojThreshold=ransac_thresh,
    )
    if matrix is None:
        return None, "Could not estimate affine transform."
    return matrix, "ok"
