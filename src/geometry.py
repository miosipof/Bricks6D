import cv2
import numpy as np
import logging
import matplotlib.pyplot as plt
from PIL import Image

from resources.constants import MIN_AREA_PX, ASPECT_MIN, ASPECT_MAX, VERTEX_MIN, VERTEX_MAX, BRICK_SIZE

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)



class Model2D():
    def __init__(self):
        pass

    def area_norm(self, mask_area, w, h):
        return mask_area/(w*h)

    def aspect_penalty(self, x):
        if ASPECT_MIN < x < ASPECT_MAX:
            return 1
        else:
            return 0

    def vertex_penalty(self, n):
        if VERTEX_MIN <= n <= VERTEX_MAX:
            return 1
        else:
            return 0    
        

    def create_poly(self, bin_mask, image_pil, visualize=False):

        contours, _ = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        log.info(f"{len(contours)} contours found")

        candidates = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < MIN_AREA_PX:          # weed out speckles
                continue

            hull = cv2.convexHull(cnt)
            solidity = area / cv2.contourArea(hull)

            # polygonal approximation
            eps = 0.01 * cv2.arcLength(cnt, True)
            poly = cv2.approxPolyDP(hull, eps, True)
            n = len(poly)

            # min‑area rectangle
            rect = cv2.minAreaRect(cnt)
            w,h = rect[1]
            if min(w,h) == 0:               # degenerate
                continue
            aspect = max(w,h) / min(w,h)

            score = (
                0.2 * self.area_norm(area,image_pil.size[0],image_pil.size[1]) +     # size dominates
                0.0 * solidity +            # prefer compact
                0.0 * self.aspect_penalty(aspect) +
                0.0 * self.vertex_penalty(n)     # gentle push toward 4–6 verts
            )
            candidates.append((score, cnt, poly))


        best = max(candidates, key=lambda x: x[0])
        best_cnt, best_poly = best[1], best[2]   

        if visualize==True:
            self.vis_poly(bin_mask, best_poly)


        return best_cnt, best_poly     
    

    def create_hexagon(self, poly, prefer_six=True, angle_tol=np.deg2rad(15)):
        """
        Sanitize / simplify a polygon to 4 vertices.

        Parameters
        ----------
        poly : (N,1,2) ndarray  -  output of cv2.approxPolyDP (N >= 4)
        prefer_six : bool       -  if True, keep 6 vertices when possible,
                                otherwise fall back to 4.
        angle_tol : float       -  angle tolerance (radians) to consider a vertex
                                nearly collinear and thus removable.

        Returns
        -------
        clean_poly : (V,2) ndarray with V in {4,6}
                    or None if the polygon cannot be simplified.
        """
        # -------------- helpers -------------------------------------------------
        def to_flat(p):
            return p.reshape(-1, 2).astype(np.float64)

        def internal_angles(pts):
            """Return array of interior angles at each vertex (radians)."""
            n = len(pts)
            v_prev = pts - np.roll(pts,  1, axis=0)
            v_next = np.roll(pts, -1, axis=0) - pts
            # normalise
            v_prev /= np.linalg.norm(v_prev, axis=1, keepdims=True)
            v_next /= np.linalg.norm(v_next, axis=1, keepdims=True)
            # angle = arccos(u·v)
            cosang = np.einsum('ij,ij->i', v_prev, v_next)
            cosang = np.clip(cosang, -1.0, 1.0)
            return np.arccos(cosang)

        # -------------- main ----------------------------------------------------
        pts = to_flat(poly)
        N   = len(pts)

        if N < 4:
            return None   # reject (too few vertices)

        if N == 4:
            return pts    # already the target size

        # iterative simplification
        target = 6 if (prefer_six and N >= 6) else 4
        while len(pts) > target:
            angles = internal_angles(pts)
            # "collinear score": how close to straight (π)
            score  = np.abs(np.pi - angles)
            print(f"{len(pts)} points, angles:{angles}, score={score}")
            # remove the vertex with smallest score (closest to π)
            idx_remove = np.argmax(score)
            # also ensure angle difference within tolerance, otherwise break
            if score[idx_remove] > angle_tol and len(pts) - 1 >= target:
                # no almost‑straight angle left but still too many vertices
                # fall back: remove the vertex with minimal score anyway
                pass
            pts = np.delete(pts, idx_remove, axis=0)

        if len(pts) != 4:
            # last resort simplification to 4
            while len(pts) > 4:
                angles = internal_angles(pts)
                idx_remove = np.argmax(np.abs(np.pi - angles))
                pts = np.delete(pts, idx_remove, axis=0)
            if len(pts) != 4:
                return None

        return pts.astype(np.float32)




    
    def split_brick_face(self, hexagon: np.ndarray):
        """
        hexagon : (6,2) array of xy vertices, ordered around the perimeter.
        Returns:
            quad_A, quad_B  # each is (4,2) array of xy vertices
        """

        # ------------------------------------------------------------------
        # 1. Compute edge vectors and their directions (unit normals)
        # ------------------------------------------------------------------
        edges   = np.roll(hexagon, -1, axis=0) - hexagon           # v_i = x_{i+1} - x_i
        norms   = edges / np.linalg.norm(edges, axis=1, keepdims=True)

        # ------------------------------------------------------------------
        # 2. Group edges into three parallel pairs
        #    We do this by clustering their angles (k‑means‑1D with k=3).
        # ------------------------------------------------------------------
        angles        = np.arctan2(norms[:, 1], norms[:, 0])        # angle of each edge
        angles        = (angles + 2*np.pi) % np.pi                  # fold 180° symmetry
        sorted_idx    = np.argsort(angles)
        groups        = [sorted_idx[[0, 3]], sorted_idx[[1, 4]], sorted_idx[[2, 5]]]

        # Groups come out unsorted — ensure they match your notation
        # We’ll re‑label so that (x1x2,x4x5) is group0 etc.
        # Because vertices are ordered, the opposite edge to index i is (i+3)%6
        g0 = np.array([0, 3])
        g1 = np.array([1, 4])
        g2 = np.array([5, 2])        # (x6x1, x3x4)

        # ------------------------------------------------------------------
        # 3. Compute the two “missing” vertices y1 and y2
        # ------------------------------------------------------------------
        x  = hexagon
        y1 = x[5] + (x[1] - x[0])    # x6 + vec(x1→x2)
        y2 = x[2] + (x[4] - x[3])    # x3 + vec(x4→x5)

        # ------------------------------------------------------------------
        # 4. Assemble the two quads
        # ------------------------------------------------------------------
        quad_front = np.vstack([x[1], x[2], x[3], y1])   # x2‑x3‑x4‑y1
        quad_top   = np.vstack([x[0], y2, x[4], x[5]])   # x1‑y2‑x5‑x6

        return quad_front.astype(np.float32), quad_top.astype(np.float32)


    def draw_quads(self, img, quadA, quadB):
        # Check shapes
        if quadA.shape != (4, 2) or quadB.shape != (4, 2):
            raise ValueError("Quads must have shape (4, 2)")

        # Sanitize input for OpenCV
        quadA = np.round(quadA).astype(int).reshape(1, 4, 2)
        quadB = np.round(quadB).astype(int).reshape(1, 4, 2)

        # Draw contours
        cv2.polylines(img, [quadA], isClosed=True, color=(0, 255, 0), thickness=2)
        cv2.polylines(img, [quadB], isClosed=True, color=(255, 0, 0), thickness=2)

    def vis_poly(self,bin_mask, poly):
    
        canvas = cv2.cvtColor(bin_mask, cv2.COLOR_GRAY2BGR)

        # Draw polygon outline
        cv2.polylines(canvas, [poly], isClosed=True, color=(0, 255, 0), thickness=2)

        # Draw vertices
        for (x, y) in poly.reshape(-1, 2):
            cv2.circle(canvas, (x, y), 4, (0, 0, 255), -1)

        # # Show the result
        # plt.figure(figsize=(6, 6))
        # plt.title(f"Polygon with {len(poly)} vertices")
        # plt.imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
        # plt.axis("off")

        cv2.imwrite('poly.jpg', canvas)


class Model3D:
    def __init__(self):
        self.brick_dims = BRICK_SIZE

    def get_brick_center(
        self,
        image_vertices: np.ndarray,
        depth_map: np.ndarray,
        intrinsics: dict
    ):
        """
        Determines the most front-facing face from a 6-vertex polygon and computes the camera-to-brick geometry.

        Returns
        -------
        image_pts      : (6,2) float32   pixel coordinates (CCW)
        object_pts     : (6,3) float32   inferred 3D coordinates
        orientation_ok : bool           winding consistency
        ray_centroid   : (3,) float64   unit ray from camera to centroid
        depth_c        : float          depth at centroid (may be NaN)
        face_type      : str            e.g. "a-b", "a-c", or "b-c"
        """

        def poly_area_2d(pts):
            pts = np.vstack([pts, pts[0]])
            return 0.5 * np.sum(pts[:-1, 0] * pts[1:, 1] - pts[1:, 0] * pts[:-1, 1])

        def face_normal_3d(pts):
            v1 = pts[1] - pts[0]
            v2 = pts[2] - pts[0]
            return np.cross(v1, v2)

        a, b, c = sorted(self.brick_dims, reverse=True)
        image_pts = image_vertices.astype(np.float32)


        l1 = np.linalg.norm(image_pts[1] - image_pts[0])
        l2 = np.linalg.norm(image_pts[2] - image_pts[1])
        edge_ratio = max(l1, l2) / min(l1, l2)
        

        if edge_ratio < 1:
            print(f"Edge length ratio: {edge_ratio} < 1. Permuting points...")
            new_pts = [image_pts[i] for i in range(1, len(image_pts))]
            new_pts.append(image_pts[0])
            image_pts = np.array(new_pts)


        object_pts = np.array([
            [0, 0, 0],
            [a, 0, 0],
            [0, b, 0],
            [a, b, 0]
        ], dtype=np.float32)

        # Ray and depth
        fx, fy, cx, cy = (intrinsics[k] for k in ("fx", "fy", "cx", "cy"))
        uc, vc = image_pts.mean(axis=0)
        ray = np.array([(uc - cx) / fx, (vc - cy) / fy, 1.0])
        ray /= np.linalg.norm(ray)

        H, W = depth_map.shape
        u_int, v_int = int(round(uc)), int(round(vc))
        depth_c = np.nan
        if 0 <= v_int < H and 0 <= u_int < W:
            depth_c = float(depth_map[v_int, u_int])

        return image_pts, object_pts, ray, depth_c



    def solve_brick_pose(self,
                         image_pts: np.ndarray,
                        object_pts: np.ndarray,
                        intrinsics: dict,
                        depth_centroid: float,
                        ray_to_centroid: np.ndarray,
                        depth_tol: float = 0.02):
        """
        Solve 6‑DoF pose of the brick given 2‑D/3‑D correspondences and a depth prior.

        Parameters
        ----------
        image_pts       : (N,2) float32   – pixel coordinates (CCW order)
        object_pts      : (N,3) float32   – matching brick coordinates in metres
        intrinsics      : dict            – {"fx":..,"fy":..,"cx":..,"cy":..}
        depth_centroid  : float           – depth (m) at polygon centroid
        ray_to_centroid : (3,) float64    – unit vector camera → centroid
        depth_tol       : float           – acceptable |t|-depth error fraction

        Returns
        -------
        R   : (3,3)  rotation matrix (camera→brick)
        t   : (3,1)  translation vector (metres)
        n   : (3,)   outward normal (unit, pointing toward camera)
        """
        fx, fy, cx, cy = (intrinsics[k] for k in ("fx","fy","cx","cy"))

        K = np.array([[fx, 0, cx],
                    [0, fy, cy],
                    [0,  0,  1]], dtype=np.float64)

        # --- initial PnP -------------------------------------------------------
        if len(image_pts) == 4:
            flag = cv2.SOLVEPNP_IPPE_SQUARE
        else:
            flag = cv2.SOLVEPNP_ITERATIVE

        ok, rvec, tvec = cv2.solvePnP(
            object_pts, image_pts, K, None,
            flags=flag
        )
        # ok, rvec, tvec, inliers = cv2.solvePnPRansac(
        #     object_pts, image_pts, K, None,
        #     flags=flag,
        #     reprojectionError=4.0,
        #     confidence=0.99,
        #     iterationsCount=100
        # )

        if not ok:
            raise RuntimeError("Initial solvePnP failed")

        # # --- enforce depth -----------------------------------------------------
        # t_norm = np.linalg.norm(tvec)
        # if not np.isnan(depth_centroid) and abs(t_norm - depth_centroid) > depth_tol * depth_centroid:
        #     scale = depth_centroid / t_norm
        #     tvec *= scale  # scale translation to match measured depth



        # (Optional) refine rotation with translation fixed
        ok2, rvec, _ = cv2.solvePnP(
            object_pts, image_pts, K, None,
            useExtrinsicGuess=True,
            rvec=rvec, tvec=tvec,
            flags=flag
        )

        R, _ = cv2.Rodrigues(rvec)

        # --- face normal (3rd column) -----------------------------------------
        n = R[:, 2]
        # ensure normal points toward camera (n·t < 0)
        if np.dot(n.flatten(), tvec.flatten()) > 0:
            n = -n

        return R, tvec, n


    def project_brick(self, R, tvec, normal, img_pts, obj_pts, intrinsics, image_pil):

        # Reproject the 3D object points to 2D image points

        rvec, _ = cv2.Rodrigues(R)  # Convert rotation matrix to rotation vector

        # Project the 3D points using the camera intrinsics and pose
        projected_pts, _ = cv2.projectPoints(
            objectPoints=obj_pts,
            rvec=rvec,
            tvec=tvec,
            cameraMatrix=np.array([
                [intrinsics["fx"], 0, intrinsics["cx"]],
                [0, intrinsics["fy"], intrinsics["cy"]],
                [0, 0, 1]
            ]),
            distCoeffs=None
        )
        # Flatten projected points for drawing
        projected_pts = projected_pts.squeeze().astype(np.int32)

        # Create canvas for visualization
        canvas = np.array(image_pil)

        # Draw reprojected polygon
        cv2.polylines(canvas, [projected_pts.reshape(-1, 1, 2)], isClosed=True, color=(0, 0, 255), thickness=2)

        # Draw original image polygon
        cv2.polylines(canvas, [img_pts.astype(int).reshape(-1, 1, 2)], isClosed=True, color=(0, 255, 0), thickness=2)

        # Display
        import matplotlib.pyplot as plt

        plt.figure(figsize=(6, 6))
        plt.imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
        plt.title("Projected 3D Brick Pose (Red) vs. Original Image Polygon (Green)")
        plt.axis("off")
        plt.show()




    def draw_vertex_labels(self, image_pil, image_pts, object_pts):
        """
        Draws numbered circles on the image at image_pts locations and displays image_pts ↔ object_pts mapping.

        Parameters
        ----------
        image       : (H,W,3) BGR image
        image_pts   : (N,2) float32 pixel coordinates
        object_pts  : (N,3) float32 3D brick coordinates
        """
        canvas = np.array(image_pil)

        for i, pt in enumerate(image_pts):
            u, v = int(pt[0]), int(pt[1])
            cv2.circle(canvas, (u, v), radius=5, color=(0, 255, 0), thickness=-1)
            cv2.putText(canvas, str(i), (u + 8, v - 8), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 255), 2, cv2.LINE_AA)

        # Print 2D–3D mapping clearly
        print(" Index |  image_pts (u, v)   |  object_pts (x, y, z)")
        print("-------+---------------------+------------------------")
        for i, (img_pt, obj_pt) in enumerate(zip(image_pts, object_pts)):
            u, v = img_pt
            x, y, z = obj_pt
            print(f"  {i:>3}  |  ({u:7.1f}, {v:7.1f})  |  ({x:6.3f}, {y:6.3f}, {z:6.3f})")

        # Show the image
        plt.figure(figsize=(6, 6))
        plt.imshow(cv2.cvtColor(canvas.astype(np.uint8), cv2.COLOR_BGR2RGB))
        plt.title("2D Vertex Indices and 3D Mapping")
        plt.axis("off")
        plt.show()


    def draw_normal_vector(self, image_pil, tvec, normal, intrinsics, length=0.05, color=(0, 0, 255), thickness=2):
        """
        Projects and draws the 3D surface normal onto a 2D image.

        Parameters
        ----------
        image       : np.ndarray   – Input image (BGR or grayscale)
        tvec        : (3,1) float  – Translation vector from solvePnP (brick center in camera frame)
        normal      : (3,) float   – Unit normal vector in camera coordinates
        intrinsics  : dict         – {"fx", "fy", "cx", "cy"}
        length      : float        – Arrow length in meters (default 5 cm)
        color       : (B,G,R)      – Arrow color (default red)
        thickness   : int          – Arrow thickness (default 2)
        """

        # Convert intrinsics to matrix
        fx, fy, cx, cy = intrinsics["fx"], intrinsics["fy"], intrinsics["cx"], intrinsics["cy"]
        K = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0,  0,  1]
        ], dtype=np.float64)

        # Project 3D center and tip of normal
        tip3D = (tvec.reshape(3) + length * normal).reshape(3, 1)
        tvec = tvec.reshape(3, 1)

        pt1, _ = cv2.projectPoints(tvec, np.zeros(3), np.zeros(3), K, None)
        pt2, _ = cv2.projectPoints(tip3D, np.zeros(3), np.zeros(3), K, None)

        p1 = tuple(pt1[0, 0].astype(int))
        p2 = tuple(pt2[0, 0].astype(int))

        # Prepare canvas
        canvas = np.array(image_pil)
        if len(canvas.shape) == 2 or canvas.shape[2] == 1:
            canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)

        # Draw arrow
        cv2.arrowedLine(canvas, p1, p2, color=color, thickness=thickness, tipLength=0.1)

        # Display
        plt.figure(figsize=(8, 6))
        plt.imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
        plt.title("Projected Normal Vector")
        plt.axis("off")
        plt.show()




"""
Helper functions 2D
"""

def _to_xy(poly):
    """(N,1,2)|(N,2) → (N,2) float64"""
    poly = np.squeeze(poly)
    return poly.astype(np.float64)

def _edge_vectors(pts):
    v = np.roll(pts, -1, axis=0) - pts
    l = np.linalg.norm(v, axis=1)
    return v, l

def _parallel_angle(u, v):
    """Return |π − θ| where θ is the angle between u and v."""
    dot = np.clip(np.dot(u, v) / (np.linalg.norm(u)*np.linalg.norm(v)), -1, 1)
    return np.abs(np.pi - np.arccos(dot))

def _homog(p):                  # (2,) → (3,)
    return np.array([p[0], p[1], 1.0])

def _line(p, q):
    return np.cross(_homog(p), _homog(q))   # p×q

def _intersection(l1, l2, tol=1e-7):
    p = np.cross(l1, l2)
    if abs(p[2]) < tol:                     # lines nearly parallel
        return None
    return p[:2] / p[2]


def _clip_to_border(p, q, w, h):
    """Return the two image‑border intersections of the infinite line p–q."""
    l = np.cross([p[0], p[1], 1], [q[0], q[1], 1])   # line coeffs
    out = []
    for U, V, is_u in [(0, None, True), (w-1, None, True),
                       (None, 0, False), (None, h-1, False)]:
        if is_u:
            if abs(l[1]) < 1e-9: continue
            v = -(l[0]*U + l[2]) / l[1]
            if 0 <= v < h: out.append((U, v))
        else:
            if abs(l[0]) < 1e-9: continue
            u = -(l[1]*V + l[2]) / l[0]
            if 0 <= u < w: out.append((u, V))
        if len(out) == 2: break
    return np.array(out[:2], int)

def _edge_vectors(pts):
    v = np.roll(pts, -1, axis=0) - pts
    l = np.linalg.norm(v, axis=1)
    return v, l

def _parallel_angle(u, v):
    dot = np.clip(np.dot(u, v) /
                  (np.linalg.norm(u)*np.linalg.norm(v)), -1, 1)
    return np.abs(np.pi - np.arccos(dot))

def _h(p): return np.array([p[0], p[1], 1.0])
def _line(p, q): return np.cross(_h(p), _h(q))
def _intersection(l1, l2, eps=1e-7):
    p = np.cross(l1, l2)
    return None if abs(p[2]) < eps else p[:2]/p[2]



"""
Helper functions 3D
"""

def poly_area_2d(pts):
    xy = np.vstack([pts, pts[0]])
    return 0.5*np.sum(xy[:-1,0]*xy[1:,1] - xy[1:,0]*xy[:-1,1])

def face_normal_3d(pts):
    v1 = pts[1]-pts[0]
    v2 = pts[2]-pts[0]
    return np.cross(v1,v2)