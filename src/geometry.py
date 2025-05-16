import cv2
import numpy as np
import logging
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import json
from resources.constants import MIN_AREA_PX, ASPECT_MIN, ASPECT_MAX, VERTEX_MIN, VERTEX_MAX, BRICK_SIZE
import open3d as o3d

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)


###______________________________________________________________________

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


    def shrink_polygon(self, polygon, shrink_factor=0.5):
        """
        Shrinks a 4-point polygon (np.array shape (4,2)) toward its centroid.
        """
        centroid = np.mean(polygon, axis=0)
        return centroid + shrink_factor * (polygon - centroid)

    def polygon_to_mask(self, image_shape, polygon):
        mask = np.zeros(image_shape, dtype=np.uint8)
        cv2.fillPoly(mask, [polygon.astype(np.int32)], 1)
        return mask

###______________________________________________________________________

class Model3D:
    def __init__(self):
        self.brick_dims = BRICK_SIZE

    def load_k(self, json_path):
        # Load camera parameters
        with open(json_path, 'r') as f:
            intrinsics = json.load(f)

        fx, fy, cx, cy = (intrinsics[k] for k in ("fx","fy","cx","cy"))
        K = np.array([[fx, 0, cx],
                    [0, fy, cy],
                    [0,  0,  1]], dtype=np.float64)        
        return K


    def project_face(self, depth_png16, mask, K, scale=0.001):
        """
        depth_png16 : uint16 ndarray, metric depth after multiplying by `scale`
        mask        : uint8 ndarray, 1 = face pixel, 0 = background
        K           : (3,3) camera matrix  [[fx, 0, cx],
                                            [0, fy, cy],
                                            [0,  0,  1]]
        scale       : real-world metres per DN in the PNG  (0.001 for mm→m)
        returns     : (N,3) array of 3-D points in camera frame
        """
        # 1-a extract pixel indices inside the mask
        ys, xs = np.nonzero(mask)
        if xs.size == 0:
            raise ValueError("mask empty")

        # 1-b vectorised back-projection
        z = depth_png16[ys, xs].astype(np.float32) * scale
        # optional filter: throw away z==0 or >sensor range
        keep = z > 0
        xs, ys, z = xs[keep], ys[keep], z[keep]

        # intrinsic parameters
        fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]

        # X = (u − cx) * Z / fx ;  Y = (v − cy) * Z / fy
        X = (xs - cx) * z / fx
        Y = (ys - cy) * z / fy
        pts = np.column_stack((X, Y, z))
        return pts


    def fit_plane(self, points, ransac_thresh=0.004, min_inliers=300):
        """
        points : (N,3) ndarray
        returns (centroid, normal, inlier_mask)
        """
        # ----- optional RANSAC to kill depth outliers -----
        best_inliers = None
        rng = np.random.default_rng()
        for _ in range(50):
            sample = points[rng.choice(points.shape[0], 3, replace=False)]
            n = np.cross(sample[1]-sample[0], sample[2]-sample[0])
            n /= np.linalg.norm(n) + 1e-8
            d = -sample[0].dot(n)
            dist = np.abs(points.dot(n) + d)
            inliers = dist < ransac_thresh
            if best_inliers is None or inliers.sum() > best_inliers.sum():
                best_inliers = inliers
        pts_in = points[best_inliers] if best_inliers.sum() >= min_inliers else points

        # ----- PCA/SVD on inliers -----
        centroid = pts_in.mean(axis=0)
        Q = pts_in - centroid
        U, S, vh = np.linalg.svd(Q, full_matrices=False)
        print("SVD singular values:", S)
        print("Flatness score (S[1]/S[2]):", S[1] / S[2])
        normal = vh[-1]                      # eigenvector with smallest eigenvalue
        # make it point *towards* the camera (positive z)
        if normal[2] > 0:
            normal = -normal
        normal /= np.linalg.norm(normal)     # ensure unit length
        return centroid, normal, best_inliers


    def fit_axis(self, mask):
        ys, xs = np.nonzero(mask)
        Q = np.column_stack((xs - xs.mean(), ys - ys.mean()))
        _, _, vh = np.linalg.svd(Q, full_matrices=False)
        axis_2d = vh[0]               # in-plane direction of longest extent
        # yaw = arctan2(axis_2d.y, axis_2d.x)   # if you need it
        return axis_2d


    def vis_plane(self, pts_in, centroid, normal, mask=None, K=None, patch_size=0.1):
        """
        Visualizes:
        - the 3D inlier points
        - a coordinate frame at the centroid
        - a square patch aligned with the fitted plane normal

        Args:
            pts_in: (N,3) inlier 3D points
            centroid: (3,) center of fitted plane
            normal: (3,) unit normal vector of plane
            mask: optional 2D mask to compute in-plane axes
            K: optional camera intrinsics
            patch_size: width/height of visual plane patch in meters
        """
        # 1. Point cloud
        cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts_in))

        # 2. Coordinate frame at centroid
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
        frame.translate(centroid)

        # 3. Construct local frame {vx, vy, normal}
        z = normal / np.linalg.norm(normal)
        if mask is not None and K is not None:
            # Use 2D PCA to get brick orientation
            ys, xs = np.nonzero(mask)
            pts_2d = np.stack((xs - xs.mean(), ys - ys.mean()), axis=1)
            _, _, vh = np.linalg.svd(pts_2d, full_matrices=False)
            dir2d = vh[0]
            fx, fy = K[0,0], K[1,1]
            vx = np.array([dir2d[0] / fx, dir2d[1] / fy, 0.0])
            vx -= vx.dot(z) * z
            vx /= np.linalg.norm(vx)
        else:
            # Arbitrary orthogonal tangent
            ref = np.array([1, 0, 0]) if abs(z[0]) < 0.9 else np.array([0, 1, 0])
            vx = ref - ref.dot(z) * z
            vx = vx / np.linalg.norm(vx)

        vy = np.cross(z, vx)
        R = np.column_stack((vx, vy, z))  # 3×3 rotation matrix

        # 4. Create box patch and rotate/translate
        box = o3d.geometry.TriangleMesh.create_box(width=patch_size, height=patch_size, depth=0.001)

        box.translate(-box.get_center())  # center at origin
        box.rotate(R, center=(0, 0, 0))
        box.translate(centroid)
        box.paint_uniform_color([0.8, 0.3, 0.3])  # reddish patch

        # 5. Visualize all
        o3d.visualization.draw_geometries([cloud, frame, box])




    def project_to_img(self, points_3d, K):
        """
        Project 3D points onto the image plane.
        points: (N, 3) ndarray in camera frame
        K: (3, 3) camera intrinsics
        Returns: (N, 2) ndarray of pixel coordinates
        """
        fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
        X, Y, Z = points_3d[:,0], points_3d[:,1], points_3d[:,2]
        u = (X * fx / Z + cx).astype(np.int32)
        v = (Y * fy / Z + cy).astype(np.int32)
        return np.stack((u, v), axis=-1)

    def draw_points(self, image_pil, points_2d, color=(255, 0, 0), radius=2):
        """
        Draw 2D points on a PIL image.
        """
        image_pil = image_pil.convert("RGB")
        draw = ImageDraw.Draw(image_pil)
        w, h = image_pil.size
        for (u, v) in points_2d:
            if 0 <= u < w and 0 <= v < h:
                draw.ellipse([(u - radius, v - radius), (u + radius, v + radius)], fill=color)

        return image_pil
    


    def vis_pose(self, image_pil, mask, centroid_3d, normal_3d, axis_2d, K,
                                scale=1.0, color_centroid=(255, 0, 0),
                                color_normal=(0, 0, 255), color_axis=(255, 255, 0)):
        """
        Overlays:
        - centroid point
        - normal vector projected into 2D
        - axis_2d direction (from mask shape)

        Args:
            image_pil: PIL image
            mask: 2D binary mask
            centroid_3d: np.array (3,)
            normal_3d: np.array (3,)
            axis_2d: np.array (2,)
            K: camera intrinsics matrix
            scale: how long the arrows should be in pixels
        """
        image = image_pil.copy().convert("RGB")
        draw = ImageDraw.Draw(image)
        fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]

        # --- Project centroid to image ---
        X, Y, Z = centroid_3d
        u = int(X * fx / Z + cx)
        v = int(Y * fy / Z + cy)

        # --- Draw centroid ---
        r = 4
        draw.ellipse([(u - r, v - r), (u + r, v + r)], fill=color_centroid)

        # --- Project normal vector ---
        normal_2d = np.array([
            normal_3d[0] * fx / Z,
            normal_3d[1] * fy / Z
        ])
        end_norm = (int(u + scale * normal_2d[0]), int(v + scale * normal_2d[1]))
        draw.line([(u, v), end_norm], fill=color_normal, width=2)

        # --- Draw in-plane axis_2d ---
        axis_2d_norm = axis_2d / np.linalg.norm(axis_2d)
        end_axis = (int(u + scale * axis_2d_norm[0]), int(v + scale * axis_2d_norm[1]))
        draw.line([(u, v), end_axis], fill=color_axis, width=2)


        image.save('pose_vis.jpg')
        image.show()        

        return image


    def overlay_points(self, image_pil, pts_3d, centroid, normal, K, patch_size=0.1):
        """
        Overlays 3D inlier points and the fitted plane patch onto the image.

        Args:
            image_pil: PIL.Image (RGB)
            pts_3d: (N,3) inlier 3D points
            centroid: (3,) np.array
            normal: (3,) np.array
            K: (3,3) camera intrinsics
            patch_size: plane size in meters
        Returns:
            PIL.Image with overlay
        """
        from PIL import ImageDraw
        image = image_pil.copy().convert("RGB")
        draw = ImageDraw.Draw(image)

        fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]

        def project(p3d):
            x, y, z = p3d
            u = int(x * fx / z + cx)
            v = int(y * fy / z + cy)
            return u, v

        # 1. Draw projected 3D points
        for p in pts_3d:
            u, v = project(p)
            draw.ellipse([(u-1, v-1), (u+1, v+1)], fill=(255, 0, 0))

        # 2. Draw plane patch (projected square)
        # Create plane-aligned frame
        z = normal / np.linalg.norm(normal)
        x0 = np.array([1, 0, 0]) if abs(z[0]) < 0.9 else np.array([0, 1, 0])
        vx = x0 - x0.dot(z) * z
        vx /= np.linalg.norm(vx)
        vy = np.cross(z, vx)

        # 4 corner points in local frame
        s = patch_size / 2
        corners_local = np.array([
            [-s, -s, 0],
            [ s, -s, 0],
            [ s,  s, 0],
            [-s,  s, 0],
        ])
        R = np.column_stack((vx, vy, z))
        corners_3d = (R @ corners_local.T).T + centroid
        corners_2d = [project(p) for p in corners_3d]

        # Draw polygon
        draw.polygon(corners_2d, outline=(0, 255, 0), width=2)

        return image






###______________________________________________________________________

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

###______________________________________________________________________

"""
Helper functions 3D
"""
