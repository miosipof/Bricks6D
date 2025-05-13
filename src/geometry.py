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
        Sanitize / simplify a polygon to 4 or 6 vertices.

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

        if N == 4 or N == 6:
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

        # if we undershot (possible when N=5 and target was 6) → force 4
        if len(pts) not in (4, 6):
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

        # Show the result
        plt.figure(figsize=(6, 6))
        plt.title(f"Polygon with {len(poly)} vertices")
        plt.imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
        plt.axis("off")


class VertexSolver:
    def __init__(self):
        pass

    def find_missing_vertex(self, poly6, eps=np.deg2rad(12)):
        """
        Parameters
        ----------
        poly6 : (6,2) ndarray, clockwise silhouette vertices

        Returns
        -------
        v      : (2,)  reconstructed vertex
        debug  : dict  for drawing
        """
        P   = poly6.astype(float)
        vec, length = _edge_vectors(P)

        # ---- 1. choose the best opposite edge pair (longest & quasi‑parallel) --
        pairs = [(0,3), (1,4), (2,5)]
        i,j   = max(pairs,
                    key=lambda ij: (length[ij[0]]+length[ij[1]])
                    if _parallel_angle(vec[ij[0]], vec[ij[1]]) < eps else -1)

        # anchor‑1 : vertex *before* edge i
        a1_idx = (i-1) % 6
        anchor1 = P[a1_idx]

        # vanishing point of edges i and j
        vp1 = _intersection(_line(P[i], P[(i+1)%6]),
                            _line(P[j], P[(j+1)%6]))
        if vp1 is None:
            return None, {"fail": "vp1 at infinity"}

        ray1 = _line(anchor1, vp1)

        # ---- 2. anchor‑2 = anchor‑1 ± 2  (choose +2, clockwise) --------------
        a2_idx = (a1_idx + 2) % 6
        anchor2 = P[a2_idx]

        # edges around anchor‑2 +- one edge → average their directions
        v_left_idx = ((a2_idx-2)%6,(a2_idx-1)%6)
        v_right_idx = ((a2_idx-2)%6,(a2_idx-1)%6)

        print(f"Left: {v_left_idx} anchor-2: {a2_idx}, right: {v_right_idx}")

        v_left  = P[v_left_idx[1]] - P[v_left_idx[0]]      # incoming
        v_right = P[v_right_idx[1]] - P[v_right_idx[0]]      # outgoing
        dir2    = v_left/np.linalg.norm(v_left) + v_right/np.linalg.norm(v_right)
        dir2    /= np.linalg.norm(dir2)

        # ray‑2 in homogeneous form: anchor2 + t*dir2
        far_pt  = anchor2 + 10000*dir2             # a distant point in that dir
        ray2    = _line(anchor2, far_pt)

        # ---- 3. intersection of ray1 & ray2 = missing vertex ------------------
        missing = _intersection(ray1, ray2)
        if missing is None:
            return None, {"fail": "rays parallel"}

        debug = {
            "anchor1_idx": a1_idx, "anchor1": anchor1,
            "edge_i": i, "edge_j": j, "vp1": vp1,
            "anchor2_idx": a2_idx, "anchor2": anchor2,
            "dir2": dir2,
            "missing": missing,
            "ray1": ray1, "ray2": ray2
        }
        return missing.astype(np.float32), debug


    def draw_vertex(self, mask, hexagon, debug):
        """
        Visualise rule‑2 reconstruction.

        Colours
        -------
        green   : original hexagon
        blue    : first edge pair + anchor‑1
        orange  : edges at anchor‑2 + anchor‑2
        red     : vanishing rays, missing vertex, helper lines
        """
        H, W = mask.shape
        canvas = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        # 1) original hexagon --------------------------------------------------
        cv2.polylines(canvas,
                    [hexagon.reshape(-1,1,2).astype(int)],
                    True, (0,255,0), 2)

        # -------- keys from debug --------------------------------------------
        a1   = debug["anchor1"].astype(int)
        a2   = debug["anchor2"].astype(int)
        i, j = debug["edge_i"], debug["edge_j"]           # first chosen pair
        missing = debug["missing"].astype(int)

        # 2) first chosen pair (BLUE) + anchor‑1 --------------------------------
        for idx in (i, j):
            p1, p2 = hexagon[idx], hexagon[(idx+1)%6]
            cv2.line(canvas, tuple(p1.astype(int)), tuple(p2.astype(int)), (255,0,0), 2)
        cv2.circle(canvas, tuple(a1), 6, (255,0,0), -1)

        # 3) second anchor edges (ORANGE) ---------------------------------------
        for idx in ((debug["anchor2_idx"]-1)%6, debug["anchor2_idx"]):
            p1, p2 = hexagon[idx], hexagon[(idx+1)%6]
            cv2.line(canvas, tuple(p1.astype(int)), tuple(p2.astype(int)), (0,165,255), 2)
        cv2.circle(canvas, tuple(a2), 6, (0,165,255), -1)

        # 4) vanishing ray‑1 (anchor‑1 → vp1)  ----------------------------------
        vp1 = debug["vp1"]
        if vp1 is not None:
            seg = _clip_to_border(a1, vp1, W, H)
            if len(seg)==2:
                cv2.line(canvas, tuple(seg[0]), tuple(seg[1]), (0,0,255), 1, cv2.LINE_AA)

        # 5) ray‑2 (anchor‑2 + dir2) -------------------------------------------
        dir2 = debug["dir2"]
        far  = (a2 + dir2*10000).astype(int)
        seg2 = _clip_to_border(a2, far, W, H)
        if len(seg2)==2:
            cv2.line(canvas, tuple(seg2[0]), tuple(seg2[1]), (0,0,255), 1, cv2.LINE_AA)

        # 6) missing vertex & helper lines -------------------------------------
        cv2.circle(canvas, tuple(missing), 6, (0,0,255), -1)
        cv2.line(canvas, tuple(a1), tuple(missing), (0,0,255), 2)
        cv2.line(canvas, tuple(a2), tuple(missing), (0,0,255), 2)

        # ----------------------------------------------------------------------
        plt.figure(figsize=(6,6))
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        plt.imshow(rgb.astype(np.uint8))
        plt.axis("off")
        plt.show()


class Model3D:
    def __init__(self):
        self.brick_dims = BRICK_SIZE

    def analyze_faces(self, image_pts: np.ndarray):
        """
        Analyze projected brick faces by edge angles and 2D area to identify best face candidate.

        Parameters
        ----------
        image_pts : (7,2) float32 – polygon vertices including center (index 6)

        Returns
        -------
        face_stats : list of dict – each with 'name', 'angle_deg', 'area', 'ratio'
        best_face  : dict         – face most orthogonal (angle ≈ 90°)
        """

        # Define edges and face pairs (by index into image_pts)
        # edge_pairs = [
        #     ((0,6), (1,2)),   # front face
        #     ((0,1), (6,2)),   # top face
        #     ((2,6), (3,4)),   # right face
        #     ((2,3), (6,4)),   # angled face
        #     ((4,6), (0,5)),   # left face
        #     ((4,5), (0,6))    # bottom face
        # ]
        edge_pairs = [
            ((0,6), (1,2)),   # front face
            # ((0,1), (6,2)),   # top face
            ((2,6), (3,4)),   # right face
            # ((2,3), (6,4)),   # angled face
            ((4,6), (0,5)),   # left face
            # ((4,5), (0,6))    # bottom face
        ]        

        def vector(p1, p2):
            return image_pts[p2] - image_pts[p1]

        def angle_between(u, v):
            u = u / np.linalg.norm(u)
            v = v / np.linalg.norm(v)
            dot = np.clip(np.dot(u, v), -1, 1)
            return np.arccos(dot)

        def face_area(p1, p2, p3, p4):
            quad = np.array([image_pts[p1], image_pts[p2], image_pts[p3], image_pts[p4]])
            # Split into two triangles for robust area
            a1 = 0.5 * np.abs(np.cross(quad[1] - quad[0], quad[2] - quad[0]))
            a2 = 0.5 * np.abs(np.cross(quad[2] - quad[0], quad[3] - quad[0]))
            return a1 + a2

        face_stats = []

        for ((i1, i2), (j1, j2)) in edge_pairs:
            v1 = vector(i1, i2)
            v2 = vector(i1, j1)
            angle = angle_between(v1, v2)
            angle_deg = np.degrees(angle)
            print(f"({(i1, i2)})-{(i1, j1)}: {angle_deg}")
            

            # Estimate quadrilateral formed by (i1, i2, j2, j1)
            area = face_area(i1, i2, j2, j1)
            ratio = np.linalg.norm(vector(i1, i2)) / np.linalg.norm(vector(j1, j2))

            face_stats.append({
                "angle_deg": angle_deg,
                "area": area,
                "ratio": ratio,
                "edge_pair": ((i1, i2), (j1, j2))
            })

        # Select face with angle closest to 90°
        best_face = min(face_stats, key=lambda f: abs(f["angle_deg"] - 90))

        return face_stats, best_face

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

        face_stats, best_face = self.analyze_faces(image_pts)

        print(f"Best face: {best_face}")
        print(f"Edge pair: {best_face['edge_pair']}")
        # Step 2: Estimate edge ratio for this face
        i1, i2 = best_face['edge_pair'][0]
        j1, j2 = best_face['edge_pair'][1]

        l1 = np.linalg.norm(image_pts[i1] - image_pts[i2])
        l2 = np.linalg.norm(image_pts[i1] - image_pts[j1])
        edge_ratio = max(l1, l2) / min(l1, l2)
        print(f"Best pair edge_ratio: {edge_ratio}")

        ratios_diff = {
            "a-b": abs(max(a, b) / min(a, b) - edge_ratio),
            "a-c": abs(max(a, c) / min(a, c) - edge_ratio),
            "b-c": abs(max(b, c) / min(b, c) - edge_ratio)
        }

        print(f"a/b/c ratios: {ratios_diff}")

        face_type = min(ratios_diff, key=ratios_diff.get)

        print(f"Identified face type: {face_type}")

        # Step 3: Assign 3D box corners based on face type
        face_dims = {
            "a-b": (a, b, c),
            "a-c": (c, a, b),
            "b-c": (b, c, a)
        }

        a_, b_, c_ = face_dims[face_type]

        object_pts = np.array([
            [0, 0, 0],
            [a_, 0, 0],
            [a_, b_, 0],
            [a_, b_, c_],
            [0, b_, c_],
            [0, 0, c_]
        ], dtype=np.float32)

        # Orientation check
        area2d = poly_area_2d(image_pts)
        normal3d_z = face_normal_3d(object_pts)[2]
        orientation_ok = (area2d > 0 and normal3d_z > 0) or (area2d < 0 and normal3d_z < 0)

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

        return image_pts, object_pts, orientation_ok, ray, depth_c, face_type



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

        # --- enforce depth -----------------------------------------------------
        t_norm = np.linalg.norm(tvec)
        if not np.isnan(depth_centroid) and abs(t_norm - depth_centroid) > depth_tol * depth_centroid:
            scale = depth_centroid / t_norm
            tvec *= scale  # scale translation to match measured depth



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