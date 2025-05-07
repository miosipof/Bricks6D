import cv2
import numpy as np
import logging
import matplotlib.pyplot as plt
from PIL import Image

from resources.constants import MIN_AREA_PX, ASPECT_MIN, ASPECT_MAX, VERTEX_MIN, VERTEX_MAX

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






"""
Helper functions
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