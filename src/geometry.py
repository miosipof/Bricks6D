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

