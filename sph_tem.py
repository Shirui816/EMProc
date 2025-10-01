import cv2
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, filters, measure, color

import numpy as np
import matplotlib.pyplot as plt
from sys import argv
from skimage import io, filters, measure, segmentation, color, morphology
import math

from scipy.spatial import cKDTree


__doc__ = """
Find weak circles and identify square/hex packing of nanoparticles.
"""

def compute_psi_n_with_jump_filter(coords, k, n,
                                   max_neighbors=None,
                                   jump_ratio=1.2,
                                   jump_abs=None):
    """Simple local order parameter func
    """

    N = coords.shape[0]
    if max_neighbors is None:
        max_neighbors = k * 2
    tree = cKDTree(coords)
    psi_n = np.zeros(N, dtype=complex)

    for i, p in enumerate(coords):
        dists, idxs = tree.query(p, k=max_neighbors + 1)
        # dists[0] == 0, idxs[0] == i
        neigh_idxs = idxs[1:]
        neigh_dists = dists[1:]

        filtered = []
        prev_d = neigh_dists[0]
        filtered.append(neigh_idxs[0])
        for dist, idx in zip(neigh_dists[1:], neigh_idxs[1:]):
            if len(filtered) >= k:
                break
            if dist / prev_d > jump_ratio:
                break
            if (jump_abs is not None) and ((dist - prev_d) > jump_abs):
                break
            filtered.append(idx)
            prev_d = dist

        m = len(filtered)
        if m == 0:
            psi_n[i] = 0
        else:
            vecs = coords[filtered] - p
            angles = np.arctan2(vecs[:, 1], vecs[:, 0])
            psi_n[i] = np.mean(np.exp(1j * n * angles))

    return psi_n


def analysis_obj(img_fn: str, minSize: int=5, showImg:bool=False):
    """Detect objects, determine the shape (TODO) and size
    """
    img = io.imread(img_fn, as_gray=True)
    th = filters.threshold_otsu(img)
    bw = img < th
    bw = morphology.remove_small_objects(bw, min_size=minSize)
    labels = measure.label(bw, connectivity=2)
    contours = measure.find_contours(bw, level=0.5)
    if showImg:
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(img, cmap='gray')
        for contour in contours:
            ax.plot(contour[:, 1], contour[:, 0], linewidth=0.1, color='red')
    
    shapeFac = []
    sizes = []
    for props in measure.regionprops(labels):
        if props.area < minSize:
            continue
        y0, x0 = props.centroid
        orientation = props.orientation
        x1 = x0 + math.cos(orientation) * 0.5 * props.axis_minor_length
        y1 = y0 - math.sin(orientation) * 0.5 * props.axis_minor_length
        x2 = x0 - math.sin(orientation) * 0.5 * props.axis_major_length
        y2 = y0 - math.cos(orientation) * 0.5 * props.axis_major_length
        shapeFac.append(props.axis_minor_length / props.axis_major_length)
        sizes.append((props.axis_minor_length + props.axis_major_length) / 2)
        if showImg:
            #ax.plot((x0, x1), (y0, y1), '-r', linewidth=2.5)
            #ax.plot((x0, x2), (y0, y2), '-b', linewidth=2.5)
            #ax.plot(x0, y0, '.g', markersize=15)
            minr, minc, maxr, maxc = props.bbox
            bx = (minc, maxc, maxc, minc, minc)
            by = (minr, minr, maxr, maxr, minr)
            #ax.plot(bx, by, '-b', linewidth=2.5)
    if showImg:
        ax.set_axis_off()
        plt.tight_layout()
    if showImg:
        plt.show()
    return shapeFac, sizes

def detect_circles_in_grayscale(image_path, output_path='out.jpg'):
    """Detect Hough Circles
    """

    _, sizes = analysis_obj(image_path, minSize=150)
    s = (np.mean(sizes) + np.median(sizes)) / 2
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)


    img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    blurred = cv2.GaussianBlur(img, (7, 7), 2)
    enhanced = cv2.equalizeHist(blurred)

    ret, otsuT = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresh = 0.333 * ret  # Magical



    circles1 = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1,             
        minDist=int(s * 0.5),        
        param1=thresh,
        param2=min(int(s * 0.5), 25),   # also magical
        minRadius=int(s * 0.7 /2.),
        maxRadius=int(s * 1.5/2.)
    )
    
    pos = []
    img_result1 = img_color.copy()
    #img_result1 = enhanced
    circles1 = np.round(circles1[0, :]).astype("int")
    
    for (x, y, r) in circles1:
        #cv2.circle(img_result1, (x, y), r, (0, 255, 0), 2)
        #cv2.circle(img_result1, (x, y), 2, (0, 0, 255), 3)
        pos.append((x, y))

        
    pos = np.asarray(pos)
    psi4 = compute_psi_n_with_jump_filter(pos, k=4, n=4,
                                          max_neighbors=8)
    psi6 = compute_psi_n_with_jump_filter(pos, k=6, n=6,
                                          max_neighbors=12)
    
    m4 = np.abs(psi4)
    m6 = np.abs(psi6)
    local_struct = np.where(m6 > m4, "hexagonal", "square")
    
    for (x, y, r), lo in zip(circles1, local_struct):
        color = (0, 0, 255)
        if lo == 'hexagonal':
            color = (255, 0, 0)
        
        cv2.circle(img_result1, (x, y), r, color, 2)
        cv2.circle(img_result1, (x, y), 2, (0, 0, 255), 3)

    
    
    plt.subplot(1, 1, 1)
    ax = plt.imshow(cv2.cvtColor(img_result1, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title("Hex: blue, Square: Red", fontsize=19)
    

    plt.tight_layout()
    plt.show()

    if output_path:
        cv2.imwrite(output_path, img_result1)
    



if __name__ == "__main__":

    image_path = "7.jpg" 
    output_path = "detected_circles_result.jpg"

    circles1 = detect_circles_in_grayscale(image_path, output_path)
