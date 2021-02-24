from scipy.spatial import ConvexHull
# All coords inside the convex hull
def point_left_to_a_line(X,Y,A,B):
    """
    returns 1 if on the left 
    """
    return np.sign((B[0] - A[0]) * (Y - A[1]) - (B[1] - A[1]) * (X - A[0]))

def points_inside(pxs, pys, hull):
    hull_points = hull.points[hull.vertices]
    nhull = len(hull_points)
    insides = np.zeros_like(pxs, dtype=bool)
    
    for ip, (px, py) in enumerate(zip(pxs, pys)):
        all_left=1
        i=0
        while all_left ==1:
            all_left *= point_left_to_a_line(px,py, hull_points[i], hull_points[i+1])
            if i == nhull-2:
                all_left *= point_left_to_a_line(px,py, hull_points[-1], hull_points[0]) #Last -> first point
                insides[ip] = all_left # 1 if True
                break
            i +=1
            
    return insides


def mask_hull(mask, ax=None):
    """
    returns 1D indices of pixels of interest
    """
    xcoords = np.tile(np.arange(mask.shape[0]),mask.shape[1])
    ycoords = np.repeat(np.arange(mask.shape[1]),mask.shape[0])
    coords = np.stack((xcoords[mask.ravel()], ycoords[mask.ravel()]), axis=1)
    hull = ConvexHull(coords)
    if ax is not None:
        for simplex in hull.simplices:
            ax.plot(coords[simplex, 0], coords[simplex, 1], 'k-')
    return points_inside(xcoords, ycoords, hull)
