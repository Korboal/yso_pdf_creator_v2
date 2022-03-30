import numpy as np
from scipy.optimize import minimize

default_nebula_size_pc = 5
#nebulae_data = tools.load_data("neb1.txt")


def convert_spherical_to_cartesian(ra, dec, dist):
    ra = ra / 180 * np.pi
    dec = dec / 180 * np.pi
    return dist * np.cos(ra) * np.cos(dec), dist * np.sin(ra) * np.cos(dec), dist * np.sin(dec)


def find_dist_to_nebulae(ra, dec, dist, nebulae_data):
    if dist is None:
        x_star, y_star, z_star = convert_spherical_to_cartesian(ra, dec, nebulae_data[:, 3].astype(float))
    else:
        x_star, y_star, z_star = convert_spherical_to_cartesian(ra, dec, dist)
    x_nebulae, y_nebulae, z_nebulae = convert_spherical_to_cartesian(nebulae_data[:, 1].astype(float), nebulae_data[:, 2].astype(float), nebulae_data[:, 3].astype(float))

    return np.sqrt(np.square(x_star - x_nebulae) + np.square(y_star - y_nebulae) + np.square(z_star - z_nebulae))


def find_distance_between_neb_and_star(dist_star, ra_star, dec_star, x_nebulae, y_nebulae, z_nebulae, size_neb):
    x_star, y_star, z_star = convert_spherical_to_cartesian(ra_star, dec_star, dist_star)
    return np.sqrt(np.square(x_star - x_nebulae) + np.square(y_star - y_nebulae) + np.square(z_star - z_nebulae)) - size_neb


def find_closest_nebulae(ra_star, dec_star, dist_star, nebulae_data):
    nebulae_sizes = nebulae_data[:, 4].astype(float) / 2   # convert sizes of nebulae to radii
    nebulae_sizes = np.where(nebulae_sizes > 0, nebulae_sizes, default_nebula_size_pc)

    star_parallax_error = 0.1
    nebula_parallax_error = 0.1

    dist_to_nebula = nebulae_data[:, 3].astype(float)
    dist_to_nebula_min = np.copy(dist_to_nebula) * (1 - nebula_parallax_error) - nebulae_sizes
    dist_to_nebula_max = np.copy(dist_to_nebula) * (1 + nebula_parallax_error) + nebulae_sizes

    within_nebulae_dist_indices = np.logical_and.reduce((dist_star * (1 - star_parallax_error) < dist_to_nebula_max, dist_star * (1 + star_parallax_error) > dist_to_nebula_min))

    if (~within_nebulae_dist_indices).all():
        within_nebulae_dist_indices = np.invert(within_nebulae_dist_indices)
        new_nebulae_data = nebulae_data[within_nebulae_dist_indices, :]
    else:
        new_nebulae_data = nebulae_data[within_nebulae_dist_indices, :]
        dist_star = None

    distances = find_dist_to_nebulae(ra_star, dec_star, dist_star, new_nebulae_data) - nebulae_sizes[within_nebulae_dist_indices]
    min_dist_index = np.argmin(distances)
    closest_nebula = new_nebulae_data[min_dist_index, 0]
    nebula_size = (nebulae_sizes[within_nebulae_dist_indices])[min_dist_index]
    min_dist = distances[min_dist_index]
    return closest_nebula, min_dist, nebula_size


def find_nebula_relation(ra, dec, dist, nebulae_data):
    n = 3
    closest_nebula, min_dist, nebula_size = find_closest_nebulae(ra, dec, dist, nebulae_data)
    max_dist_with_low_confidence = nebula_size + 10
    if min_dist <= 0:
        return closest_nebula, closest_nebula, min_dist + nebula_size, 1.0
    elif min_dist <= max_dist_with_low_confidence:
        #return str(int((min_dist + nebula_size) // nebula_size) * "?") + "_" + closest_nebula, closest_nebula, min_dist + nebula_size, 1 - int((min_dist + nebula_size) // nebula_size) / (n + 1)
        return closest_nebula, closest_nebula, min_dist + nebula_size, 1 - int((min_dist) // ((max_dist_with_low_confidence) / n) + 1) / (n + 1)
    else:
        return closest_nebula, "_", min_dist + nebula_size, 0.0


def find_nebula(star_obj, nebulae_data, filename_to_save):
    from tools import save_in_txt_topcat
    ra = star_obj.ra
    dec = star_obj.dec
    dist = star_obj.distance_ge3
    closest_nebula, related_nebula_name, distance_to_nebula_center, confidence = find_nebula_relation(ra, dec, dist, nebulae_data)
    save_in_txt_topcat([star_obj.source_id, ra, dec, star_obj.pmra, star_obj.pmdec, closest_nebula, related_nebula_name, distance_to_nebula_center, confidence], filename_to_save)



#print(find_nebula_relation(314.9, 44.3, 800, nebulae_data))
# # source_id123	closest_neb	nebula	dist

"""def find_distance_between_neb_and_star(dist_star, ra_star, dec_star, x_nebulae, y_nebulae, z_nebulae, size_neb):
    x_star, y_star, z_star = convert_spherical_to_cartesian(ra_star, dec_star, dist_star)
    return np.sqrt(np.square(x_star - x_nebulae) + np.square(y_star - y_nebulae) + np.square(z_star - z_nebulae)) - size_neb

def find_closest_nebulae(ra, dec, dist, nebulae_data):
    nebulae_sizes = nebulae_data[:, 4].astype(float) / 2   # convert sizes of nebulae to radii
    nebulae_sizes = np.where(nebulae_sizes > 0, nebulae_sizes, default_nebula_size_pc)
    distances = find_dist_to_nebulae(ra, dec, dist, nebulae_data) - nebulae_sizes
    min_dist_index = np.argmin(distances)
    closest_nebula = nebulae_data[min_dist_index, 0]
    nebula_size = nebulae_sizes[min_dist_index]

    if distances[min_dist_index] > 0:
        x_nebulae, y_nebulae, z_nebulae = convert_spherical_to_cartesian(nebulae_data[min_dist_index, 1].astype(float), nebulae_data[min_dist_index, 2].astype(float), nebulae_data[min_dist_index, 3].astype(float))
        new_dist = minimize(find_distance_between_neb_and_star, dist, args=(ra, dec, x_nebulae, y_nebulae, z_nebulae, nebula_size), bounds=((dist*0.8, dist*1.2),))['x']

        distances = find_dist_to_nebulae(ra, dec, new_dist, nebulae_data) - nebulae_sizes
        min_dist_index = np.argmin(distances)
        closest_nebula = nebulae_data[min_dist_index, 0]
        nebula_size = nebulae_sizes[min_dist_index]

    min_dist = distances[min_dist_index]
    return closest_nebula, min_dist, nebula_size"""