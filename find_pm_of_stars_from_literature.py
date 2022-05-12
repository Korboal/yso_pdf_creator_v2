import numpy as np
import tools

filename_to_load = "mol_clouds/stars_to_find_pm_nebula_v2/herczeg_2019_serpens_main_low_ge3_low_err.txt"
filename_to_save = "mol_clouds/stars_to_find_pm_nebula_v2/herczeg_2019_serpens_main_low_pmra_pmdec.txt"

tools.save_in_txt_topcat(["# cluster\tpmra\tpmra_std\tpmdec\tpmdec_std\tstars_amount"], filename_to_save)

data = tools.load_data(filename_to_load)
#source_ids = data[:, 0]
cluster_names = data[:, 1]
#ra = data[:, 2].astype(float)
#dec = data[:, 3].astype(float)
#parallax = data[:, 4].astype(float)
#parallax_error = data[:, 5].astype(float)
pmra = data[:, 7].astype(float)
#pmra_error = data[:, 8].astype(float)
pmdec = data[:, 9].astype(float)
#pmdec_error = data[:, 10].astype(float)

clusters_to_go_through = np.unique(cluster_names)

for cluster in clusters_to_go_through:
    print(cluster)
    ids_in_cluster = np.where(cluster_names == cluster)[0]
    pmra_stars_in_cluster = pmra[ids_in_cluster]
    pmdec_stars_in_cluster = pmdec[ids_in_cluster]

    pmra_cluster_median = np.median(pmra_stars_in_cluster)
    pmdec_cluster_median = np.median(pmdec_stars_in_cluster)

    pmra_cluster_std = np.std(pmra_stars_in_cluster)
    pmdec_cluster_std = np.std(pmdec_stars_in_cluster)

    tools.save_in_txt_topcat([cluster, pmra_cluster_median, pmra_cluster_std, pmdec_cluster_median, pmdec_cluster_std, np.size(ids_in_cluster)],
                             filename_to_save)
