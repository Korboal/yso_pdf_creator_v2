import numpy as np
import astropy
from astropy import units as u
from astropy.coordinates import SkyCoord

def load_data(path_textfile):
    data = np.loadtxt(path_textfile, dtype=str)
    return data

def str_tab_topcat(value):  # converts to string and adds tab at the end
    """
    Converts the value to string, replaces commas by dots and adds tab at the end. Used for saving into txt file.
    Made to be read by TOPCAT (because of dots)
    :param value: Value to be converted
    :return: Cleaned string with tab at the end
    """
    return str(value).replace(',', '.') + "\t"

def save_in_txt_topcat(text, filename):
    """
    Saves text in file, separating each element in text by tab and adding a new line below it. To be read by TOPCAT
    because saves with dots instead of commas.
    :param text: 1D array with words to write
    :param filename: Path and filename where to save
    :return: Returns nothing, but appends the text file
    """
    with open(filename, 'a+') as f:
        for word in text:
            f.write(str_tab_topcat(word))
        f.write('\n')

data = load_data("mol_clouds/mol_clouds_zucker.txt")
source_id = data[:, 0].astype(str)
l = data[:, 2].astype(float)
b = data[:, 3].astype(float)
#size_1 = data[:, 3].astype(float)
#size_2 = data[:, 4].astype(float)
#dist = data[:, 5].astype(float)

dist = data[:, 4].astype(float)

#print(ra[0], dec[0])



#print(data)
#print(astropy.coordinates.Angle(c_icrs.galactic.l))
#print(np.append(data, c_icrs.galactic[0]))

#save_in_txt_topcat(np.append(data, c_icrs.galactic[0]), "new_l_b.txt")

#print(c_icrs.galactic)

for i in range(np.size(source_id)):
    c_icrs = SkyCoord(l[i], b[i], frame='galactic', unit='deg')
    #mid_size = (size_1[i] + size_2[i]) / 2
    #size_cloud = np.tan(mid_size / 180 * np.pi / 2) * dist[i] * 2
    #save_in_txt_topcat([source_id[i], c_icrs.icrs.ra.value, c_icrs.icrs.dec.value, dist[i], size_cloud], "mol_clouds/conv_mol_cloud_parsecscale.txt")
    save_in_txt_topcat([source_id[i], c_icrs.icrs.ra.value, c_icrs.icrs.dec.value, dist[i]],
                       "mol_clouds/conv_mol_cloud_zucker.txt")