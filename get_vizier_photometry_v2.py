import numpy as np
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import time
import os
from os import listdir
from os.path import isfile, join
import tools


def load_coord_to_analyse(filename):
    data = tools.load_data(filename)
    ids = data[:, 0].astype(int)
    ras = data[:, 1].astype(float)
    decs = data[:, 2].astype(float)
    return ids, ras, decs


options = Options()
browser = webdriver.Chrome('/Users/Quantum/Downloads/chromedriver', options=options)

ids, ras, decs = load_coord_to_analyse("ids_with_coord/all/" + ".txt")
for i in range(np.size(ids)):
    url = "https://vizier.cds.unistra.fr/viz-bin/sed?-c=" + str(ras[i]) + "%20" + str(decs[i]) + "&-c.rs=1"
    print(ids[i], ras[i], decs[i])
    browser.get((url))
    time.sleep(5)
    directory = "../../../Downloads/"
    trying = True
    while trying:
        try:
            all_files = [f for f in listdir(directory) if
                         isfile(join(directory, f))]
            if "vizier_votable.vot" in all_files:
                old_file = os.path.join(directory, "vizier_votable.vot")
                new_file = os.path.join(directory, str(ids[i]) + ".vot")
                os.rename(old_file, new_file)
                trying = False
            else:
                print("file not found yet")
                time.sleep(3)
        except:
            print("trying again")
            time.sleep(3)