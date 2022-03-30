import numpy as np
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import time
import os
from os import listdir
from os.path import isfile, join
import webbrowser
import subprocess

def load_data(path_textfile):
    data = np.loadtxt(path_textfile, dtype=str)
    return data

def load_coord_to_analyse(filename):
    data = load_data(filename)
    ids = data[:, 0].astype(int)
    ras = data[:, 1].astype(float)
    decs = data[:, 2].astype(float)
    return ids, ras, decs


def v1():
    for i in range(np.size(ids)):
        print(ids[i], ras[i], decs[i])
        url = "http://vizier.u-strasbg.fr/vizier/sed/?submitSimbad=Photometry&-c=" + str(ras[i]) + "%20" + str(decs[i]) + "&-c.r=1&-c.u=arcsec&show_settings=1"
        webbrowser.open(url)
        #subprocess.Popen(["output_pdf/Duplicate/" + str(ids[i]) + ".pdf"], shell=False, stdout=subprocess.PIPE)
        #os.system("output_pdf/Duplicate/" + str(ids[i]) + ".pdf")
        #path = "../../../Documents/Duplicate/154900851685204992.pdf"
        #path = "/Users/Quantum/Dropbox/Python_projects/master_thesis/output_pdf/Duplicate/93653484171803520.pdf"
        #os.system("output_pdf/Duplicate/" + str(ids[i]) + ".pdf")
        #os.system(path)
        #webbrowser.get().open(path)
        a = input("ready?")

def v2():
    while True:
        id_to_search = input("Input ID: ")
        index = np.where(ids == id_to_search)
        print(ids[index][0], ras[index][0], decs[index][0])
        url = "http://vizier.u-strasbg.fr/vizier/sed/?submitSimbad=Photometry&-c=" + str(ras[index][0]) + "%20" + str(
            decs[index][0]) + "&-c.r=1&-c.u=arcsec&show_settings=1"
        webbrowser.open(url)

ms = "ms" #done
rg = "rg" # done
rg2 = "rg2" # done
others = "others" # done
yso = "yso" # done

data = load_data("ids/all.txt")
ids = data[:, 0]
ras = data[:, 1]
decs = data[:, 2]

v2()
