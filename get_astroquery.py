from astroquery.ipac.irsa import Irsa
from astroquery.esasky import ESASky
import astropy.coordinates as coord
import astropy.units as u
import numpy as np
import pandas as pd
import os
from astroquery.vsa import Vsa

#Irsa.print_catalogs()
import tools


def get_irsa_astro_query(ra, dec, catalogue: str, xmatch_radius: float):
    query_result = Irsa.query_region(coord.SkyCoord(ra, dec, unit=(u.deg, u.deg), frame='icrs'),
                                     catalog=catalogue, radius=xmatch_radius * u.arcsec)

    if np.size(query_result) != 0:
        xmatched_star = query_result[np.argmin(query_result['dist'])]
    else:
        xmatched_star = None
    return xmatched_star

def add_irsa_missing_entry(all_source_ids, table_path, table_empty_path, source_id, ra, dec, catalogue, xmatch_radius):
    if source_id not in all_source_ids:
        result = get_irsa_astro_query(ra, dec, catalogue, xmatch_radius)
        if result is not None:
            tb = pd.DataFrame([list(result)], columns=list(result.colnames))
            tb = tb.assign(source_id_gdr2=source_id)
            tb.to_csv(table_path, mode='a', header=not os.path.exists(table_path))
        if result is None:
            tools.save_in_txt_topcat([source_id], table_empty_path)



def get_xmm_astro_query(ra, dec, catalogue: str, xmatch_radius: float):
    query_result = ESASky.query_region_catalogs(position=coord.SkyCoord(ra, dec, unit=(u.deg, u.deg), frame='icrs'),
                                                radius=xmatch_radius * u.arcsec,
                                                catalogs=catalogue)

    if np.size(query_result) != 0:
        xmatched_star = query_result
    else:
        xmatched_star = None
    return xmatched_star

def add_xmm_missing_entry(all_source_ids, table_path, table_empty_path, source_id, ra, dec, catalogue, xmatch_radius):
    if source_id not in all_source_ids:
        result = get_xmm_astro_query(ra, dec, catalogue, xmatch_radius)
        if result is not None:
            for table in result:
                for row_number in range(len(table)):
                    tb = pd.DataFrame([list(np.array(table)[row_number])], columns=list(table.colnames))
                    tb = tb.assign(source_id_gdr2=source_id)
                    tb.to_csv(table_path, mode='a', header=not os.path.exists(table_path))
        if result is None:
            tools.save_in_txt_topcat([source_id], table_empty_path)


def get_vvv_astro_query(ra, dec, catalogue: str, xmatch_radius: float):
    query_result = Vsa.query_region(coordinates=coord.SkyCoord(ra, dec, unit=(u.deg, u.deg), frame='icrs'),
                                    radius=xmatch_radius * u.arcsec, database=catalogue, system='J2000', programme_id='VVV')

    if np.size(query_result) != 0:
        xmatched_star = query_result
    else:
        xmatched_star = None
    return xmatched_star

def add_vvv_missing_entry(all_source_ids, table_path, table_empty_path, source_id, ra, dec, catalogue, xmatch_radius):
    if source_id not in all_source_ids:
        #catalogue = "VVVDR4"
        result = get_vvv_astro_query(ra, dec, catalogue, xmatch_radius)
        if result is not None:
            for table in result:
                tb = pd.DataFrame([list(table)], columns=list(table.colnames))
                tb = tb.assign(source_id_gdr2=source_id)
                tb.to_csv(table_path, mode='a', header=not os.path.exists(table_path))
        if result is None:
            tools.save_in_txt_topcat([source_id], table_empty_path)


"""
result = get_astro_query(ra, dec, 'allwise_p3as_psd', 50)
tb = pd.DataFrame([list(result)], columns=list(result.colnames))
tb = tb.assign(source_id=source_id_star)
tb.to_csv(output_path, mode='a', header=not os.path.exists(output_path))"""


"""ra, dec = 84.10333479819896, -6.291824642281453
ra, dec = 68.06700096824186, 18.212806236086358
source_id_star = "3314309890186259712"
output_path = "output123.csv"

if os.path.isfile(output_path):
    tb = pd.read_csv(output_path)
    all_source_ids = tb["source_id_gdr2"].values.astype(str)
else:
    all_source_ids = []
add_xmm_missing_entry(all_source_ids, output_path, source_id_star, ra, dec, 'XMM-EPIC-STACK', 1)
tb = pd.read_csv(output_path)
print(tb)"""



"""ra, dec = 84.10333479819896, -6.291824642281453
source_id_star = "123123123"
output_path = "output1234.csv"

if os.path.isfile(output_path):
    tb = pd.read_csv(output_path)
    all_source_ids = tb["source_id_gdr2"].values.astype(str)
else:
    all_source_ids = []
add_xmm_missing_entry(all_source_ids, output_path, source_id_star, ra, dec, 'allwise_p3as_psd', 0.1)

if os.path.isfile(output_path):
    tb = pd.read_csv(output_path)
    print(tb)"""



"""allwise_p3as_psd                AllWISE Source Catalog
allsky_4band_p3as_psd           WISE All-Sky Source Catalog
fp_psc                          2MASS All-Sky Point Source Catalog (PSC)
slphotdr4                       SEIP Source List
dr4_clouds_hrel                 C2D Fall '07 High Reliability (HREL) CLOUDS Catalog (CHA_II, LUP, OPH, PER, SER)
dr4_off_cloud_hrel              C2D Fall '07 High Reliability (HREL) OFF-CLOUD Catalog (CHA_II, LUP, OPH, PER, SER)
dr4_cores_hrel                  C2D Fall '07 High Reliability (HREL) CORES Catalog
dr4_stars_hrel                  C2D Fall '07 High Reliability (HREL) STARS Catalog
csi2264t1                       CSI 2264 Object Table
cygx_cat                        Cygnus-X Catalog
glimpse_s07                     GLIMPSE I Spring '07 Catalog (highly reliable)
glimpse2_v2cat                  GLIMPSE II Spring '08 Catalog (highly reliable)
glimpse2ep1c08                  GLIMPSE II Epoch 1 December '08 Catalog (highly reliable)
glimpse2ep2mra09                GLIMPSE II Epoch 2 November '09 More Reliable Archive (more reliable)
glimpse3d_v1cat_tbl             GLIMPSE 3D, 2007-2009 Catalog (highly reliable)
glimpse3dep1c                   GLIMPSE 3D Epoch 1 Catalog (highly reliable)
glimpse3dep2mra                 GLIMPSE 3D Epoch 2 More Reliable Archive (more complete, less reliable)
glimpse360c                     GLIMPSE360 Catalog (highly reliable)
velcarc                         Vela-Carina Catalog (highly reliable)
glimpsesmogc                    SMOG Catalog (highly reliable)
glimpsecygxc                    GLIMPSE Cygnus-X Catalog (highly reliable)
mipsgalc                        MIPSGAL Catalog
taurus_2008_2_1                 Taurus Catalog October 2008 v2.1
ysoggd1215obj                   YSOVAR GGD 12-15 Object Table
ysoi20050obj                    YSOVAR IRAS 20050+2720 Object Table
ysol1688obj                     YSOVAR L1688 Object Table
yson1333obj                     YSOVAR NGC1333 Object Table
ppsc_70                         PACS Point Source Catalog: 70 microns
ppsc_100                        PACS Point Source Catalog: 100 microns"""

#ztf_objects_dr9                 ZTF DR9 Objects