from reportlab.pdfgen import canvas
import numpy as np
import matplotlib.pyplot as plt
from star_class import Star
from config_file import temp_path
from typing import Union


def format_decim(value: float, decimal_amount: int) -> Union[str, float]:
    """
    Rounds the float value up to specific decimal amount and returns string value of it

    :param value: Original value
    :param decimal_amount: How many decimal places want to have left
    :return: Rounded value as a string or "NaN" if value is empty
    """
    if np.size(value) != 0:
        return round(float(value), decimal_amount)
    else:
        return "NaN"


def write_gaia_info_for_one_star(canvas_to_use: canvas.Canvas, star_obj: Star):
    """
    Takes in star information and saves it in a PDF file as strings

    :param canvas_to_use: Canvas object
    :param star_obj: Star object
    """

    alt_type = str(star_obj.other_types_simbad).replace(" ", "")
    temp_alt_type = alt_type.split('|')
    cleaned_alt_type = []
    [cleaned_alt_type.append(j) for j in temp_alt_type if j not in cleaned_alt_type]
    cleaned_alt_type = ('|'.join(cleaned_alt_type).replace("|*|", "|")).replace("|", "_")

    dist_error_temp = star_obj.parallax_error_ge3 / star_obj.parallax_ge3 * 100
    dist_txt = f"{format_decim(star_obj.distance_ge3, 0)} pc ± {int(format_decim(dist_error_temp, 0))} %"

    if star_obj.gaia_g_light_curve is None:
        star_obj.lightcurve_fit_and_plot(False, False, False, False)
    if star_obj.ir_slope25 is None:
        star_obj.sed_line_fit_and_plot("0001000")
    if star_obj.ir_slope20 is None:
        star_obj.sed_line_fit_and_plot("0000100")
    if star_obj.ztf_g_light_curve is None or star_obj.ztf_r_light_curve is None:
        star_obj.analyse_ztf_lightcurves(False, False, False, False, False)

    strings_to_write = np.array([f"GEDR3: {star_obj.source_id}",
                                 f"SIMBAD: {star_obj.name_simbad}",
                                 f"SIMBAD types: {star_obj.main_type_simbad}", cleaned_alt_type,
                                 f"Dist GE3: {dist_txt}",
                                 f"Extinction G2: {format_decim(star_obj.extinction_g2, 4)} mag",
                                 f"G mag G2: {format_decim(star_obj.g_mag_g2, 4)} mag",
                                 f"G mag GE3: {format_decim(star_obj.g_mag_ge3, 4)} mag",
                                 f"Slope 20: {format_decim(star_obj.ir_slope20, 2)}",
                                 f"Slope 25: {format_decim(star_obj.ir_slope25, 2)}",
                                 f"Per G2: {format_decim(star_obj.period_g2, 2)} ± {format_decim(star_obj.period_err_g2, 2)} d",
                                 f"GDR2: {star_obj.source_id_g2}"])
    """,
                                 "My per: " + str(format_decim(star_obj.gaia_g_light_curve.period_fit, 2)) + " days",
                                 "ZTF g per: " + str(format_decim(star_obj.ztf_g_light_curve.period_fit, 2)) + " d",
                                 "ZTF r per: " + str(format_decim(star_obj.ztf_r_light_curve.period_fit, 2)) + " d"])"""
    """,
    "RV Teff templ G2: " + str(format_decim(star_obj.teff_template_g2, 0)) + " K",
                                 "Teff G2: " + str(format_decim(star_obj.teff_val_g2, 0)) + " K",
                                 "NRMSE G G: " + str(format_decim(star_obj.gaia_g_light_curve.nrmse_fit, 2)),
                                 "NRMSE G BP with G G: " + str(format_decim(star_obj.gaia_bp_light_curve.nrmse_using_gaia_g, 2)),
                                 "NRMSE G RP with G G: " + str(format_decim(star_obj.gaia_rp_light_curve.nrmse_using_gaia_g, 2)),
                                 " ",
                                 "NRMSE ZTF g: " + str(format_decim(star_obj.ztf_g_light_curve.nrmse_fit, 2)),
                                 "NRMSE ZTF r: " + str(format_decim(star_obj.ztf_r_light_curve.nrmse_fit, 2)),
                                 "NRMSE ZTF g with G G: " + str(format_decim(star_obj.ztf_g_light_curve.nrmse_using_gaia_g, 2)),
                                 "NRMSE ZTF r with G G: " + str(format_decim(star_obj.ztf_r_light_curve.nrmse_using_gaia_g, 2)),
                                 "NRMSE ZTF g with ZTF r: " + str(format_decim(star_obj.ztf_g_light_curve.nrmse_using_other_ztf, 2)),
                                 "NRMSE ZTF r with ZTF g: " + str(format_decim(star_obj.ztf_r_light_curve.nrmse_using_other_ztf, 2))
                                 ])"""

    x_offset_left = 100
    x_offset_right = 320
    y_offset_og = 810
    offset_change = 15

    y_offset = y_offset_og
    for i in range(0, np.size(strings_to_write), 2):
        canvas_to_use.drawString(x_offset_left, y_offset, strings_to_write[i])
        y_offset = y_offset - offset_change

    y_offset = y_offset_og
    for i in range(1, np.size(strings_to_write), 2):
        canvas_to_use.drawString(x_offset_right, y_offset, strings_to_write[i])
        y_offset = y_offset - offset_change


def draw_hr_diagram_with_star(data, star_obj: Star, save_location: str):
    """
    Takes all star's data from a catalogue and plots on a HR diagram (BP-RP vs G_abs). Then plots the specific star on top of it as a red point.

    :param data: FITS data table
    :param star_obj: Star object
    :param save_location: Where to save HR diagram temporarily
    """

    g_mag = data.field('phot_g_mean_mag_ge3').astype(float)
    par = data.field('parallax_ge3').astype(float)
    abs_g = g_mag + 5 - 5 * np.log10(1000 / abs(par))

    bp = data.field('phot_bp_mean_mag_ge3').astype(float)
    rp = data.field('phot_rp_mean_mag_ge3').astype(float)

    plt.scatter(bp - rp, abs_g, s=5, color='grey')
    plt.xlabel("BP-RP [mag]")
    plt.ylabel("G abs [mag]")
    plt.title("HR diagram")
    plt.gca().invert_yaxis()

    g_mag_star = star_obj.g_mag_ge3
    dist_star = star_obj.distance_ge3
    abs_g_star = g_mag_star + 5 - 5 * np.log10(dist_star)

    bp_star = star_obj.bp_mag_ge3
    rp_star = star_obj.rp_mag_ge3

    plt.scatter(bp_star - rp_star, abs_g_star, marker="+", color='red', s=100)

    plt.savefig(save_location)

    plt.close('all')


def draw_pdf_graphs(canvas_to_use: canvas.Canvas, star_obj: Star, data_all_gaia_stars):
    """
    Draws all graphs for the specific star: lightcurves, SED, HR

    :param canvas_to_use: Canvas object
    :param star_obj: Star object
    :param data_all_gaia_stars: All stars' data from FITS file
    """

    init_x_image = 23

    init_y_image_page_n = 480
    image_width = 260
    image_height = int(image_width * 3/4)
    image_spacing = 13

    init_y_image_page_1 = init_y_image_page_n - image_height  #345

    hr_dia_dir = temp_path + "/hr_" + str(star_obj.source_id) + ".png"
    draw_hr_diagram_with_star(data_all_gaia_stars, star_obj, hr_dia_dir)

    left_images_directory_page_1 = [hr_dia_dir, star_obj.raw_data_gaia_bp_band_output_png,
                                    star_obj.raw_data_gaia_g_band_output_png]
    right_images_directory_page_1 = [star_obj.sed_fit_directory_png, star_obj.raw_data_gaia_rp_band_output_png,
                                     star_obj.fitted_curve_gaia_g_output_png]
    draw_graphs_on_canvas(canvas_to_use, image_height, image_spacing, image_width, init_x_image, init_y_image_page_1,
                          left_images_directory_page_1, right_images_directory_page_1)
    canvas_to_use.showPage()

    left_images_directory_page_2 = [star_obj.periodogram_gaia_g_png_directory, star_obj.folded_light_curve_gaia_bp_with_gaia_g_output_png,
                                    star_obj.ztf_output_pictures_raw, star_obj.ztf_output_pictures_folded_g_with_gaia_g_fit]
    right_images_directory_page_2 = [star_obj.folded_light_curve_gaia_g_output_png, star_obj.folded_light_curve_gaia_rp_with_gaia_g_output_png,
                                     star_obj.ztf_output_pictures_folded, star_obj.ztf_output_pictures_folded_r_with_gaia_g_fit]
    draw_graphs_on_canvas(canvas_to_use, image_height, image_spacing, image_width, init_x_image, init_y_image_page_n,
                          left_images_directory_page_2, right_images_directory_page_2)
    canvas_to_use.showPage()

    left_images_directory_page_3 = [star_obj.ztf_output_pictures_fit_ztf_g, star_obj.ztf_output_pictures_fit_ztf_r, star_obj.sed_bar_dir_png, None]
    right_images_directory_page_3 = [star_obj.ztf_output_pictures_folded_ztf_g, star_obj.ztf_output_pictures_folded_ztf_r, None, star_obj.output_gaia_all_bands_raw_data_png]
    draw_graphs_on_canvas(canvas_to_use, image_height, image_spacing, image_width, init_x_image, init_y_image_page_n,
                          left_images_directory_page_3, right_images_directory_page_3)
    canvas_to_use.showPage()

    left_images_directory_page_4 = [star_obj.ztf_output_pictures_periodogram_ztf_g_png, star_obj.ztf_output_pictures_folded_g_with_ztf_r_fit,
                                    star_obj.output_frequency_periodogram_gaia_g_png, star_obj.output_frequency_periodogram_ztf_r_png]
    right_images_directory_page_4 = [star_obj.ztf_output_pictures_periodogram_ztf_r_png, star_obj.ztf_output_pictures_folded_r_with_ztf_g_fit,
                                     star_obj.output_frequency_periodogram_ztf_g_png, None]
    draw_graphs_on_canvas(canvas_to_use, image_height, image_spacing, image_width, init_x_image, init_y_image_page_n,
                          left_images_directory_page_4, right_images_directory_page_4)
    """canvas_to_use.showPage()

    left_images_directory_page_5 = [star_obj.output_multiband_frequency_periodogram_gaia_png, star_obj.output_multiband_gaia_fit_gaia_bp_png,
                                    star_obj.output_multiband_frequency_periodogram_ztf_png, star_obj.output_multiband_ztf_fit_ztf_r_png]
    right_images_directory_page_5 = [star_obj.output_multiband_gaia_fit_gaia_g_png, star_obj.output_multiband_gaia_fit_gaia_rp_png,
                                     star_obj.output_multiband_ztf_fit_ztf_g_png, None]
    draw_graphs_on_canvas(canvas_to_use, image_height, image_spacing, image_width, init_x_image, init_y_image_page_n,
                          left_images_directory_page_5, right_images_directory_page_5)
    canvas_to_use.showPage()

    left_images_directory_page_6 = [star_obj.output_multiband_frequency_periodogram_all_png, star_obj.output_multiband_all_fit_gaia_bp_png,
                                    star_obj.output_multiband_all_fit_ztf_g_png]
    right_images_directory_page_6 = [star_obj.output_multiband_all_fit_gaia_g_png, star_obj.output_multiband_all_fit_gaia_rp_png,
                                     star_obj.output_multiband_all_fit_ztf_r_png]
    draw_graphs_on_canvas(canvas_to_use, image_height, image_spacing, image_width, init_x_image, init_y_image_page_n,
                          left_images_directory_page_6, right_images_directory_page_6)
    canvas_to_use.showPage()"""


def draw_graphs_on_canvas(canvas_to_use, image_height, image_spacing, image_width, init_x_image, init_y_image,
                          left_images_directory, right_images_directory):
    for i in range(len(left_images_directory)):
        draw_canvas_image(canvas_to_use, left_images_directory[i], init_x_image, init_y_image, image_width)
        draw_canvas_image(canvas_to_use, right_images_directory[i], init_x_image + image_width + image_spacing,
                          init_y_image, image_width)
        init_y_image = init_y_image - image_height - image_spacing


def draw_canvas_image(canvas_to_use: canvas.Canvas, image_directory: str, x_image: float, y_image: float,
                      image_width: float):
    """
    Draws image in the canvas, if image directory is given and the image exists. Otherwise prints that no image is given

    :param canvas_to_use: canvas where to draw
    :param image_directory: path to the image
    :param x_image: x coordinate of image top left corner
    :param y_image: y coordinate of image top left corner
    :param image_width: width of image in pixels
    """
    if image_directory is not None:
        try:
            canvas_to_use.drawImage(image_directory, x_image, y_image, width=image_width, preserveAspectRatio=True)
        except:
            print(f"No image at {image_directory}")


def create_new_canvas_for_a_star(star_obj: Star, data_all_gaia_stars):
    """
    Takes in a specific star's data and all stars' data from FITS file and makes one PDF document

    :param star_obj: Star object
    :param data_all_gaia_stars: All data from FITS file (uses specifically BP, RP, parallax, G)
    """

    c = canvas.Canvas(star_obj.pdf_dir)
    c.setLineWidth(.3)
    c.setFont('Helvetica', 12)

    write_gaia_info_for_one_star(c, star_obj)
    draw_pdf_graphs(c, star_obj, data_all_gaia_stars)

    c.save()
