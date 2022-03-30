import unittest
import tools
import numpy as np
import light_curve_V3
import sed_linefit_v4_0
import config_file as cf


class MyTestCase(unittest.TestCase):

    # Test light curve

    def test_fit_sin(self):
        N, amp, phase, offset, noise = 1000, 1., .5, 4., 3

        for period in range(10, 1000, 10):
            tt = np.linspace(0, 3 * period, N)
            omega = 2. * np.pi / period
            yy = amp * np.sin(omega * tt + phase) + offset

            res1, res2, res3, res4 = light_curve_V3.fit_sin(tt, yy, np.ones(np.size(tt)), 0)
            self.assertAlmostEqual(period, res1['period'])

    def test_fit_sin_with_const_period(self):
        N, amp, phase, offset, noise = 1000, 1., .5, 4., 3

        for period in range(10, 1000, 10):
            tt = np.linspace(0, 3 * period, N)
            omega = 2. * np.pi / period
            yy = amp * np.sin(omega * tt + phase) + offset

            res1 = light_curve_V3.fit_sin_with_const_period(tt, yy, np.ones(np.size(tt)), period)
            self.assertAlmostEqual(period, res1['period'])
            self.assertAlmostEqual(amp, res1['amp'])

    def test_convert_data_to_three_arrays(self):
        res1, res2, res3 = light_curve_V3.convert_data_to_three_arrays(np.array([1, 2, 3]))
        self.assertEqual(1, res2)
        self.assertEqual(2, res3)
        self.assertEqual(3, res1)

        res1, res2, res3 = light_curve_V3.convert_data_to_three_arrays(np.array([]))
        self.assertEqual([], list(res2))
        self.assertEqual([], list(res3))
        self.assertEqual([], list(res1))

        res1, res2, res3 = light_curve_V3.convert_data_to_three_arrays(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
        self.assertEqual([1, 4, 7], list(res2))
        self.assertEqual([2, 5, 8], list(res3))
        self.assertEqual([3, 6, 9], list(res1))

    def test_get_clean_data(self):
        clean_data, dirty_data = light_curve_V3.get_clean_data(np.asarray(["G", "G", "BP", "RP"]), "G", np.asarray([0, 0, 0, 0]),
                                                               np.asarray([10, 20, 30, 40]), np.asarray([1, 2, 3, 4]), np.asarray([100, 200, 300, 400]),
                                                               np.asarray([True, False, False, False]), np.asarray([5, 6, 7, 8]))
        self.assertEqual(6, clean_data[0][0])
        self.assertEqual(200, clean_data[0][1])
        self.assertEqual(5, dirty_data[0][0])
        self.assertEqual(100, dirty_data[0][1])

    def test_calc_mag_error(self):
        res = light_curve_V3.calc_mag_error(375.02, 1.0374)
        self.assertAlmostEqual(0.0030, res, delta=0.0001)

    # Test sedfit linefit

    def test_model_function_rayleigh_jeans_in_linear_space(self):
        res = sed_linefit_v4_0.model_function_rayleigh_jeans_in_linear_space(5, 3)
        self.assertEqual(8 * pow(10, -6), res)

    def test_model_function_rayleigh_jeans_in_log_space(self):
        res = sed_linefit_v4_0.model_function_rayleigh_jeans_in_log_space(5, 3)
        self.assertEqual(-18, res)

    def test_model_function_sed_slope_in_linear_space(self):
        res = sed_linefit_v4_0.model_function_sed_slope_in_linear_space(5, 2, 3)
        self.assertEqual(25000, res)

    def test_model_function_sed_slope_in_log_space(self):
        res = sed_linefit_v4_0.model_function_sed_slope_in_log_space(5, 10, 20)
        self.assertEqual(70, res)

    def test_get_param_for_rayleigh_jeans(self):
        res = sed_linefit_v4_0.get_param_for_rayleigh_jeans(np.array([10]), np.array([10]))
        self.assertAlmostEqual(-4, res['c'].value)
        res = sed_linefit_v4_0.get_param_for_rayleigh_jeans(np.power(10.0, [cf.sed_linear_line_wavelength_start * 0.9, cf.sed_linear_line_wavelength_start,
                                                             cf.sed_linear_line_wavelength_start * 1.1]), np.power(10.0, [-10, -11, -12]))
        self.assertAlmostEqual(abs(-11 - cf.line_fit_power_rayleigh_jeans * cf.sed_linear_line_wavelength_start), res['c'].value)

    def test_get_param_for_sed_linear_slope(self):
        res = sed_linefit_v4_0.get_param_for_sed_linear_slope(np.array([-5, -4.5]), np.array([3, 2.5]))
        self.assertEqual(-1, res['m'].value)
        self.assertEqual(-2, res['c'].value)

    def test_dlt_elem_by_indices(self):
        res = sed_linefit_v4_0.dlt_elem_by_indices(np.array([1, 2, 3, 4, 5, 6]), np.array([1, 2]))
        expected_result = [1, 4, 5, 6]

        for i in range(np.size(res)):
            self.assertEqual(expected_result[i], res[i])

    def test_dlt_three_arrays_by_indices(self):
        res1, res2, res3 = sed_linefit_v4_0.three_arr_dlt_elem_by_indices([1, 2, 3], [4, 5, 6], [7, 8, 9], [1])
        result = [res1, res2, res3]
        expected_result = [[1, 3], [4, 6], [7, 9]]

        for i in range(3):
            for j in range(len(expected_result[0])):
                self.assertEqual(expected_result[i][j], result[i][j])

    def test_sort_flux(self):
        flux = [0, np.nan, 2, 3, 4, 5, 6, 7, 8, 9]
        err_flux = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
        wavelengths = [20, 21, 22, 23, 24, 25, 26, 27, 28, 29]
        flags = ["1", "1", "2", "3", "4", ">", "U", "X", "0", "S"]

        x_good, y_good, error_good, x_up, y_up, error_up = sed_linefit_v4_0.sort_flux(wavelengths, flux, err_flux,
                                                                                      flags)
        self.assertEqual([2, 8], y_good)
        self.assertEqual([22, 28], x_good)
        self.assertEqual([12, 18], error_good)
        self.assertEqual([3, 5, 6], y_up)
        self.assertEqual([13, 15, 16], error_up)
        self.assertEqual([23, 25, 26], x_up)

    def test_calc_arith_mean(self):
        res = sed_linefit_v4_0.calc_weighted_arith_mean(np.array([7, 5, 8, 4]), np.array([9, 3, 2, 1]))
        self.assertEqual(98/15, res)

        res = sed_linefit_v4_0.calc_weighted_arith_mean(np.array([12, 15, 18]), np.array([20, 40, 30]))
        self.assertEqual(138 / 9, res)

    def test_calc_error_for_arith_mean(self):
        a = 5.0
        test_array = np.array([a, a])
        res = sed_linefit_v4_0.calc_error_for_weighted_arith_mean(test_array, 1 / test_array)
        self.assertAlmostEqual(0.5 * pow(2, 0.5) * a, res)

        a = 2.0
        test_array = np.array([a, a])
        res = sed_linefit_v4_0.calc_error_for_weighted_arith_mean(test_array, 1 / test_array)
        self.assertAlmostEqual(0.5 * pow(2, 0.5) * a, res)

        res = sed_linefit_v4_0.calc_error_for_weighted_arith_mean(np.array([3, 5]), np.array([0.5, 0.5]))
        self.assertAlmostEqual(2.915475947, res)

        res = sed_linefit_v4_0.calc_error_for_weighted_arith_mean(np.array([3, 3]), np.array([0.2, 0.8]))
        self.assertAlmostEqual(2.473863375, res)

        res = sed_linefit_v4_0.calc_error_for_weighted_arith_mean(np.array([3, 4, 5]), np.array([8, 6, 7]))
        self.assertAlmostEqual(2.321642237, res)

    def test_prepare_variables(self):
        res1, res2, res3 = sed_linefit_v4_0.prepare_data_for_interpolate([1, 1, 2, 3], [2, 3, 4, 5], [0.1, 0.2, 0.3, 0.4])
        self.assertEqual(3, np.size(res1))
        self.assertEqual(3, np.size(res2))
        self.assertEqual(3, np.size(res3))

        self.assertEqual(1, res1[0])
        self.assertEqual(2, res1[1])
        self.assertEqual(3, res1[2])

        self.assertEqual(4, res2[1])
        self.assertEqual(5, res2[2])

        self.assertEqual(0.3, res3[1])
        self.assertEqual(0.4, res3[2])

        self.assertGreaterEqual(res2[0], 2)
        self.assertLessEqual(res2[0], 3)

        self.assertAlmostEqual(2 + 1/3, res2[0])

        self.assertGreaterEqual(res3[0], 0.0)
        self.assertLessEqual(res3[0], 0.1)

    def test_calculate_ir_slope_error(self):
        x = np.array([1, 2, 3, 4, 5, 6, 7, 8])
        x_mean = 4.5
        y = np.array([1, 1, 2, 3, 6, 7, 8, 8])
        error = np.array([0, 0.5, 1.1, 0, 0.5, 1, 0, 1])

        slope = 1
        const = 0
        res = sed_linefit_v4_0.calculate_ir_slope_error(x, y, error, slope, const)

        sum_yy = np.sum(np.square(np.array([0, 1, 1, 1, 1, 1, 1, 0])))
        n = 6
        sum_xx = np.sum(np.square(x - x_mean))

        expected_result = np.sqrt(sum_yy / n) / np.sqrt(sum_xx)

        self.assertEqual(expected_result, res)

        x = np.array([1, 2])
        y = np.array([1, 2])
        error = np.array([0, 0.5])
        slope = 1
        const = 0
        res = sed_linefit_v4_0.calculate_ir_slope_error(x, y, error, slope, const)
        self.assertEqual(0, res)

        x = np.array([1, 2, 3, 4])
        y = np.array([1, 2, 3.5, 3.5])
        error = np.array([0, 30, 0.5, 20])
        slope = 1
        const = 0
        res = sed_linefit_v4_0.calculate_ir_slope_error(x, y, error, slope, const)
        self.assertAlmostEqual(np.sqrt(0.05), res)

    # Test tools

    def test_check_if_not_nan_or_zero(self):
        res = tools.check_if_not_nan_or_zero(0)
        self.assertEqual(False, res)
        res = tools.check_if_not_nan_or_zero(np.nan)
        self.assertEqual(False, res)
        res = tools.check_if_not_nan_or_zero("")
        self.assertEqual(False, res)
        res = tools.check_if_not_nan_or_zero(0.0)
        self.assertEqual(False, res)
        res = tools.check_if_not_nan_or_zero(35)
        self.assertEqual(True, res)

    def test_find_nearest(self):
        res = tools.find_nearest(np.asarray([1, 2, 3, 4, 5, 6]), 3.2)
        self.assertEqual(3, res)
        res = tools.find_nearest(np.asarray([1, 2, 3, 4, 5, 6]), 3.7)
        self.assertEqual(4, res)
        res = tools.find_nearest(np.asarray([1, 2, 3, 4, 5, 6]), -2)
        self.assertEqual(1, res)
        res = tools.find_nearest(np.asarray([1, 2, 3, 4, 5, 6]), 7)
        self.assertEqual(6, res)
        res = tools.find_nearest(np.asarray([1, 2, 3, 4, 5, 6]), 5)
        self.assertEqual(5, res)

    def test_find_nearest_index(self):
        res = tools.find_nearest_index(np.asarray([1, 2, 3, 4, 5, 6]), 3.2)
        self.assertEqual(2, res)
        res = tools.find_nearest_index(np.asarray([1, 2, 3, 4, 5, 6]), 3.7)
        self.assertEqual(3, res)
        res = tools.find_nearest_index(np.asarray([1, 2, 3, 4, 5, 6]), -2)
        self.assertEqual(0, res)
        res = tools.find_nearest_index(np.asarray([1, 2, 3, 4, 5, 6]), 7)
        self.assertEqual(5, res)
        res = tools.find_nearest_index(np.asarray([1, 2, 3, 4, 5, 6]), 5)
        self.assertEqual(4, res)

    def test_prepare_error_for_weights(self):
        res = tools.prepare_error_for_weights(np.asarray([0.0, 1.0, 2.0, 10e-35]), np.asarray([10, 10000, 10, 10]))
        self.assertEqual(list(np.square([0.075, 1 / 10000, 0.2, 0.075])), list(res))

    def test_get_ids_to_analyse(self):
        res = tools.get_ids_to_analyse(np.array(["1", "2", "3", "4", "5", "6"]), np.array(["3", "6", "9"]))
        self.assertEqual(['3', '6'], list(res))

    def test_conv_wavelength_to_freq(self):
        res = tools.conv_wavelength_to_freq(0.015)
        self.assertAlmostEqual(1.998616387e10, res, delta=1000)

    def test_compare_two_floats(self):
        res = tools.compare_two_floats(0.000409, 0.00041, 0.01)
        self.assertEqual(True, res)
        res2 = tools.compare_two_floats(0.004, 0.00041, 0.0001)
        self.assertEqual(False, res2)
        res = tools.compare_two_floats(0.00041, 0.00041, 0.000001)
        self.assertEqual(True, res)
        res = tools.compare_two_floats(100, 0.00041, 0.01)
        self.assertEqual(False, res)
        res = tools.compare_two_floats(0.00041, 100, 0.01)
        self.assertEqual(False, res)
        res = tools.compare_two_floats(-0.00041, 100, 0.01)
        self.assertEqual(False, res)
        res = tools.compare_two_floats(0.00041, -100, 0.01)
        self.assertEqual(False, res)
        res = tools.compare_two_floats(-100, 0.00041, 0.01)
        self.assertEqual(False, res)
        res = tools.compare_two_floats(100, -0.00041, 0.01)
        self.assertEqual(False, res)
        res = tools.compare_two_floats(-100, -0.00041, 0.01)
        self.assertEqual(False, res)

    def test_take_within_x_boundaries(self):
        res1, res2, res3 = tools.take_within_x_boundaries(np.array([1, 2, 3, 4, 5]), np.array([6, 7, 8, 9, 10]),
                                             np.array([11, 12, 13, 14, 15]), 1.5, 3, 1)
        self.assertEqual([2, 3], list(res1))

        res1, res2, res3 = tools.take_within_x_boundaries(np.array([1, 2, 3, 4, 5]), np.array([6, 7, 8, 9, 10]),
                                                          np.array([11, 12, 13, 14, 15]), 1.5, 3, 3)
        self.assertEqual([1, 2, 3, 4, 5], list(res1))

        res1, res2, res3 = tools.take_within_x_boundaries(np.array([1, 2, 3, 4, 5]), np.array([6, 7, 8, 9, 10]),
                                                          np.array([11, 12, 13, 14, 15]), 3, 3, 0)
        self.assertEqual([3], list(res1))

    def test_calculate_pearson_chi_squared(self):
        time = np.array([0, 1, 2, 3])
        y = np.array([1, 2, 3, 4])
        func = lambda xx: xx + 1
        res = tools.calculate_lightcurve_fit_error(time, y, np.zeros(np.size(y)), func)
        self.assertEqual(0, res)

        #time = np.array([0, 1, 2, 3])
        #y = np.array([1, 1, 4, 4])
        #func = lambda xx: xx + 1
        #res = tools.calculate_lightcurve_fit_error(time, y, np.zeros(np.size(y)), func)
        #self.assertAlmostEqual(5/6, res)

    # Test BB fits

    #def test_wiens_law(self):
        #res = my_sedfit_v6.wiens_law([0.5 * np.power(10.0, -6)], [10.0])
        #self.assertAlmostEqual(5796, res, delta=1)


if __name__ == '__main__':
    unittest.main()
