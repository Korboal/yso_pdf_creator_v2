from setuptools import setup, Extension

module = Extension('sine_curve_2', sources=['sine_curve_2.pyx'])

setup(
    name='sine_curve_normal',
    version='1.0',
    author='Quantum',
    ext_modules=[module]
)
