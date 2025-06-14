from setuptools import find_packages, setup

package_name = 'linreg_analysis'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='anild',
    maintainer_email='anild@LAPTOP-8MU56RJE',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
                'height_weight_node = linreg_analysis.linear_regression_height_weight:main',
                'brain_weight_node = linreg_analysis.linear_regression_brain_node:main',
                'boston_housing_node = linreg_analysis.linear_regression_boston_node:main',

        ],
    },
)
