from setuptools import setup

setup(name='package',
version='0.1',
description='1st version equipe 3',
url='#',
author='equipe3',
author_email='equipe3hubfiep@gmail.com',
license='N/D',
packages=['package'],
package_dir={'package'},
install_requires=[
          'pandas',
          'numpy',
          'streamlit',
          'folium',
          'statistics',
          'opencage.geocoder'       
      ],
zip_safe=False)