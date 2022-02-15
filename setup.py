from setuptools import setup

setup(name='chargingmodel',
      version='0.1',
      description='Calculate optimal and uncontrolled EV Charging loads. Input: residual load time series, mobility profiles in the form of tripchains',
      author='Leo Strobel',
      packages=['chargingmodel'],
      zip_safe=False,
      install_requires=[
          'gurobipy',
          'numpy',
          'pandas'
      ])