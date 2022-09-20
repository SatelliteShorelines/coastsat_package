
# CoastSat Package
Good news a prototype of the CoastSat package has been created! It has all the code included in CoastSat's GitHub repo and it is installable via PyPi. T
**Prototype URL**:

## Dependencies
1. geopandas # depends on GDAL
2. notebook
3. scikit-image
4.  earthengine-api>=0.1.304
5. "matplotlib
6. "astropy
7. pyqt5
8. scipy
9. numpy<1.23.0 # make it compatible with scipy

## Problems
1. **GDAL Dependency**
CoastSat relies on `geopandas` as a dependency which in turn has `GDAL` . `GDAL` is a uncompiled source distribution which means the user has to build GDAL on their system if they want to use the official release. Alternatively users can download a precompiled python wheel for their OS from [Christoph Gohlke’s website](https://www.lfd.uci.edu/~gohlke/pythonlibs/#gdal).
2. **ReadMe is not compatible with PyPi**
In order to allow the ReadMe to be uploaded to PyPi I had tor remove a few sections of the readme that weren't compatible. I kept the original Readme but renamed it to `READMEgithub.md`.

## Solutions
1. **GDAL Dependency**
For now I've found the easist solution is to install `geopandas` with `conda` as it takes care of the dirty work of creating a compatible environment for `geopandas` and `GDAL`. The downside of this approach is it means the user will have to run the commands to install `geopandas`,`pip`, and `coastsat_pkg` in a that exact order. When I tried the order  `pip`,  `coastsat_pkg`, then `geopandas` the installation failed on my windows machine.

	In the future I hope we can put the dependency for geopandas in the same pip package or conda package as coastsat but for now this solution works. Maybe we could detect what OS the user has and install the corresponding GDAL wheel on their system so that we can have geopandas as a pip dependency? [This guide shows how to install GDAL using pip](https://opensourceoptions.com/blog/how-to-install-gdal-for-python-with-pip-on-windows/)
