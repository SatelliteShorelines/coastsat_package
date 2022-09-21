## Install CoastSat with Pip
[PyPi Repo](https://pypi.org/project/coastsat-package/)
- Note: This package is still in beta and being actively developed.
1. Create a conda environment
- We will install the CoastSat package and its dependencies in this environment.
	> `conda create --name coastsat_pkg -y`	
2. Activate your conda environment
	>`conda activate coastsat_pkg `	
3. Install geopandas with Conda
	- [Geopandas](https://geopandas.org/en/stable/) has [GDAL](https://gdal.org/) as a dependency so its best to install it with conda.
	>`conda install geopandas -y`
4. Install pip in your conda environment
- We will use pip to install the coastsat package from [PyPi](https://pypi.org/)
	>  `conda install pip -y`
5. Install the pip package
- `-U` (upgrade flag) this gets the latest release of the CoastSat package
	> `pip install coastsat_package -U`


