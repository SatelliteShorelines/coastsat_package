# CoastSat

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.2779293.svg)](https://doi.org/10.5281/zenodo.2779293)
[![Join the chat at https://gitter.im/CoastSat/community](https://badges.gitter.im/spyder-ide/spyder.svg)](https://gitter.im/CoastSat/community)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![GitHub release](https://img.shields.io/github/release/kvos/CoastSat)](https://GitHub.com/kvos/CoastSat/releases/)

CoastSat is an open-source software toolkit written in Python that enables users to obtain time-series of shoreline position at any coastline worldwide from 30+ years (and growing) of publicly available satellite imagery.

![Alt text](https://github.com/kvos/CoastSat/blob/master/doc/example.gif)

:point_right: Relevant publications:

- Shoreline detection algorithm: https://doi.org/10.1016/j.envsoft.2019.104528 (Open Access)
- Accuracy assessment and applications: https://doi.org/10.1016/j.coastaleng.2019.04.004
- Beach slope estimation: https://doi.org/10.1029/2020GL088365 (preprint [here](https://www.essoar.org/doi/10.1002/essoar.10502903.2))
- Satellite-derived shorelines along meso-macrotidal beaches: https://doi.org/10.1016/j.geomorph.2021.107707
- Beach-face slope dataset for Australia: https://doi.org/10.5194/essd-14-1345-2022

:point_right: Other repositories and addons related to this toolbox:

- [CoastSat.slope](https://github.com/kvos/CoastSat.slope): estimates the beach-face slope from the satellite-derived shorelines obtained with CoastSat.
- [CoastSat.PlanetScope](https://github.com/ydoherty/CoastSat.PlanetScope): shoreline extraction for PlanetScope Dove imagery (near-daily since 2017 at 3m resolution).
- [InletTracker](https://github.com/VHeimhuber/InletTracker): monitoring of intermittent open/close estuary entrances.
- [CoastSat.islands](https://github.com/mcuttler/CoastSat.islands): 2D planform measurements for small reef islands.
- [CoastSeg](https://github.com/dbuscombe-usgs/CoastSeg): image segmentation, deep learning, doodler.
- [CoastSat.Maxar](https://github.com/kvos/CoastSat.Maxar): shoreline extraction on Maxar World-View images (in progress)

:point_right: Visit the [CoastSat website](http://coastsat.wrl.unsw.edu.au/) to explore and download regional-scale datasets of satellite-derived shorelines and beach slopes generated with CoastSat.

:star: **If you like the repo put a star on it!** :star:

### Latest updates

:arrow_forward: *(2022/08/01)* CoastSat 2.0 (major release):

+ new download function for Landsat images (better alignment between panchromatic and multispectral bands)
+ quality-control steps added for fully automated shoreline extraction
+ post-processing of the shorelne time-series, including despiking and computing seasonal-averages.

:arrow_forward: *(2022/07/20)* Option to switch off panchromatic sharpening on Landsat 7, 8 and 9 imagery.

:arrow_forward: *(2022/05/02)* Compatibility with Landsat 9 and Landsat Collection 2

### Project description

Satellite remote sensing can provide low-cost long-term shoreline data capable of resolving the temporal scales of interest to coastal scientists and engineers at sites where no in-situ field measurements are available. CoastSat enables the non-expert user to extract shorelines from Landsat 5, Landsat 7, Landsat 8, Landsat 9 and Sentinel-2 images.
The shoreline detection algorithm implemented in CoastSat is optimised for sandy beach coastlines. It combines a sub-pixel border segmentation and an image classification component, which refines the segmentation into four distinct categories such that the shoreline detection is specific to the sand/water interface.

The toolbox has four main functionalities:

1. assisted retrieval from Google Earth Engine of all available satellite images spanning the user-defined region of interest and time period.
2. automated extraction of shorelines from all the selected images using a sub-pixel resolution technique.
3. intersection of the 2D shorelines with user-defined shore-normal transects.
4. tidal correction using measured water levels and an estimate of the beach slope.
5. post-processing of the shoreline time-series, despiking and seasonal averaging.
6. validation example at Narrabeen
