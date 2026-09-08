# Copyright 2026 Google LLC
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     https://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# For the analogy involving map projections, the 'basemap' library is used
# to provide a familiar visual reference for non-Cartesian coordinates.
# !pip install basemap

import itertools, jax, numpy, matplotlib.pyplot, scipy.interpolate
from mpl_toolkits import basemap   # Requires `pip install basemap`
from jax import numpy as jnp
jax.config.update("jax_enable_x64", True)


def get_map(axes):
  "Returns Basemap initialized for South Polar stereographic projection."
  bmap = basemap.Basemap(projection='spstere',
                         boundinglat=50,
                         lon_0=0,
                         resolution='l', ax=axes)
  bmap.drawcoastlines()
  bmap.fillcontinents(color='coral', lake_color='aqua')
  bmap.drawmapboundary(fill_color='aqua')
  bmap.drawparallels(numpy.linspace(-80., 80, 17))
  bmap.drawmeridians(numpy.arange(-180, 180, 36))
  # Adding a Cartesian grid overlay to show the map's (x,y) coordinates.
  x_min, x_max = axes.get_xlim()
  y_min, y_max = axes.get_ylim()
  for x in numpy.linspace(x_min, x_max, 11):
      axes.axvline(x, color='gray', linestyle='--', linewidth=0.75, zorder=1)
  for y in numpy.linspace(y_min, y_max, 11):
      axes.axhline(y, color='gray', linestyle='--', linewidth=0.75, zorder=1)
  return bmap

# Abbreviations for coordinate transformations back-and-forth
# between latitude/longitude and x/y map-plot coordinates - where
# our map size will be 10 cm x 10 cm, and coordinates are in cm.
MAP_SIZE_CM = 10


def xy_from_lat_lon(bmap, lat, lon):
  "Computes map-(x, y) from basemap, latitude, and longitude."
  x0, y0 = bmap(lon, lat)
  # Basemap is a bit unusual w.r.t. the role of '[xy]min' and '[xy]max'.
  return (MAP_SIZE_CM * (x0 - bmap.xmax) / (bmap.xmin - bmap.xmax),
          MAP_SIZE_CM * (y0 - bmap.ymax) / (bmap.ymin - bmap.ymax))


def xy_flight_path_from_ll_flight_path(bmap, lat_lon_from_t):
  def xy_from_t(t):
    # We are spelling out all the intermediate steps here.
    lat, lon = lat_lon_from_t(t)
    return xy_from_lat_lon(bmap, lat, lon)
  return xy_from_t


demo_path_lat_lon = numpy.array(
    [[-33.87, 151.21 - 360],   # Sydney
     [8.98, -79.52],           # Panama City
     [-34.60, -58.38],         # Buenos Aires
     [-33.92, 18.42],          # Cape Town
     [41.38, 2.17],            # Barcelona
     [4.17, 73.51],            # Male (Maldives)
     [35.68, 139.76],          # Tokyo
     [-33.87, 151.21]],        # Return to Sydney
    dtype=numpy.float64)
demo_path_ts = numpy.arange(demo_path_lat_lon.shape[0]) * 6 * 3600.0
demo_path_ll = scipy.interpolate.interp1d(demo_path_ts, demo_path_lat_lon,
                                          kind='linear', axis=0)
get_demo_path_xy = lambda bmap: (
    xy_flight_path_from_ll_flight_path(bmap, demo_path_ll))

fig, axes = matplotlib.pyplot.subplots(figsize=(24, 8), ncols=3)
ax_a, ax_b, ax_c = axes
ax_a.grid(); ax_b.grid()
ax_b.set_aspect('equal'); ax_c.set_aspect('equal')
ax_a.set_ylabel('Latitude'); ax_a.set_xlabel('Longitude')
bmap = get_map(ax_c)
flight_ts = numpy.linspace(0, demo_path_ts[-1], 1001)
demo_path_xy = get_demo_path_xy(bmap)
flight_lls = [demo_path_ll(t) for t in flight_ts]
ax_a.plot([lon for lat, lon in flight_lls],
          [lat for lat, lon in flight_lls], '-b')
ax_a.plot([lon for lat, lon in flight_lls[::50]],
          [lat for lat, lon in flight_lls[::50]], 'ob')
ax_b.set_xlabel('Map-X'); ax_b.set_ylabel('Map-Y')
flight_xys = [demo_path_xy(t) for t in flight_ts]
ax_b.plot([x for x, y in flight_xys],
          [y for x, y in flight_xys], '-k')
ax_b.plot([x for x, y in flight_xys[::50]],
          [y for x, y in flight_xys[::50]], 'ok')
x_map, y_map = bmap([lon for lat, lon in flight_lls],
                    [lat for lat, lon in flight_lls])
ax_c.plot(x_map, y_map, '-w', linewidth=3.5, label='Flight Path')
fig.savefig('/tmp/flight_path.pdf'); fig.show()
