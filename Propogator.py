
from datetime import datetime
from sgp4.api import Satrec, WGS72
import numpy as np
from skyfield.api import EarthSatellite, wgs84
import pandas as pd
import random
import Plotter
import os
import subprocess
prediction_window_cache = {}

global count_of_SPG4_failed_propogations

def run_Monte_Carlo_reentry_locations(max_monte_carlo_iterations, df_reentry_tle_mass, ts, tle_expiration):
  # Monte Carlo to test the distribution of reentries over space between 2002-2021
  # Monte Carlo to test the distribution of reentries over space between 2002-2021
  iteration = 0
  reentry_count = 0
  centrality_tolerance = 1e-5
  hasReachedCentrality = False

  bin_latslongs = pd.DataFrame()

  while iteration < max_monte_carlo_iterations and hasReachedCentrality == False:
    iteration = iteration + 1
    df_successfully_propogated, errors = propogate_df(True, df_reentry_tle_mass, ts, tle_expiration)
    print(len(errors['reampled reentry time'])+ len(errors['no_prop_solution']))
    # Produce csv file and call Matlab binning
    outdir = './MonteCarloFiles/' + str(iteration)
    if not os.path.exists(outdir):
      os.mkdir(outdir)
    df_successfully_propogated.to_csv(
      './MonteCarloFiles/' + str(iteration) + '/propogated_satellites_' + str(iteration) + '.csv')
    subprocess.run(['./run_matlab_equal_area_binning.sh', str(iteration)], capture_output=True)

    # Read in binned values
    binned_data = pd.read_csv('./MonteCarloFiles/' + str(iteration) + '/binned_positions_' + str(iteration) + '.csv')
    binned_data.columns = ['Bin Lat', 'Bin Long', 'New Count']
    if iteration == 1:
      bin_latslongs = binned_data
      bin_latslongs['Total Count'] = bin_latslongs['New Count']
      bin_latslongs.drop(columns = ['New Count'], inplace=True)
      bin_latslongs['previous_normalized_bins'] = np.divide(bin_latslongs['Total Count'],
                                                            len(df_successfully_propogated))
      reentry_count = reentry_count + len(df_successfully_propogated)
      bin_latslongs['normalized_bins'] = np.divide(bin_latslongs['Total Count'], reentry_count)
      continue
    else:
      bin_latslongs = pd.merge(bin_latslongs, binned_data, how='outer', on=['Bin Lat', 'Bin Long'])
      bin_latslongs['Total Count'].fillna(0, inplace = True)
      bin_latslongs['New Count'].fillna(0, inplace = True)
      bin_latslongs['previous_normalized_bins'].fillna(0, inplace = True)
      bin_latslongs['Total Count'] = bin_latslongs['Total Count'] + bin_latslongs['New Count']
      bin_latslongs.drop(columns=['New Count'], inplace=True)

    reentry_count = reentry_count + len(df_successfully_propogated)
    bin_latslongs['normalized_bins'] = np.divide(bin_latslongs['Total Count'], reentry_count)
    print("Length of Normalized Bins: " + str(len(bin_latslongs)))

    hasReachedCentrality = determine_if_centrality(bin_latslongs['previous_normalized_bins'],bin_latslongs['normalized_bins'], centrality_tolerance)
    bin_latslongs['previous_normalized_bins'] = bin_latslongs['normalized_bins']

  normalized_vis_data = bin_latslongs

  if (hasReachedCentrality == False):
    print("Warning: Centrality not reached to specified criteria.")
  else:
    print("Monte Carlo has reached centrality to specified criteria.")
    print("Reach centrality in: " + str(iteration))
  normalized_vis_data.to_csv("./normalized_reentry_location_data.csv")

  return normalized_vis_data

def determine_if_centrality(old_bins, new_bins, centrality_criteria):
    diff_logical = abs(np.subtract(new_bins, old_bins)) < centrality_criteria
    return (diff_logical).all()

# Function to propogate an object until its predicted reentry and storing the lat and long
def satellite_propogation(satellite_info, shouldRandomSample, ts, tle_expiration):
  epoch_datetime = pd.to_datetime(satellite_info['epoch (UTC)'])
  reference_datetime = datetime.strptime("1949-12-31 00:00:00.0", "%Y-%m-%d %H:%M:%S.%f")
  epoch_difference_days = (((epoch_datetime - reference_datetime).to_numpy()) / np.timedelta64(1, 'D')).astype(float)

  # Creating satellite object for propogator
  satrec = Satrec()
  satrec.sgp4init(
    WGS72,  # gravity model
    'i',  # 'a' = old AFSPC mode, 'i' = improved mode
    int(satellite_info['norad']),  # satnum: Satellite number
    epoch_difference_days,  # epoch: days since 1949 December 31 00:00 UT
    satellite_info['bstar'],  # bstar: drag coefficient (/earth radii)
    satellite_info['dn_o2'],  # ndot: ballistic coefficient (revs/day)
    satellite_info['ddn_o6'],  # nddot: second derivative of mean motion (revs/day^3)
    satellite_info['ecc'],  # ecco: eccentricity
    satellite_info['argp'] * (np.pi / 180),  # argpo: argument of perigee (radians)
    satellite_info['inc'] * (np.pi / 180),  # inclo: inclination (radians)
    satellite_info['M'] * (np.pi / 180),  # mo: mean anomaly (radians)
    satellite_info['n'] * (2 * np.pi) / (24 * 60),  # no_kozai: mean motion (radians/minute)
    satellite_info['raan'] * (np.pi / 180),  # nodeo: right ascension of ascending node (radians)
  )
  satellite = EarthSatellite.from_satrec(satrec, ts)
  # Calculating reentry time
  reentry_time = pd.to_datetime(satellite_info['Aerospace Reentry Prediction (UTC)'])
  error_bound = satellite_info['Aerospace Stated Uncertainty 25% Rule (+/- hrs)']
  if pd.isnull(reentry_time):
    reentry_time = pd.to_datetime(satellite_info['DECAY_EPOCH'])
    error_bound = satellite_info['Space Track Reentry Uncertainity (hr)']
    if pd.isnull(error_bound):
      error_bound = 24


  #Sampled Reentry Time
  if shouldRandomSample == True:
    if satellite_info['norad'] in prediction_window_cache.keys():
      bounds = prediction_window_cache[satellite_info['norad']]
      random_sample_time = random.uniform(-bounds[0] * 60, bounds[1] * 60)
      reentry_time = reentry_time + pd.Timedelta(minutes=random_sample_time)
    elif pd.isnull(error_bound) == False:
        lowerbound =min(error_bound, (reentry_time - epoch_datetime).delta *2.77778e-13)
        upperbound = min(error_bound, (epoch_datetime + pd.Timedelta(days=tle_expiration) - reentry_time).delta *2.77778e-13)
        random_sample_time = random.uniform(-lowerbound * 60, upperbound * 60)
        reentry_time = reentry_time + pd.Timedelta(minutes=random_sample_time)
        prediction_window_cache[satellite_info['norad']] = [lowerbound, upperbound]

  # Check that sampled time is within TLE expiration constraints
  time_difference = (reentry_time - epoch_datetime).delta *2.77778e-13 / 24 # days
  if abs(time_difference) > tle_expiration:
    print("WARNING: Sampled Reentry time is out of TLE expiration date")
    print("Norad: " + str(satellite_info['norad']))
    print("Epoch time: " + str(epoch_datetime))
    print("Reentry time: " + str(reentry_time))
    print("time difference:" + str(time_difference))
  t = ts.utc(reentry_time.year, reentry_time.month, reentry_time.day, reentry_time.hour, reentry_time.minute,
             reentry_time.second)

  geocentric = satellite.at(t)
  lat, lon = wgs84.latlon_of(geocentric)
  error_code = 0

  # Error Codes:
  # 0 : no errors
  # 1 : changed sample time
  # 2 : no propogation solution found

  if np.isnan(lat.degrees) or np.isnan(lon.degrees):

    # Try to find time where propogator returns sensical data
    sample_time_changed = True
    hasErroredOut = True
    maxIter = tle_expiration * 24
    count = 0
    error_code = 1

    while hasErroredOut and reentry_time >= epoch_datetime and count < maxIter:

      reentry_time = reentry_time - pd.Timedelta(hours=1)
      t = ts.utc(reentry_time.year, reentry_time.month, reentry_time.day,
                 reentry_time.hour, reentry_time.minute, reentry_time.second)
      geocentric = satellite.at(t)
      lat, lon = wgs84.latlon_of(geocentric)
      count = count + 1

      if not np.isnan(lat.degrees) and not np.isnan(lon.degrees):
        hasErroredOut = False

      if count >= maxIter or (count < maxIter and reentry_time < epoch_datetime):
        print("No solution found for satellite: " + str(satellite_info['norad']))
        error_code = 2

  return (lat, lon, reentry_time, error_code, geocentric)


def propogate_df(shouldRandomSample, df_reentry_tle_mass, ts, tle_expiration):
  lats = []
  longs = []
  geocentric_position_objects = []
  resampled_reentry_times = []
  no_prop_solution_errors = []
  count_of_SPG4_failed_propogations= 0
  for index, row in df_reentry_tle_mass.iterrows():
    norad_id = row['norad']
    propogation_output = satellite_propogation(row, shouldRandomSample, ts, tle_expiration)
    lats.append(propogation_output[0].degrees)
    longs.append(propogation_output[1].degrees)
    geocentric_position_objects.append(propogation_output[4])
    if propogation_output[3] == 1:
      resampled_reentry_times.append([norad_id, propogation_output[2]])
    if propogation_output[3] == 2:
      no_prop_solution_errors.append(norad_id)
  if len(lats) == len(df_reentry_tle_mass.index) and len(longs) == len(df_reentry_tle_mass.index):
    df_reentry_tle_mass['Predicted Reentry Latitude'] = lats
    df_reentry_tle_mass['Predicted Reentry Longitude'] = longs
    df_reentry_tle_mass['Geocentric Position Object'] = geocentric_position_objects
  df_reentry_tle_mass.dropna(subset=['Predicted Reentry Latitude', 'Predicted Reentry Longitude'], inplace=True)
  print("Reentries- after propogation - " + str(len(df_reentry_tle_mass)))
  return (df_reentry_tle_mass, {'no_prop_solution':no_prop_solution_errors, 'reampled reentry time':resampled_reentry_times})
