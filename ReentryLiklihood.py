from skyfield.api import load
import netCDF4 as nc
from Trajectory import Trajectory
import DataLoader
import Propogator
import matlab.engine
import Plotter
import pandas as pd
import numpy as np

def determine_if_centrality(old_bins, new_bins, centrality_criteria):
  new_bins['n_centerpoint_lons'] = new_bins['centerpoint_lons']
  new_bins['n_centerpoint_lats'] = new_bins['centerpoint_lats']
  new_bins['n_global_bins'] = new_bins['global_bins']
  new_bins.drop(['centerpoint_lons', 'centerpoint_lats', 'global_bins'], inplace=True)

  merged_bins = pd.merge(old_bins, new_bins, how='outer', left_on =['centerpoint_lons', 'centerpoint_lats'], right_on =['n_centerpoint_lons', 'n_centerpoint_lats'])
  diff_logical = abs(np.subtract(np.array(merged_bins['n_global_bins']), np.array(merged_bins['global_bins']))) < centrality_criteria
  return (diff_logical).all()

def merge_spatial_bins(original_bins, new_bins):
  merged_bins = pd.merge(original_bins, new_bins, how='outer', left_on =['centerpoint_lons', 'centerpoint_lats'], right_on =['n_centerpoint_lons', 'n_centerpoint_lats'])
  merged_bins['global_bins'] = merged_bins['global_bins'].values + merged_bins['n_global_bins'].values
  merged_bins.drop(['n_global_bins'], inplace = True)
  return merged_bins

def monte_carlo_reentry_location_liklihood(monte_carlo_iterations, binarea):
  # Monte Carlo to test the distribution of reentries over space between 2002-2021
  iteration = 1
  reentry_count = 0
  centrality_tolerance = 1e-5
  hasReachedCentrality = False
  bins= pd.DataFrame({'n_global_bins': 0, 'n_centerpoint_lons': 0, 'n_centerpoint_lats': 0})
  monte_carlo_df = df_reentry_tle_mass.copy()

  while iteration < monte_carlo_iterations and hasReachedCentrality == False:

    iteration = iteration + 1
    monte_carlo_df = df_reentry_tle_mass.copy()

    df_successfully_propogated, errors =Propogator.propogate_df(shouldRandomSample=False, df_reentry_tle_mass=monte_carlo_df, ts=ts, tle_expiration=tle_expiration)
    [latbin, lonbin, count] = eng.hista(df_successfully_propogated['Latitude'], df_successfully_propogated['Longitude'], binarea=binarea)
    new_bins = pd.DataFrame({'n_global_bins': count, 'n_centerpoint_lons': lonbin, 'n_centerpoint_lats': latbin})
    bins = merge_spatial_bins(bins, new_bins)

    reentry_count = reentry_count + len(df_successfully_propogated)
    bins['normalized_bins'] = np.divide(np.array(bins['global_bins']), reentry_count)

    if iteration ==1:
      previous_normalized_bins = bins
    if iteration % 10 == 0:
      hasReachedCentrality = determine_if_centrality(previous_normalized_bins, bins, centrality_tolerance)
      previous_normalized_bins = bins

  if (hasReachedCentrality == False):
    print("Warning: Centrality not reached to specified criteria.")
  else:
    print("Monte Carlo has reached centrality to specified criteria.")

  return bins

def main():
  global tle_expiration
  global df_reentry_tle_mass
  global trajectory_calculator
  global ts
  global rb_ablation_profile
  global satellite_ablation_profile
  global eng

  eng = matlab.engine.start_matlab()

  #initializing global parameters
  tle_expiration = 5  # 5 days after tle epoch, tle is considered bad
  trajectory_calculator = Trajectory()
  ts = load.timescale()
  rb_ablation_profile = DataLoader.load_rb_ablation_profile()
  satellite_ablation_profile=DataLoader.load_satellite_ablation_profile()

  #Creating combined dataset and propogating
  df_reentry_tle_mass = DataLoader.create_merged_data(tle_expiration)
  data, errors = Propogator.propogate_df(shouldRandomSample=False, df_reentry_tle_mass=df_reentry_tle_mass, ts=ts, tle_expiration=tle_expiration)
  normalized_reentry_liklihood = monte_carlo_reentry_location_liklihood(monte_carlo_iterations= 10000, binarea=100)
  Plotter.plot_histogram_bubble_over_world_map(normalized_reentry_liklihood, "Global Distribution of Reentries from 2002-2021")


if __name__ == "__main__":
  main()




