from skyfield.api import load
import netCDF4 as nc
import DataLoader
import Emissions
import Propogator
import Plotter
import pandas as pd
import numpy as np

def main():
  global tle_expiration
  global df_reentry_tle_mass
  global ts
  global rb_ablation_profile
  global satellite_ablation_profile

  #initializing global parameters
  tle_expiration = 5  # 5 days after tle epoch, tle is considered bad
  ts = load.timescale()
  rb_ablation_profile = DataLoader.load_rb_ablation_profile()
  satellite_ablation_profile=DataLoader.load_satellite_ablation_profile()

  #Creating combined dataset and propogating
  #df_reentry_tle_mass_data, df_reentry_mass, df_reentries, df_zero_mass_reentries= DataLoader.create_merged_data(tle_expiration)
  #df_aerospace_data = DataLoader.getAerospaceDataframe()

  df_reentry_tle_mass_data, df_reentry_mass, df_reentries = DataLoader.read_in_clean_datasets()

  # #Plot Reentry Mass since 1957
  #Plotter.plot_Aerospace_data_number_of_reentries_by_year_by_kind(df_aerospace_data)
  #Plotter.reentries_by_mass_per_year_per_kind(df_reentry_mass)

  #Plot reentries without mass estimates
  # Plotter.plot_histogram_reentries_with_no_mass(df_zero_mass_reentries)


  # #Compute Total Al Injection for each reentry object
  # title = "Anthropogenic Aluminum Mass Injection per Year: 1957-2021"
  df_reentry_al_mass = Emissions.calculate_total_al_injection_per_object(df_reentry_mass, satellite_ablation_profile,  rb_ablation_profile)
  # Plotter.plot_aluminium_per_year_per_kind(df_reentry_mass, title)
  # # #
  # # #Al injection over altitude
  altitude_range= np.arange(30,92, 2)
  space_debris_contributions= Emissions.track_altitude_injection(altitude_range, df_reentry_mass, satellite_ablation_profile, rb_ablation_profile, year=2021)
  Plotter.plot_mass_injection_over_alt(space_debris_contributions, altitude_range, year=2021)
  #
  #Plotter.plot_hist_space_debris_mass(df_reentry_mass)

  # Run Monte Carlo Simulation
  # monte_carlo_iterations = 10000
  # normalized_vis_data= Propogator.run_Monte_Carlo_reentry_locations(monte_carlo_iterations, df_reentry_tle_mass_data, ts, tle_expiration)
  # #normalized_vis_data = pd.read_csv("./normalized_reentry_location_data.csv")
  # print(len(normalized_vis_data['normalized_bins'] ))
  # Plotter.plot_histogram_bubble_over_world_map(normalized_vis_data, "Reentry Locations at 100km Across Equal Area Bins around Globe ")

if __name__ == "__main__":
  main()




