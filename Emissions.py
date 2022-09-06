import numpy as np
import pandas as pd
import Plotter
from skyfield.api import wgs84
from skyfield.toposlib import ITRSPosition
from skyfield import positionlib as poslib
from skyfield import constants as constants
import re

def determine_Al_mass(reentry_object):
  # From SMAD, structure and mechanisms comprise, on average, 24% of dry mass. Pg 948 SMAD
  # Al Injected Mass per Year Calculation Assumes:
  # 1. that all of the structure and mechanism satellite mass is Aluminium, rho = 0.24 of dry mass
  # 3. rocket body frame is aluminium and structural mass minus engine mass is the aluminum mass
  # 4. rocket body engine is alpha = 470kg / (4000-470) kg = 0.13 engine to structure ratio (from Merlin 1D Dry Mass and Falcon 9 Upper Stage)

  alpha = 0.7
  rho = 0.21
  if reentry_object['Type'] == 'R':
    al = reentry_object['DryMass'] *  alpha
    search_for_centaur = reentry_object['Discos Object Name'].__contains__("CENTAUR")
    if search_for_centaur == True:
      search_for_centaur = not (reentry_object['Discos Object Name'].__contains__("PAYLOAD FAIRING"))
    search_for_delta = reentry_object['Discos Object Name'].__contains__("DELTA II")
    search_for_antares = reentry_object['Discos Object Name'].__contains__("ANTARES")
    if search_for_delta == True or search_for_centaur == True or search_for_antares == True:
      al = reentry_object['DryMass'] * 0.05
      #print("Ignoring: " + reentry_object['Discos Object Name'] + " : " + str(reentry_object['DryMass']))
  else:
    al = reentry_object['DryMass'] * rho
  return al


def update_geocentric_object(geocentric_object):
  position = wgs84.subpoint_of(geocentric_object)
  altitude_km = wgs84.height_of(geocentric_object).km
  new_geocentric_object = geocentric_object
  if altitude_km > 120:
    altitude_km = 120
    earth_position = wgs84.latlon(position.latitude.degrees, position.longitude.degrees, altitude_km * 1000)
    updated_gcrs_position = ITRSPosition(earth_position.itrs_xyz).at(geocentric_object.t)
    new_geocentric_object = poslib.build_position(updated_gcrs_position.position.km / constants.AU_KM,geocentric_object.velocity.km_per_s / constants.AU_KM * constants.DAY_S,t=geocentric_object.t, center=399)
  return new_geocentric_object

def calculate_reentry_trajectory(reentry_object, trajectory_calculator, ts):
  # Euler method to solve equations of motion assuming:
  # 1. Spherical, non-rotating Earth
  # 2. Constant Ballasitc Coefficient
  # 3. No initial accerlation of reentry object
  # 4. Cd is constant and equals 1.2
  # 5. Assuming all objects have a circular cross sectional area. Assumed a default diameter for rocket bodies and for satellites

  #Output File to save results
  output_filename = "./Trajectory Logging/" + str(reentry_object['norad']) + "_trajectory_output.txt"
  output_file = open(output_filename, "w")

  #Generating Parameters
  Cd = 1.2
  if  reentry_object['Diameter'] > 0:
    S = reentry_object['Diameter']**2 * (np.pi/4)
  else:
    if(reentry_object['Type']=='S'):
      uniform_density = 1.33/10e3 #assumes homogenous density in 1U cubesat weighing 1.33kg, cite nasa cubesat overview https://www.nasa.gov/mission_pages/cubesats/overview
    else:
      uniform_density = 2350 / (np.pi / 4 * 2.7**2 * 6.7) #assumes homogenous density in SL-4 Rocket Body (a kind representing the majority by mass distribution)
    volume = reentry_object['DryMass'] * uniform_density
    S = volume ** (2 / 3) * (np.pi / 4)

  B = reentry_object['DryMass'] / (Cd * S)
  dt = 0.1  # dt must be in terms of seconds
  maxIters = 100000 # max iters must be longer than 5 mins
  end_altitude = 20  # km
  mu = 3.986004418e14 #m3s-2

  #Force initial position to be at 120 km altitude
  geocentric_obj = update_geocentric_object(reentry_object['Geocentric Position Object'])
  params = {'R': 6378e3, 'L/D': 0, 'g0': 9.81, 'B': B, 'GeocentricObject':geocentric_obj, 'dt':dt, 'max_iters':maxIters, 'end_altitude':end_altitude, 'S':S, 'Cd':Cd, 'mu':mu, 'outputfile':output_file, 'ts':ts}

  #Generating initial states
  initial_position =  np.array(geocentric_obj.position.km *1000) #meters
  initial_velocity = np.array(geocentric_obj.velocity.km_per_s * 1000) # m/s
  # initial_height = wgs84.height_of(geocentric_obj).km * 1000 # m
  # initial_flight_path_angle = 0 #assumed fpa since uncontrolled reentry
  initial_density = 0
  # initial_gravity = 9.81 * (params['R'] / (params['R']+initial_height)) ** 2
  initial_states = [initial_position, initial_velocity, initial_density ]
  trajectory  = trajectory_calculator.forward_euler(initial_states, params)

  trajectory_data = pd.DataFrame(trajectory, columns = ['position','velocity', 'density','altitude', 'latitude','longitude'])
  output_file.close()
  return trajectory_data

def determine_mass_fraction_loss_for_altitude_range(altitude_range, debris_type, satellite_ablation_profile, rb_ablation_profile):
  mass_fraction_loss_over_altitude_range = np.zeros(len(altitude_range))
  if debris_type in 'S':
    ablation_profile = satellite_ablation_profile.sort_values(by='Altitude')
  else:
    ablation_profile = rb_ablation_profile.sort_values(by='Altitude')

  for i in range(0, len(altitude_range)):
    altitude_index = np.argmax(altitude_range[i] < ablation_profile['Altitude'])
    lower_altitude = ablation_profile['Altitude'][altitude_index - 1]
    upper_altitude = ablation_profile['Altitude'][altitude_index]
    if abs(lower_altitude - altitude_range[i]) < abs(upper_altitude - altitude_range[i]):
      mass_fraction_loss = ablation_profile['Al Mass Loss Fraction'][altitude_index - 1]
    else:
      mass_fraction_loss = ablation_profile['Al Mass Loss Fraction'][altitude_index]

    mass_fraction_loss_over_altitude_range[i] = mass_fraction_loss
  return mass_fraction_loss_over_altitude_range

def compute_altitude_injection(row, satellite_mass_fraction_loss, rocketbody_mass_fraction_loss):
  al_composition = determine_Al_mass(row)
  if row['Type'] in 'S':
    al_contribution =  np.multiply(al_composition, satellite_mass_fraction_loss)
  else:
    al_contribution = np.multiply(al_composition, rocketbody_mass_fraction_loss)
  return al_contribution


def determine_emissions(df_reentry_tle_mass, trajectory_calculator, satellite_ablation_profile, rb_ablation_profile, ts):
  altitude_range = np.arange(20, 110, 1)
  satellite_mass_fraction_loss = determine_mass_fraction_loss_for_altitude_range(altitude_range, 'S', satellite_ablation_profile, rb_ablation_profile)
  rocketbody_mass_fraction_loss = determine_mass_fraction_loss_for_altitude_range(altitude_range, 'R', satellite_ablation_profile, rb_ablation_profile)

  for index, row in df_reentry_tle_mass.iterrows():
    # Compute trajectory
    trajectory = calculate_reentry_trajectory(row, trajectory_calculator, ts)
    #Plotter.plot_trajectory_3Dgrid(trajectory['position'])
    #Plotter.plot_trajectory_overEarth(trajectory['latitude'], trajectory['longitude'], trajectory['altitude'])
    Plotter.plot_trajectory_overEarth_interactive(trajectory['position'], trajectory['latitude'], trajectory['longitude'])
    print(row)
    #Compute al injection at each point on trajectory
    al_contribution = compute_altitude_injection(row, satellite_mass_fraction_loss, rocketbody_mass_fraction_loss)

    #Connect trajectory locations and aluminium contributions
    trajectory_slice = trajectory_calculator.slice_trajectory(altitude_range)
    trajectory_slice['Emission'] = al_contribution

    break
    # Save results in NET CDF format

def calculate_total_al_injection_per_object(df_reentry_mass, satellite_ablation_profile, rb_ablation_profile):
  sat_total_al_mass_fraction_loss = np.trapz(satellite_ablation_profile['Al Mass Loss Fraction'],
                                             x=satellite_ablation_profile['Altitude'])
  rb_total_al_mass_fraction_loss = np.trapz(rb_ablation_profile['Al Mass Loss Fraction'],
                                            x=rb_ablation_profile['Altitude'])
  al_mass = []
  for index, row in df_reentry_mass.iterrows():
    if row['Type'] == 'R':
      al = determine_Al_mass(row) * rb_total_al_mass_fraction_loss
    else:
      al = determine_Al_mass(row) * sat_total_al_mass_fraction_loss
    al_mass.append(al)

  df_reentry_mass['Al Mass Contribution'] = al_mass
  return df_reentry_mass

def determine_mass_fraction_loss_for_altitude_range(altitude_range, ablation_profile):
    mass_fraction_loss_over_altitude_range = np.zeros(len(altitude_range))

    for i in range(0, len(altitude_range) - 1):
      altitude_index = np.argmax(altitude_range[i] < ablation_profile['Altitude'])
      lower_altitude = ablation_profile['Altitude'][altitude_index - 1]
      upper_altitude = ablation_profile['Altitude'][altitude_index]
      if abs(lower_altitude - altitude_range[i]) < abs(upper_altitude - altitude_range[i]):
        mass_fraction_loss = ablation_profile['Al Mass Loss Fraction'][altitude_index - 1]
      else:
        mass_fraction_loss = ablation_profile['Al Mass Loss Fraction'][altitude_index]

      mass_fraction_loss_over_altitude_range[i] = mass_fraction_loss
    return mass_fraction_loss_over_altitude_range

def track_altitude_injection(altitude_range, df_reentry_mass, satellite_ablation_profile, rb_ablation_profile, year=None):
    df_track = df_reentry_mass.copy()
    satellite_al_contribution = np.zeros(len(altitude_range), dtype=np.dtype(float))
    rocket_body_al_contribution = np.zeros(len(altitude_range), dtype=np.dtype(float))

    if year is not None:
      df_track = df_reentry_mass.loc[df_reentry_mass['Year'] == year]

    satellite_mass_fraction_loss = determine_mass_fraction_loss_for_altitude_range(altitude_range, satellite_ablation_profile)
    rocketbody_mass_fraction_loss = determine_mass_fraction_loss_for_altitude_range(altitude_range, rb_ablation_profile)

    for index, row in df_track.iterrows():
      if row['Type'] in 'S':
        satellite_al_contribution = np.add(satellite_al_contribution,
                                           np.multiply(determine_Al_mass(row), satellite_mass_fraction_loss))
      else:
        rocket_body_al_contribution = np.add(rocket_body_al_contribution,
                                             np.multiply(determine_Al_mass(row), rocketbody_mass_fraction_loss))
    return (satellite_al_contribution, rocket_body_al_contribution)