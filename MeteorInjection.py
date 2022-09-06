import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math

def load_Plane_injection_profile():
  df_meteoric_injection_rate = pd.read_csv("plane21_al_meteoric_injection_rate.csv")
  df_meteoric_injection_rate = df_meteoric_injection_rate.sort_values(by='Altitude (km)').reset_index()
  return df_meteoric_injection_rate

def calculate_atmospheric_volume(dm, start_alt_km, end_alt_km):
  #Constants
  radius_earth = 6378 #km

  #Shell Radii (R is outer distance, and r is inner distance)
  atmosphere_lower_increment = np.arange(start_alt_km, end_alt_km+dm/2, dm)
  R = radius_earth + atmosphere_lower_increment + dm
  r = radius_earth + atmosphere_lower_increment

  volume_shell = (4/3)*math.pi * dm * (np.multiply(R,R) + np.multiply(R,r) + np.multiply(r,r)) #km^3
  return (volume_shell, atmosphere_lower_increment)

def compute_meteoric_injection_rate_over_altitude(atmosphere_increment):
  meteroic_injection_rates = np.zeros(len(atmosphere_increment))
  min_altitude_in_meteor_data = min(df_meteoric_injection_rate['Altitude (km)'].values)
  for i in range(0, len(atmosphere_increment) - 1):
    if atmosphere_increment[i] >= min_altitude_in_meteor_data:
      altitude_index = np.argmax(atmosphere_increment[i] > df_meteoric_injection_rate['Altitude (km)'])
      if altitude_index > 0:
        lower_altitude = df_meteoric_injection_rate['Altitude (km)'][altitude_index - 1]
        upper_altitude = df_meteoric_injection_rate['Altitude (km)'][altitude_index]

        if abs(lower_altitude - atmosphere_increment[i]) < abs(upper_altitude - atmosphere_increment[i]):
          meteroic_injection_rate = df_meteoric_injection_rate['Injection Rate (atom/cms)'][altitude_index - 1]
        else:
          meteroic_injection_rate = df_meteoric_injection_rate['Injection Rate (atom/cms)'][altitude_index]
      else:
        meteroic_injection_rate = df_meteoric_injection_rate['Injection Rate (atom/cms)'][altitude_index]

      meteroic_injection_rates[i] = meteroic_injection_rate

  return meteroic_injection_rates

def convert_concentration_rate_to_mass_rate(concentration_rate_over_altitude, volume_over_altitude):
  #Constants
  atomic_mass_to_kg = 1.66053906660e-27  # kg/d
  al_atomic_mass = 26.98  # daltons
  corrective_factor = 5  # see plane 2021 paper
  #Unit Conversion
  meteoric_al_mass_injected = np.multiply(volume_over_altitude,concentration_rate_over_altitude) * 84600 * 365 * al_atomic_mass * atomic_mass_to_kg * (1000 ** -1) * ((100000) ** 3) * corrective_factor
  return meteoric_al_mass_injected

def compute_meteoric_mass_injection_over_altitude(dm, start_alt_km, end_alt_km):
  fig, ax = plt.subplots(3)
  ax[0].plot(df_meteoric_injection_rate['Injection Rate (atom/cms)'], df_meteoric_injection_rate['Altitude (km)'])
  volume_shell_at_alt, atmosphere_increments = calculate_atmospheric_volume(dm, start_alt_km, end_alt_km)
  meteroic_injection_rates_at_alt = compute_meteoric_injection_rate_over_altitude(atmosphere_increments)
  ax[1].plot(meteroic_injection_rates_at_alt, atmosphere_increments)
  meteoric_al_mass_injected = convert_concentration_rate_to_mass_rate(meteroic_injection_rates_at_alt, volume_shell_at_alt)
  ax[2].plot(meteoric_al_mass_injected, atmosphere_increments)
  fig.show()
  total_meteoric_influx_year = np.trapz(meteoric_al_mass_injected, x=atmosphere_increments)
  print(total_meteoric_influx_year)
  return meteoric_al_mass_injected



def main():
  global df_meteoric_injection_rate
  df_meteoric_injection_rate = load_Plane_injection_profile()

  dm = 0.1
  start_alt_km = 0
  end_alt_km = 110
  meteoric_al_mass_injected= compute_meteoric_mass_injection_over_altitude(dm, start_alt_km, end_alt_km)


if __name__ == "__main__":
  main()
