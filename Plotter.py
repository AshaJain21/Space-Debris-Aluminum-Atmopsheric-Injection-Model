import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits import mplot3d
import numpy as np
from mpl_toolkits.basemap import Basemap
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
import plotly.express as px
import plotly.graph_objects as go
import math

def plot_Aerospace_data_number_of_reentries_by_year_by_kind(df_aerospace):
    # plotting the number of predicted reentries per year by kind by mass
    rocket_body_reenties = (df_aerospace.loc[df_aerospace['Type-clean'].str.contains('R')].groupby(['Reentry Year']))['norad'].count()
    rocket_body_reenties = rocket_body_reenties.drop(2022, errors='ignore')
    satellite_mass = (df_aerospace.loc[df_aerospace['Type-clean'].str.contains('S')].groupby(['Reentry Year']))['norad'].count()
    satellite_reentries = satellite_mass.drop([2000,2022],errors='ignore')

    labels = (rocket_body_reenties.index).astype(int)
    width = 0.5  # the width of the bars: can also be len(x) sequence

    fig, ax = plt.subplots()
    fig.set_size_inches(25, 10.5)
    ax.bar(labels, satellite_reentries.values, width, label='Satellite and Fragmented Debris Reentries', tick_label=labels)
    ax.bar(labels, rocket_body_reenties.values, width, bottom=satellite_reentries.values, label='Rocket Body Reentries',
           tick_label=labels)

    ax.set_ylabel('Number of Reentries', fontsize=40)
    ax.set_xlabel('Year', fontsize=40)
    ax.set_title('Aerospace Corporation Center for Orbital \nand Reentry Debris Studies Database: 2000-2021', fontsize=40)

    ax.legend(fontsize =30)

    plt.xticks(rotation=315, fontsize=30)
    plt.yticks(fontsize=30)
    plt.savefig('./aerospace_cords_reentries_by_kind_per_year.png')
    plt.show()

def plot_mass_injection_over_alt(contributions, altitude_range):
    sat_al_contribution_tonnes = np.divide(contributions[0], 1000)
    rb_al_contribution_tonnes = np.divide(contributions[1], 1000)

    total_decadel_al_injection_from_debris = sum(sat_al_contribution_tonnes) + sum(rb_al_contribution_tonnes)
    print(total_decadel_al_injection_from_debris)

    labels = altitude_range
    width = 1  # the width of the bars: can also be len(x) sequence

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.bar(labels, sat_al_contribution_tonnes, width, label='Satellite Debris Reentries', tick_label=labels)
    ax.bar(labels, rb_al_contribution_tonnes, width, bottom=sat_al_contribution_tonnes, label='Rocket Body Reentries',
           tick_label=labels)
    # ax.bar(labels + width, sat_al_contribution_tonnes, width, label='Meteors', color ='black')
    ax.legend(prop={"size": 22})

    ax.set_xlabel('Altitude (km)', fontsize=15)
    ax.set_ylabel('Injected Aluminum Mass (metric tons)', fontsize=15)
    ax.set_title('Aluminum Mass Injection over Altitude by Space Debris Kind', fontsize=15, pad=20)
    ax.legend()
    plt.ylim(0,200)
    plt.rcParams.update({'legend.fontsize': 'xx-large'})
    plt.xticks(rotation=270, ha="right", fontsize=15)
    plt.yticks(fontsize=15)
    plt.legend(loc='upper right', fontsize=15)
    plt.savefig('./al_injection_over_altitude_allyears.png')
    plt.show()


def plot(x_values, y_values):
    plt.plot(x_values, y_values)


def extract_position(sublist_position, geocentric_positions):
    return [position[sublist_position] for position in geocentric_positions]


def plot_trajectory_3Dgrid(geocentric_positions):
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    x = extract_position(0, geocentric_positions)
    y = extract_position(1, geocentric_positions)
    z = extract_position(2, geocentric_positions)
    ax.plot3D(x, y, z, 'gray')
    ax.set_title("Trajectory")
    plt.show()


def plot_trajectory_overEarth(pos_lats: object, pos_lons: object, pos_alts: object) -> object:
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    # Define lower left, uperright lontitude and lattitude respectively
    extent = [-180, 180, -90, 90]
    # Create a basemap instance that draws the Earth layer
    bm = Basemap(llcrnrlon=extent[0], llcrnrlat=extent[2],
                 urcrnrlon=extent[1], urcrnrlat=extent[3],
                 projection='cyl', resolution='l', fix_aspect=False, ax=ax)
    # Add Basemap to the figure
    ax.add_collection3d(bm.drawcoastlines(linewidth=0.25))
    ax.add_collection3d(bm.drawcountries(linewidth=0.35))
    ax.view_init(azim=230, elev=50)
    ax.set_xlabel('Longitude (°E)', labelpad=20)
    ax.set_ylabel('Latitude (°N)', labelpad=20)
    ax.set_zlabel('Altitude (km)', labelpad=20)
    # Add meridian and parallel gridlines
    lon_step = 30
    lat_step = 30
    meridians = np.arange(extent[0], extent[1] + lon_step, lon_step)
    parallels = np.arange(extent[2], extent[3] + lat_step, lat_step)
    ax.set_yticks(parallels)
    ax.set_yticklabels(parallels)
    ax.set_xticks(meridians)
    ax.set_xticklabels(meridians)
    ax.set_zlim(0, 120)

    ax.plot(pos_lons, pos_lats, pos_alts)
    plt.show()


def compute_sphere(radius_km):
    theta = np.linspace(0, 2 * np.pi, 100)
    phi = np.linspace(0, np.pi, 100)
    x0 = radius_km * np.outer(np.cos(theta), np.sin(phi))
    y0 = radius_km * np.outer(np.sin(theta), np.sin(phi))
    z0 = radius_km * np.outer(np.ones(100), np.cos(phi))
    return (x0, y0, z0)


def plot_trajectory_overEarth_interactive(position_data, latitude_data, longitude_data):
    pos_x = [pos[0] / 1000 for pos in position_data.values]
    pos_y = [pos[1] / 1000 for pos in position_data.values]
    pos_z = [pos[2] / 1000 for pos in position_data.values]
    earth_radius = 6378  # km
    x0, y0, z0 = compute_sphere(earth_radius)
    # separate_position_data= pd.DataFrame({'x':pos_x, 'y':pos_y, 'z':pos_z})
    # fig = px.scatter_3d(separate_position_data, x='x', y='y', z='z')
    fig = go.Figure()
    # fig.add_trace(go.Scatter3d(
    #   x=pos_x,
    #   y=pos_y,
    #   z=pos_z)
    # )
    # fig.add_trace(go.Surface(
    #   x=x0,
    #   y=y0,
    #   z=z0)
    # )

    # if you are passing just one lat and lon, put it within "[]"
    fig.add_trace(go.Scattergeo(lat=latitude_data.values, lon=longitude_data.values))
    fig.update_geos(projection_type="orthographic")
    # this projection_type = 'orthographic is the projection which return 3d globe map'

    fig.show()


def plot_histogram_bubble_over_world_map(data, title):
    normalized_bins = data['normalized_bins']
    centerpoint_lons = data['Bin Long']
    centerpoint_lats = data['Bin Lat']

    fig = go.Figure()
    sizeref = 2. * np.max(normalized_bins) / (10 ** 1.3)
    colors = ["royalblue", "crimson", "lightseagreen", "orange", "lightgrey"]
    #max rbg
    red_r = 255
    red_g = 0
    red_b = 0
    #middle rbg
    orange_r = 255
    orange_g = 125
    orange_b = 0
    #low green rbg
    green_r = 0
    green_g=255
    green_b=0

    max_bin_count = np.max(normalized_bins)
    for index, row in data.iterrows():
        bin_count = row['normalized_bins']
        if bin_count > 0:
            p = (bin_count) / (max_bin_count)
            color_r = green_r + p * (orange_r - green_r)
            color_b = green_b + p * (orange_b - green_b)
            color_g = green_g + p * (orange_g - green_g)
            m_color = 'rgb(' + str(int(color_r)) + "," + str(int(color_g)) + "," + str(int(color_b)) + ")"
            # if bin_count > 0.33* max_bin_count:
            #     p = (bin_count - 0.33* max_bin_count) / (max_bin_count-0.33* max_bin_count)
            #     color_r = orange_r + p*(red_r - orange_r)
            #     color_b = orange_b + p*(red_b - orange_b)
            #     color_g = orange_g + p*(red_g - orange_g)
            #     m_color = 'rgb('+ str(int(color_r))+","+str(int(color_g)) +"," + str(int(color_b))+")"
            # else:
            #     p = (bin_count) / (0.33 * max_bin_count)
            #     color_r = green_r + p * (orange_r - green_r)
            #     color_b = green_b + p * (orange_b - green_b)
            #     color_g = green_g + p * (orange_g - green_g)
            #     m_color = 'rgb(' + str(int(color_r)) + "," + str(int(color_g)) + "," + str(int(color_b)) + ")"

            fig.add_trace(go.Scattergeo(
                lon=[row['Bin Long']],
                lat=[row['Bin Lat']],
                text='Normalized Number of Reentries: {0}'.format(bin_count),
                marker=dict(
                    size=bin_count / sizeref,
                    color=m_color,
                    line_color='rgb(40,40,40)',
                    line_width=0.5,
                    sizemode='area'
                ),
                name='Lat:{0} - Lon:{1}'.format(row['Bin Lat'], row['Bin Long'])))
    fig.update_geos(lataxis_showgrid=True, lonaxis_showgrid=True)
    fig.update_layout(
        title_text=title,
        showlegend=False,
        geo=dict(
            landcolor='rgb(217, 217, 217)',
        )
    )

    fig.update_geos(lataxis_showgrid=True, lonaxis_showgrid=True)
    fig.show()


# Determining mesh grid for Earth and counting reentry occurances in mesh
def compute_mesh_grid_and_bin_counts(mesh_grid_size, df_sucessfullypropogated):
    # Computing the grid tick marks for mesh grid square size. Works best if multiple of 5
    lon_ticks = np.linspace(-180, 180, int(360 / mesh_grid_size) + 1)
    lat_ticks = np.linspace(90, -90, int(180 / mesh_grid_size) + 1)
    global_bins = np.zeros((len(lat_ticks) - 1, len(lon_ticks) - 1))

    # Determining bin counts in each square of grid
    for index, satellite_row in df_sucessfullypropogated.iterrows():
        sat_lat = satellite_row.loc['Predicted Reentry Latitude']
        sat_lon = satellite_row.loc['Predicted Reentry Longitude']
        matrix_coordinates = find_square_index(sat_lon, sat_lat, lon_ticks, lat_ticks)
        global_bins[matrix_coordinates[0]][matrix_coordinates[1]] = global_bins[matrix_coordinates[0]][
                                                                        matrix_coordinates[1]] + 1

    # Computing the centerpoints of each grid square
    lon_center_offset = (abs(lon_ticks[0]) - abs(lon_ticks[1])) / 2
    lat_center_offset = (abs(lat_ticks[0]) - abs(lat_ticks[1])) / 2
    centerpoint_lons = (lon_ticks + lon_center_offset)[0:-1]
    centerpoint_lats = (lat_ticks - lat_center_offset)[0:-1]

    return {"global_bins": global_bins, "lon_center_offset": lon_center_offset, "lat_center_offset": lat_center_offset,
            "centerpoint_lats": centerpoint_lats, "centerpoint_lons": centerpoint_lons}


def find_square_index(sat_lon, sat_lat, lon_ticks, lat_ticks):
    sat_lat_upper_index = -1
    sat_lon_upper_index = -1
    for i in range(0, len(lat_ticks)):
        if (sat_lat > lat_ticks[i]):
            sat_lat_upper_index = i;
            break
    for j in range(0, len(lon_ticks)):
        if (sat_lon < lon_ticks[j]):
            sat_lon_upper_index = j;
            break
    if sat_lat_upper_index == -1 or sat_lon_upper_index == -1:
        print("Warning: invalid lat/lon index found")
        print(sat_lat)
        print(sat_lon)
    return (sat_lat_upper_index - 1, sat_lon_upper_index - 1)


def reentries_by_mass_per_year_per_kind(data):
    # plotting the number of predicted reentries per year by kind by mass
    rocket_body_reenties = (data.loc[data['Type'].str.contains('R')].groupby(['Reentry Year']))['DryMass'].sum()
    rocket_body_reenties = (rocket_body_reenties.drop(2022, errors='ignore')) / 1000 #metric tonnes
    satellite_mass = data.loc[data['Type'].str.contains('S')]
    satellite_reentries = ((satellite_mass.groupby('Reentry Year'))['DryMass'].sum().drop(2022,errors='ignore')) / 1000 #metric tonnes

    avg_reentry_mass= (sum(rocket_body_reenties) + sum(satellite_reentries)) / (len(rocket_body_reenties))
    print("Average year reentry mass: " + str(avg_reentry_mass))
    start_year =min((rocket_body_reenties.index).astype(int))
    end_year = max((rocket_body_reenties.index).astype(int))

    labels = (rocket_body_reenties.index).astype(int)
    width = 0.5  # the width of the bars: can also be len(x) sequence

    fig, ax = plt.subplots()
    fig.set_size_inches(25, 10.5)
    plt.rcParams.update({'axes.titlesize': 'xx-large'})
    plt.rcParams.update({'axes.labelsize': 'xx-large'})
    plt.rcParams.update({'legend.fontsize': 'xx-large'})
    plt.rcParams.update({'font.size': 15})
    ax.bar(labels, satellite_reentries.values, width, label='Satellite Debris Reentries', tick_label=labels)
    ax.bar(labels, rocket_body_reenties.values, width, bottom=satellite_reentries.values, label='Rocket Body Reentries',
           tick_label=labels)

    ax.set_ylabel('Mass of Reentries (metric tons)', fontsize=25)
    ax.set_xlabel('Year', fontsize=25)
    ax.set_title('Historical Reentries per Year by Kind: ' + str(start_year) + "-" + str(end_year), fontsize=30)

    ax.legend()

    plt.xticks(rotation=270, fontsize = 20)
    plt.yticks(fontsize = 30)
    plt.savefig('./all_reentries_by_kind_per_year.png')
    plt.show()

def plot_aluminium_per_year_per_kind(data, title):
    rocket_body_reenties_per_year_mass = (data.loc[data['Type'].str.contains('R')].groupby(['Reentry Year']))
    rbr_al_mass = rocket_body_reenties_per_year_mass['Al Mass Contribution'].sum().drop(2022)

    satellite_mass = data.loc[data['Type'].str.contains("S")]
    satellite_al_mass = (satellite_mass.groupby('Reentry Year'))['Al Mass Contribution'].sum().drop(2022)

    space_debris_avg_al_injection = (sum(rbr_al_mass) + sum(satellite_al_mass)) / len(satellite_al_mass.index) / 1000

    total_decadal_al_mass_from_debris = (sum(rbr_al_mass) + sum(satellite_al_mass)) / 1000
    start_year = min(data['Reentry Year'])
    end_year = max(data['Reentry Year'])
    print("Total Reentered Al Mass: " + str(total_decadal_al_mass_from_debris) + " from years " + str(start_year) + "-" + str(end_year))

    labels = (satellite_al_mass.index).astype(int)
    width = 0.5  # the width of the bars

    fig, ax = plt.subplots()
    fig.set_size_inches(25, 7)
    ax.bar(labels, satellite_al_mass.values / 1000, width, label='Satellite Debris', tick_label=labels)
    ax.bar(labels, rbr_al_mass.values / 1000, width, bottom=satellite_al_mass.values / 1000, label='Rocket Body',
           tick_label=labels)

    ax.set_ylabel('Aluminum Mass Injected\n(metric tons)', fontsize=20)
    # ax.set_title(title, fontsize=35, pad=20)
    ax.legend()

    daily_meteor_influx = 30  # tonnes
    meteoric_al_composition = 0.017
    meteoric_al_ablation = 0.14
    meteoric_injection = daily_meteor_influx * 365 * meteoric_al_composition * meteoric_al_ablation

    meteoric_label = 'Meteoric Al Injected Mass: ' + str(round(meteoric_injection)) + " metric tons"
    x = np.linspace(min(satellite_al_mass.index), max(satellite_al_mass.index), 10)
    y = np.ones(10) * meteoric_injection
    plt.plot(x, y, color='k', linewidth=5, label=meteoric_label)

    avg_space_debris_label = 'Average Debris Al Injected Mass: ' + str(
        round(space_debris_avg_al_injection)) + " metric tons"
    x = np.linspace(min(satellite_al_mass.index), max(satellite_al_mass.index), 10)
    y = np.ones(10) * space_debris_avg_al_injection
    plt.plot(x, y, '--', linewidth=5, color='g', label=avg_space_debris_label)
    plt.ylim(0, 110)
    plt.xticks(rotation=270, fontsize=20)
    plt.yticks(fontsize = 20)
    plt.xlabel("Year", fontsize=15)
    plt.legend(loc='upper right', fontsize=15)
    plt.savefig('./total_al_influx_per_year.png')
    plt.show()

def plot_hist_space_debris_mass(data):
    masses = data['DryMass'] /1000
    (n, bins, patches) = plt.hist(masses, bins=20, rwidth=0.25)
    plt.xlabel("Space Debris Mass (metric tons)", fontsize=15)
    plt.ylabel("Number of Reentries", fontsize=15)
    plt.yscale("log")
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.title("Mass Distribution of Reentries", fontsize=15)

    peak = bins[np.where(n == np.amax(n))]
    print("Peak mass in: " + str(peak))
    # plt.show()
    fig = plt.gcf()
    fig.set_size_inches(12.5, 10.5)
    plt.rcParams.update({'font.size': 20})
    plt.rc('xtick', labelsize=20)
    plt.rc('ytick', labelsize=20)
    plt.savefig('./all_reentries_mass_distribution.png', dpi=100)
    plt.show()

def plot_histogram_TLE_inclinations(data):
    inclinations = data['inc']
    (n, bins, patches) = plt.hist(inclinations, bins=20, rwidth=0.25)
    plt.xlabel("Reentry TLE Inclinations (degrees)", fontsize=25)
    plt.ylabel("Number of Reentries", fontsize=25)
    plt.yscale("log")
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    plt.title("Distribution of Reentries TLE Inclinations", fontsize=30)

    # plt.show()
    fig = plt.gcf()
    fig.set_size_inches(10.5, 8.5)
    plt.savefig('./distribution_inclinations_tle', dpi=100)
    plt.show()

def plot_histogram_TLE_eccentricity(data):
    eccentricities = data['ecc']
    (n, bins, patches) = plt.hist(eccentricities, bins=20, rwidth=0.25)
    plt.xlabel("Reentry TLE Eccentricities (degrees)", fontsize=25)
    plt.ylabel("Number of Reentries", fontsize=25)
    plt.yscale("log")
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    plt.title("Distribution of Reentries TLE Eccentricities", fontsize=30)

    # plt.show()
    fig = plt.gcf()
    fig.set_size_inches(10.5, 8.5)
    plt.savefig('./distriution_eccentricities_tle.png', dpi=100)
    plt.show()

def plot_histogram_reentries_with_no_mass(data):
    year_counts = data.groupby('Reentry Year')['norad'].count()
    labels = (year_counts.index).astype(int)
    plt.bar(year_counts.index, year_counts, width=0.4)
    plt.xlabel("Reentry Year", fontsize=25)
    plt.ylabel("Number of Reentries without mass estimates", fontsize=25)
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    plt.title("Distribution of Reentries without Mass Estimates", fontsize=30)

    # plt.show()
    fig = plt.gcf()
    fig.set_size_inches(10.5, 8.5)
    plt.savefig('./distriution_no_mass_estimates.png', dpi=100)
    plt.show()