from datetime import datetime

import pandas as pd
from tletools import pandas as tle_pd
import numpy as np
import DiscosWEB_Loader


# Loading datasets:
#   1. Aerospace Corp Predicted Reentries Database (excel file taken on Jan. 10 2022)
#   2. Mass Properties Dataset for Reentry Objects from planet4589.com
#   3. TLE Data taken from space-track.com on Jan. 10 2022
#   4. ESA SAM Aluminium Ablation Profiles for Representative Satellite and Rocket Body
#   5. Load DiscoWEB reentry dataset and DiscoWEB data for any other reentry bodies

def load_clean_reentry_data():
    df = pd.read_excel('./Reentry_History_Spreadsheet_01-07-22_modifiedDuplicates.xlsx')
    # Cleaning data
    df['Object Name'] = df['Object Name'].str.upper()
    # Formatting time string columns to datetimes
    df['Aerospace Prediction TLE Epoch (UTC)'] = pd.to_datetime(df['TLE Epoch (UTC)'])
    df['Aerospace Reentry Prediction (UTC)'] = pd.to_datetime(df['Aerospace Reentry Prediction (UTC)']).dt.tz_convert(
        None)

    df['Reentry Year'] = [d.astype('datetime64[Y]').astype(int) + 1970 for d in
                          df['Aerospace Reentry Prediction (UTC)'].values]
    df['norad'] = df['SSN']

    df.drop(['Type', 'SSN'], axis=1, inplace=True)

    # Checking for duplicates in SSN/norad
    df.drop_duplicates(subset=['norad'], inplace=True)

    return df


def load_clean_satcat_data():
    df_space_object_info = pd.read_csv("satcat.csv", encoding="utf-8", low_memory=False)
    df_space_object_info.replace('-', np.nan, inplace=True)

    # Refining Dataframe so that it only contains reentered space objects that burn in atmosphere
    df_space_object_info = df_space_object_info.loc[df_space_object_info['Status'].isin(['R', 'AR', 'D'])]

    df_space_object_info = df_space_object_info[df_space_object_info['Satcat'].notna()]
    df_space_object_info['norad'] = [int(d) for d in df_space_object_info['Satcat'].values]
    df_space_object_info['Mass'] = [float(d) for d in df_space_object_info['Mass'].values]
    df_space_object_info['DryMass'] = [float(d) for d in df_space_object_info['DryMass'].values]
    df_space_object_info['Diameter'] = [float(d) for d in df_space_object_info['Diameter'].values]

    df_space_object_info.drop(columns=['Satcat'], inplace=True)

    # Checking for duplicates in SSN/norad
    df_space_object_info.drop_duplicates(subset=['norad'], inplace=True)
    return df_space_object_info


def load_clean_TLE_data():
    # Load in Complete TLE Data taken from space-track.com on Jan. 10 2022
    df_tle = tle_pd.load_dataframe("tle_spacetrack.txt")
    # Formatting data
    df_tle['name'] = [s.strip("0 ") for s in df_tle['name']]
    df_tle['norad'] = [int(d) for d in df_tle['norad'].values]
    df_tle['epoch (UTC)'] = pd.to_datetime(df_tle['epoch'])
    df_tle.drop(columns=['epoch'], inplace=True)

    # Checking for duplicates in SSN/norad
    df_tle.drop_duplicates(subset=['norad'], inplace=True)

    return df_tle


def load_satellite_ablation_profile():
    df_satellite_ablation_profile = pd.read_csv("esa_satellite_ablation_profile.csv")
    return df_satellite_ablation_profile


def load_rb_ablation_profile():
    df_rb_ablation_profile = pd.read_csv("esa_rocket_body_ablation_profile.csv")
    return df_rb_ablation_profile

def load_discoWeb_data():
    df_discosweb = pd.read_excel('./discoweb_reentries.xlsx')
    df_discosweb['discos_id'] = df_discosweb['id']
    df_discosweb['norad'] = df_discosweb['attributes.satno']
    df_discosweb['DryMass_disco'] = df_discosweb['attributes.mass']
    df_discosweb['CrossSection_disco'] = df_discosweb['attributes.xSectAvg']

    df_discosweb.drop(columns=['type','attributes.vimpelId', 'links.self', 'id', 'attributes.satno', 'attributes.mass', 'attributes.xSectMin', 'attributes.xSectAvg', 'attributes.xSectMax'], inplace = True)
    df_discosweb['Reentry Data URL'] = df_discosweb['relationships.reentry.links.related']
    df_discosweb['Reentry Status'] = df_discosweb['relationships.reentry.data.type']
    df_discosweb = df_discosweb[df_discosweb.columns.drop(list(df_discosweb.filter(regex='relationship.*')))]

    df_discosweb = df_discosweb.loc[df_discosweb['Reentry Status'].isna() == False]

    return df_discosweb

def load_space_track_reentries():
    df_space_track_reentries = pd.read_csv("space_track_reentries.csv", encoding="utf-8", low_memory=False)
    df_space_track_reentries['DECAY_EPOCH'] = pd.to_datetime(df_space_track_reentries['DECAY_EPOCH'])
    df_space_track_reentries['norad'] = df_space_track_reentries['NORAD_CAT_ID']
    df_space_track_reentries.drop(columns=['NORAD_CAT_ID'], inplace = True)
    df_space_track_reentries['Space Track Reentry Uncertainity (hr)']= np.nan

    df_tip_message = pd.read_csv("space-track-tip.csv", encoding="utf-8", low_memory=False)

    #need to remove duplicates, saving the latest tip message
    df_duplicates = df_space_track_reentries.loc[df_space_track_reentries.duplicated(subset = ['norad'])]
    duplicated_ids = df_duplicates['norad'].unique()

    for id in duplicated_ids:
        subset = df_space_track_reentries.loc[df_space_track_reentries['norad'] == id]
        subset_tip_message = subset.loc[subset['SOURCE'] == 'TIP_msg']
        if len(subset_tip_message) > 0:
            most_recent_tip_prediction = max(subset_tip_message['MSG_EPOCH'])
            save_row_index = subset_tip_message.loc[subset_tip_message['MSG_EPOCH'] == most_recent_tip_prediction].index
            indices = subset.index.tolist()
            indices.remove(save_row_index)
            save_row_index = save_row_index.tolist()
            df_space_track_reentries.drop(indices, inplace = True)
            tip_data= (df_tip_message.loc[df_tip_message['NORAD_CAT_ID']==id]).loc[df_tip_message['MSG_EPOCH'] == most_recent_tip_prediction]
            range = tip_data['WINDOW'].values[0]
            df_space_track_reentries.loc[save_row_index[0], 'Space Track Reentry Uncertainity (hr)'] = float(range)

        else:
            sat_cat_prediction = subset.loc[subset['SOURCE']=='satcat']
            day60_prediction_set = subset.loc[subset['SOURCE']=='60day_msg']

            if len(sat_cat_prediction) > 0:
                save_row_index =  sat_cat_prediction.index
            else:
                latest_60day_prediction = max(day60_prediction_set['MSG_EPOCH'])
                day60_prediction = subset.loc[subset['MSG_EPOCH'] == latest_60day_prediction]
                save_row_index = day60_prediction.index

            indices = subset.index.tolist()
            indices.remove(save_row_index)
            df_space_track_reentries.drop(indices, inplace = True)


    return df_space_track_reentries

def consildate_mass_estimates(df):
    df_satcat = load_clean_satcat_data()
    #Consildating Dry Mass
    mass = []
    for index, row in df.iterrows():
        drymass_preferred = row['DryMass_disco']
        alternative_drymass = df_satcat.loc[df_satcat['norad']==row['norad']]['DryMass'].values
        if np.isnan(drymass_preferred) == True:
            if len(alternative_drymass)>0:
                mass.append(alternative_drymass[0])
            else:
                mass.append(0)
        else:
            mass.append(drymass_preferred)

    df.drop(columns=['DryMass_disco'], inplace=True)
    df['DryMass'] = mass
    return df

def remove_shuttle_reentries(df):
    possible_shuttle_reentries = (df.loc[df['Discos Object Name'].str.contains('STS')==True]).loc[df['DryMass']>40000]
    colubmia_explosion_norad = 27647
    possible_shuttle_reentries = possible_shuttle_reentries[possible_shuttle_reentries['norad'] !=colubmia_explosion_norad ]
    df.drop(possible_shuttle_reentries.index, inplace=True)
    possible_buran_reentries = (df.loc[df['Discos Object Name'].str.contains('BURAN')==True]).loc[df['DryMass']>40000]
    df.drop(possible_buran_reentries.index, inplace=True)
    return df

def create_merged_data(tle_expiration):
    df_discoWeb = load_discoWeb_data()
    df_aerospace_predictions = load_clean_reentry_data()
    #df_satcat = load_clean_satcat_data()
    df_tle = load_clean_TLE_data()
    df_spacetrack = load_space_track_reentries()

    #Merge all four reentry datasets (ESA discos, Space Force spacetrack, Gunter satcat, and Aerospace Corp cords)
    #df_reentry = pd.merge(df_discoWeb, df_satcat, on='norad', how='inner')
    df_reentry = pd.merge(df_discoWeb, df_spacetrack, on='norad', how='inner')
    df_reentry = pd.merge(df_reentry, df_aerospace_predictions, on='norad', how='left')


    # Label reentries as rocket body or as satellite
    df_reentry['Type-clean'] = 'S'
    #df_reentry.Type = df_reentry.Type.fillna('')
    df_reentry['attributes.objectClass'] =  df_reentry['attributes.objectClass'].fillna('')

    #df_reentry.loc[df_reentry['Type'].str.contains('R'), 'Type-clean'] = 'R'
    df_reentry.loc[df_reentry['attributes.objectClass'].str.contains('Rocket'), 'Type-clean'] = 'R'
    #df_reentry['Detailed Type'] = df_reentry['Type']
    df_reentry['Type'] = df_reentry['Type-clean']
    df_reentry.drop('Type-clean', axis=1, inplace=True)

    #Merging Reentry Years
    missing_reentry_years = df_reentry[df_reentry['Reentry Year'].isna()]
    #missing_reentry_years['DDate'] = missing_reentry_years['DDate'].fillna('')
    missing_reentry_years['DECAY_EPOCH'] = missing_reentry_years['DECAY_EPOCH'].fillna('')
    for index, row in missing_reentry_years.iterrows():
        df_reentry.loc[index, 'Reentry Year'] = row['DECAY_EPOCH'].year

    #Merging Object Names
    df_reentry['Object_Name_Clean'] = df_reentry['OBJECT_NAME'].fillna('')
    missing_object_names = df_reentry[df_reentry['OBJECT_NAME'].isna()]
    missing_object_names['attributes.name'] =  missing_object_names['attributes.name'].fillna('')
    missing_object_names['OBJECT_NAME'] = missing_object_names['OBJECT_NAME'].fillna('')
    for index, row in missing_object_names.iterrows():
        df_reentry.loc[index, 'Object_Name_Clean'] = row['attributes.name']
    df_reentry['Object_Name'] = df_reentry['Object_Name_Clean'].str.upper()
    df_reentry['Discos Object Name'] = df_reentry['attributes.name'].str.upper()
    df_reentry.drop(columns=['attributes.name', 'OBJECT_NAME','Object_Name_Clean'], inplace = True)


    #Clean up columns - Dry Mass
    df_reentries = consildate_mass_estimates(df_reentry)
    #Remove Shuttle Reentries
    df_reentries = remove_shuttle_reentries(df_reentries)
    #Select Dry Masses that are greater than 0
    df_zero_mass_reentries = df_reentries[df_reentries['DryMass'] == 0]
    df_reentries_mass = df_reentries[df_reentries['DryMass'] > 0]


    print("Number of reentries w and wo mass estimates: " + str(len(df_reentries)))
    # Joining Merged Reentries Dataset and TLE dataset, removing rows without TLEs
    df_reentry_tle = pd.merge(df_reentries, df_tle, on='norad', how='inner')
    df_reentry_tle['Aerospace Reentry Prediction (UTC)'] = df_reentry_tle['Aerospace Reentry Prediction (UTC)'].fillna('')
    print("Number of reentries with TLEs: " + str(len(df_reentry_tle)))
    df_reentry_tle_mass = df_reentry_tle[df_reentry_tle['DryMass'] > 0]
    print("Number of reentries with TLEs and mass estimates: " + str(len(df_reentry_tle_mass)))

    # Checking that the TLE Epoch is within 5 days of the Aerospace reentry prediction date
    aerospace_data_rows =  df_reentry_tle_mass.loc[df_reentry_tle_mass['Aerospace Reentry Prediction (UTC)'].isna()==False]

    aerospace_data_rows['TLE Epoch and Reentry Prediction Difference (Days)'] = [x.total_seconds()/86400 for x in (aerospace_data_rows['Aerospace Reentry Prediction (UTC)'] - aerospace_data_rows['epoch (UTC)'])]
    print("Aerospace Predicitions: Are there any TLE's out of date compared to Aerospace predicted reentry time?")
    print((abs(aerospace_data_rows['TLE Epoch and Reentry Prediction Difference (Days)']) > tle_expiration).values.any())
    orignal_len = len(df_reentry_tle_mass)
    out_of_date_rows = aerospace_data_rows.loc[abs(aerospace_data_rows['TLE Epoch and Reentry Prediction Difference (Days)']) > tle_expiration]
    if len(out_of_date_rows) > 0:
        print("Aerospace Predicitions: Dropping reentry objects with out-of-date TLEs")
        df_reentry_tle_mass.drop(index=out_of_date_rows.index,inplace=True)
        print('Dropped ' + str(orignal_len-len(df_reentry_tle_mass)) + ' rows')

    #Checking that the TLE Epoch is within 5 days of the Space Track reentry prediction date or
    tip_data_rows = df_reentry_tle_mass.loc[df_reentry_tle_mass['DECAY_EPOCH'].isna() == False]
    tip_data_rows['TLE Epoch and Reentry Prediction Difference (Days)'] = [x.total_seconds()/86400 for x in (tip_data_rows['DECAY_EPOCH'] - tip_data_rows['epoch (UTC)'])]
    print("Are there any TLE's out of date compared to Space Track predicted reentry time?")
    orignal_len = len(df_reentry_tle_mass)
    out_of_date_rows = tip_data_rows.loc[abs(tip_data_rows['TLE Epoch and Reentry Prediction Difference (Days)']) > tle_expiration]
    if len(out_of_date_rows) >0:
        out_of_date_rows_without_aerospace_predictions =out_of_date_rows.loc[out_of_date_rows['Aerospace Reentry Prediction (UTC)'].isna() == True]
        print("Dropping reentry objects with out-of-date TLEs")
        df_reentry_tle_mass.drop(index=out_of_date_rows_without_aerospace_predictions.index,inplace=True)
        print('Dropped ' + str(orignal_len-len(df_reentry_tle_mass)) + ' rows')

    # Checking for duplicates in SSN/norad
    df_duplicates = df_reentry_tle_mass.loc[df_reentry_tle_mass['norad'].duplicated() == True]
    if len(df_duplicates) > 0:
        print('ERROR- duplicates in reentry database unique ID: norad')
        print("Dropping duplicates")
        df_reentry_tle_mass.drop_duplicates(subset=['norad'])

    df_reentry_tle_mass.to_csv("./CleanData/reentry_tle_mass.csv")
    df_reentries_mass.to_csv("./CleanData/reentry_mass.csv")
    df_reentries.to_csv("./CleanData/reentries.csv")
    df_zero_mass_reentries.to_csv('./CleanData/no_mass_reentries.csv')

    print("Number of Propogatable Reentries - prior to propogation - " + str(len(df_reentry_tle_mass)))
    return (df_reentry_tle_mass, df_reentries_mass, df_reentries, df_zero_mass_reentries)

def getAerospaceDataframe():
    df = pd.read_excel('./Reentry_History_Spreadsheet_01-07-22_modifiedDuplicates.xlsx')
    # Cleaning data
    df['Object Name'] = df['Object Name'].str.upper()
    # Formatting time string columns to datetimes
    df['TLE Epoch (UTC)'] = pd.to_datetime(df['TLE Epoch (UTC)'])
    df['Aerospace Reentry Prediction (UTC)'] = pd.to_datetime(df['Aerospace Reentry Prediction (UTC)']).dt.tz_convert(
        None)
    df['Reentry Year'] = [d.astype('datetime64[Y]').astype(int) + 1970 for d in
                          df['Aerospace Reentry Prediction (UTC)'].values]
    df['norad'] = df['SSN']
    df['Type-clean'] = 'S'
    df.loc[df['Type'].str.contains('R/B'), 'Type-clean'] = 'R'
    df.drop('SSN', axis=1, inplace=True)
    return df

# def create_merged_data_2_out_of_3(tle_expiration):
#     df_discoWeb = load_discoWeb_data()
#     df_aerospace_predictions = load_clean_reentry_data()
#     df_satcat = load_clean_satcat_data()
#     df_tle = load_clean_TLE_data()
#     df_spacetrack = load_space_track_reentries()
#
#     #Merge all four reentry datasets (ESA discos, Space Force spacetrack, Gunter satcat, and Aerospace Corp cords)
#     df_reentry = pd.merge(df_discoWeb, df_satcat, on='norad', how='outer')
#     df_reentry = pd.merge(df_reentry, df_spacetrack, on='norad', how='outer')
#     df_reentry = pd.merge(df_reentry, df_aerospace_predictions, on='norad', how='outer')
#
#     # Label reentries as rocket body or as satellite
#     df_reentry['Type-clean'] = 'S'
#     df_reentry.Type = df_reentry.Type.fillna('')
#     df_reentry['attributes.objectClass'] =  df_reentry['attributes.objectClass'].fillna('')
#
#     df_reentry.loc[df_reentry['Type'].str.contains('R'), 'Type-clean'] = 'R'
#     df_reentry.loc[df_reentry['attributes.objectClass'].str.contains('Rocket'), 'Type-clean'] = 'R'
#     df_reentry['Detailed Type'] = df_reentry['Type']
#     df_reentry['Type'] = df_reentry['Type-clean']
#     df_reentry.drop('Type-clean', axis=1, inplace=True)
#
#     #Merging Reentry Years
#     missing_reentry_years = df_reentry[df_reentry['Reentry Year'].isna()]
#     missing_reentry_years['DDate'] = missing_reentry_years['DDate'].fillna('')
#     missing_reentry_years['DECAY_EPOCH'] = missing_reentry_years['DECAY_EPOCH'].fillna('')
#     for index, row in missing_reentry_years.iterrows():
#         if(row['DDate']):
#             df_reentry.loc[index, 'Reentry Year'] = int(row['DDate'][0:4] )
#         elif(row['DECAY_EPOCH']):
#             df_reentry.loc[index, 'Reentry Year'] = row['DECAY_EPOCH'].year
#
#     #Merging Object Names
#     df_reentry['Object_Name_Clean'] = df_reentry['OBJECT_NAME'].fillna('')
#     missing_object_names = df_reentry[df_reentry['OBJECT_NAME'].isna()]
#     missing_object_names['attributes.name'] =  missing_object_names['attributes.name'].fillna('')
#     missing_object_names['Name'] = missing_object_names['Name'].fillna('')
#     missing_object_names['OBJECT_NAME'] = missing_object_names['OBJECT_NAME'].fillna('')
#     for index, row in missing_object_names.iterrows():
#         if(row['attributes.name']):
#             df_reentry.loc[index, 'Object_Name_Clean'] = row['attributes.name']
#         elif(row['Name']):
#             df_reentry.loc[index, 'Object_Name_Clean'] = row['Name']
#     df_reentry['Object_Name'] = df_reentry['Object_Name_Clean']
#     df_reentry.drop(columns=['attributes.name', 'Name', 'OBJECT_NAME','Object_Name_Clean'], inplace = True)
#
#
#     #select ids that exist in at least two of the three datasets
#     df_merged_disco_satcat = pd.merge(df_discoWeb, df_satcat, on='norad', how='inner')
#     df_disco_uniques = pd.merge(df_discoWeb, df_merged_disco_satcat, on='norad', how='left', indicator=True)
#     df_disco_uniques = df_disco_uniques.loc[df_disco_uniques['_merge'] == 'left_only']
#     df_satcat_uniques = pd.merge(df_satcat, df_merged_disco_satcat, on='norad', how='left', indicator=True)
#     df_satcat_uniques = df_satcat_uniques.loc[df_satcat_uniques['_merge'] == 'left_only']
#     df_spacetrack_disco_overlap = pd.merge(df_spacetrack, df_disco_uniques, on='norad', how='inner')
#     df_spacetrack_satcat_overlap = pd.merge(df_spacetrack, df_satcat_uniques, on='norad', how='inner')
#     df_valid_ids = pd.merge(df_merged_disco_satcat, df_spacetrack_disco_overlap, on='norad', how='outer')
#     df_valid_ids = pd.merge(df_valid_ids, df_spacetrack_satcat_overlap, on='norad', how='outer')
#
#     valid_ids = df_valid_ids['norad']
#     df_reentry = df_reentry.loc[df_reentry['norad'].isin(valid_ids)]
#
#     #Clean up columns - Dry Mass
#     df_reentries = consildate_mass_estimates(df_reentry)
#     #Remove Shuttle Reentries
#     df_reentries = remove_shuttle_reentries(df_reentries)
#     #Select Dry Masses that are greater than 0
#     df_reentries_mass = df_reentries[df_reentries['DryMass'] > 0]
#
#
#     print("Number of reentries: " + str(len(df_reentries)))
#     # Joining Merged Reentries Dataset and TLE dataset, removing rows without TLEs
#     df_reentry_tle = pd.merge(df_reentries, df_tle, on='norad', how='inner')
#     print("Number of reentries with TLEs: " + str(len(df_reentry_tle)))
#     df_reentry_tle_mass = df_reentry_tle[df_reentry_tle['DryMass'] > 0]
#     print("Number of reentries with TLEs and mass estimates: " + str(len(df_reentry_tle_mass)))
#
#     # Checking that the TLE Epoch is within 5 days of the Aerospace reentry prediction date
#     aerospace_data_rows =  df_reentry_tle_mass.loc[df_reentry_tle_mass['Aerospace Reentry Prediction (UTC)'].isna()==False]
#
#     aerospace_data_rows['TLE Epoch and Reentry Prediction Difference (Days)'] = (
#         (aerospace_data_rows['Aerospace Reentry Prediction (UTC)'] - aerospace_data_rows['epoch (UTC)']).to_numpy()).astype(
#         'timedelta64[D]').astype(int)
#     print("Aerospace Predicitions: Are there any TLE's out of date compared to Aerospace predicted reentry time?")
#     print((abs(aerospace_data_rows['TLE Epoch and Reentry Prediction Difference (Days)']) > tle_expiration).values.any())
#     orignal_len = len(df_reentry_tle_mass)
#     if len(aerospace_data_rows[
#                abs(aerospace_data_rows['TLE Epoch and Reentry Prediction Difference (Days)']) > tle_expiration]) > 0:
#         print("Aerospace Predicitions: Dropping reentry objects with out-of-date TLEs")
#         df_reentry_tle_mass.drop(index=aerospace_data_rows[
#             abs(aerospace_data_rows['TLE Epoch and Reentry Prediction Difference (Days)']) > tle_expiration].index,
#                             inplace=True)
#         print('Dropped ' + str(orignal_len-len(df_reentry_tle_mass)) + ' rows')
#
#     #Checking that the TLE Epoch is within 5 days of the TIP reentry prediction date
#     tip_data_rows = df_reentry_tle_mass.loc[df_reentry_tle_mass['SOURCE'] == 'TIP_msg']
#     tip_data_rows['TLE Epoch and Reentry Prediction Difference (Days)'] = (
#         (tip_data_rows['DECAY_EPOCH'] - tip_data_rows['epoch (UTC)']).to_numpy()).astype(
#         'timedelta64[D]').astype(int)
#     print("Are there any TLE's out of date compared to Aerospace predicted reentry time?")
#     print((abs(aerospace_data_rows['TLE Epoch and Reentry Prediction Difference (Days)']) > tle_expiration).values.any())
#     orignal_len = len(df_reentry_tle_mass)
#     if len(tip_data_rows[abs(tip_data_rows['TLE Epoch and Reentry Prediction Difference (Days)']) > tle_expiration]) > 0:
#         print("Dropping reentry objects with out-of-date TLEs")
#         df_reentry_tle_mass.drop(index=tip_data_rows[
#             abs(tip_data_rows['TLE Epoch and Reentry Prediction Difference (Days)']) > tle_expiration].index,
#                             inplace=True)
#         print('Dropped ' + str(orignal_len-len(df_reentry_tle_mass)) + ' rows')
#
#     # Checking for duplicates in SSN/norad
#     df_duplicates = df_reentry_tle_mass.loc[df_reentry_tle_mass['norad'].duplicated() == True]
#     if len(df_duplicates) > 0:
#         print('ERROR- duplicates in reentry database unique ID: norad')
#         print("Dropping duplicates")
#         df_reentry_tle_mass.drop_duplicates(subset=['norad'])
#
#     # df_reentry_tle_mass.to_csv("./CleanData/reentry_tle_mass.csv")
#     # df_reentries_mass.to_csv("./CleanData/reentry_mass.csv")
#     # df_reentries.to_csv("./CleanData/reentries.csv")
#
#     print("Number of Propogatable Reentries - prior to propogation - " + str(len(df_reentry_tle_mass)))
#     return (df_reentry_tle_mass, df_reentries_mass, df_reentries)

# def create_merged_data_without_discos(tle_expiration):
#     df_reentry = load_clean_reentry_data()
#     df_mass = load_clean_satcat_data()
#     df_tle = load_clean_TLE_data()
#
#     # Joining Mass Properties and TLE dataset - expand the dataset to include reentered objects beyond Aerospace data
#     df_mass_tle = pd.merge(df_mass, df_tle, on='norad', how='left')
#     df_mass_tle = df_mass_tle[df_mass_tle['DryMass'] > 0]
#
#     # Joining TLE/Mass dataframe and Aerospace Reentry Predictions dataframe
#     df_reentry_tle_mass = pd.merge(df_mass_tle, df_reentry, on='norad', how='left')
#
#     # Note: unable to find a tle for SSN/Norad ID: 32259
#     df_reentry_tle_mass = df_reentry_tle_mass[df_reentry_tle_mass['norad'] != 32259]
#
#     # Checking that the TLE Epoch is within 5 days of the reentry date
#     aerospace_data_rows = df_reentry_tle_mass.loc[
#         df_reentry_tle_mass['Aerospace Reentry Prediction (UTC)'].isna() == False]
#
#     aerospace_data_rows['TLE Epoch and Reentry Prediction Difference (Days)'] = (
#         (aerospace_data_rows['Aerospace Reentry Prediction (UTC)'] - aerospace_data_rows[
#             'epoch (UTC)']).to_numpy()).astype(
#         'timedelta64[D]').astype(int)
#     print("Are there any TLE's out of date compared to Aerospace predicted reentry time?")
#     print(
#         (abs(aerospace_data_rows['TLE Epoch and Reentry Prediction Difference (Days)']) > tle_expiration).values.any())
#     orignal_len = len(df_reentry_tle_mass)
#     if len(aerospace_data_rows[
#                abs(aerospace_data_rows['TLE Epoch and Reentry Prediction Difference (Days)']) > tle_expiration]) > 0:
#         print("Dropping reentry objects with out-of-date TLEs")
#         df_reentry_tle_mass.drop(index=aerospace_data_rows[
#             abs(aerospace_data_rows['TLE Epoch and Reentry Prediction Difference (Days)']) > tle_expiration].index,
#                                  inplace=True)
#         print('Dropped ' + str(orignal_len - len(df_reentry_tle_mass)) + ' rows')
#
#     # Label reentries as rocket body or as satellite
#     df_reentry_tle_mass['Type-clean'] = 'S'
#     df_reentry_tle_mass.loc[df_reentry_tle_mass['Type'].str.contains('R'), 'Type-clean'] = 'R'
#     df_reentry_tle_mass['Detailed Type'] = df_reentry_tle_mass['Type']
#     df_reentry_tle_mass['Type'] = df_reentry_tle_mass['Type-clean']
#     df_reentry_tle_mass.drop('Type-clean', axis=1, inplace=True)
#
#     # Merging Reentry Years
#     missing_reentry_years = df_reentry_tle_mass[df_reentry_tle_mass['Reentry Year'].isna()]
#     df_reentry_tle_mass.loc[missing_reentry_years.index, 'Reentry Year'] = [int(d[0:4]) for d in
#                                                                             missing_reentry_years['DDate']]
#
#     # Checking for duplicates in SSN/norad
#     df_duplicates = df_reentry_tle_mass.loc[df_reentry_tle_mass['norad'].duplicated() == True]
#     if len(df_duplicates) > 0:
#         print('ERROR- duplicates in reentry database unique ID: norad')
#         print("Dropping duplicates")
#         df_reentry_tle_mass.drop_duplicates(subset=['norad'])
#
#     # Checking if any reentry objects did not have a matching tle
#     print("Missing TLEs:" + str(sum(df_reentry_tle_mass['raan'].isnull().values)))
#     df_missing_tle = df_reentry_tle_mass.loc[df_reentry_tle_mass['raan'].isna() == True]
#     df_reentry_mass = df_reentry_tle_mass.copy()
#     print("Dropping rows without TLE information: " + str(len(df_missing_tle)) + " rows")
#     df_reentry_tle_mass.drop(index=df_missing_tle.index, inplace=True)
#
#     print("Number of Propogatable Reentries - prior to propogation - " + str(len(df_reentry_tle_mass)))
#     # df_reentry_tle_mass.to_csv("./PreviousCleanData/reentry_tle_mass.csv")
#     # df_reentry_mass.to_csv("./PreviousCleanData/reentry_mass.csv")
#
#     return (df_reentry_tle_mass, df_reentry_mass)

def read_in_clean_datasets():
    df_reentry_tle_mass = pd.read_csv("./CleanData/reentry_tle_mass.csv")
    df_reentries_mass = pd.read_csv("./CleanData/reentry_mass.csv")
    df_reentries = pd.read_csv("./CleanData/reentries.csv")
    return (df_reentry_tle_mass, df_reentries_mass, df_reentries)