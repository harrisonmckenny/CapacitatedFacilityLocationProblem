# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 14:15:08 2023

@author: Harrison McKenny
"""


import numpy as np
import pandas as pd
import pysal as ps
import geopandas as gpd
from sklearn import cluster
from sklearn.preprocessing import scale
from math import sin, cos, asin, acos, radians
from pulp import LpProblem, LpMinimize, LpVariable, LpBinary, lpSum, LpStatus, value
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import csv
import re
import folium



# Load dataframe


 df_ms =pd.read_csv(
'./Alfalfa_Locations_Finished.csv')
 
 #'Tulare','Kern''
 
 county_small_list =['Tulare','Kern']
 
# Use the .isin() method to filter the DataFrame
 df_ms =  df_ms[ df_ms['county_name'].isin( county_small_list )].reset_index(drop=True)
 
#--------------------------------------------------------------------

#OK We are Inverting the Alogrithm here..... Demand is now Supply and Vice Versa
#-------------------------------MS DEMAND------------------------------------
#Demand is composed of:
# 1. The Total Population of Dairies in CA
# 2. Respective Herd Size and their feed needs for grain, silage and hay 
# We will be inverting the Demand/Supply side of this approach to treat dairies as depots

RANDOM_STATE_ms = 2 # For reproducibility
FRACTION_CUSTOMERS_ms = 1 # Fraction of parcels we want to keep as "customers" **Inverted'**
FRACTION_WAREHOUSES_ms = 1 # Fraction of dairies location from previous work we want to keep
FRACTION_DEMAND_ms = 1 # Fraction of Parcels across the state that may order a product

df_ms['demand'] = df_ms['Total Alfalfa Lbs']*FRACTION_DEMAND_ms
#----------------------------------------------------------------------------

facility_df = df.\
loc[df.admin_name.isin(REGION_LIST)].\
loc[df.capital.isin(['admin', 'minor'])].\
sample(frac=FRACTION_WAREHOUSES, random_state=RANDOM_STATE, ignore_index = True)


facility_df.head()
facility_df.columns.values

#--------------------------Already Determined Locations Here:------

facility_df_ms = pd.read_csv(
'./All_Dairy_Info_Feed.csv')

facility_df_ms.NAME.head()

# list(data) or
list(facility_df_ms.columns)



# Use the .isin() method to filter the DataFrame
facility_df_ms =  facility_df_ms[ facility_df_ms['NAME'].isin( county_small_list )].reset_index(drop=True)


#Dropping the ID column
#facility_df_ms.drop('ID', axis=1, inplace=True)

#Dropping the unnamed column
#facility_df_ms.drop('Unnamed: 0', axis=1, inplace=True)

#------------------------------------------------------------------

#----------------------------"Customer Locations" are crop growing parcels-----------------------

customer_df_ms = df_ms

customer_df_ms.head()

# list(data) or
list(customer_df_ms.columns)

# Customers IDs list
customer_df_ms['customer_id'] = range(1, 1 + customer_df_ms.shape[0])


#-----------------Simplifying Naming Convention-------

customer_df_ms['lng'] = customer_df_ms['lon']

customer_df_ms['city'] = customer_df_ms['parcel_city']

facility_df_ms['lat'] = facility_df_ms['Latitude']

facility_df_ms['lng'] = facility_df_ms['Longitude']
#-------------------------------------------------------------------------------------


def add_geocoordinates(df, lat='lat', lng='lng'):
    '''
    Add column "geometry" with <shapely.geometry.point.Point> objects 
        built from latitude and longitude values in the input dataframe
    
    Args:
        - df: input dataframe
        - lat: name of the column containing the latitude (default: lat)
        - lng: name of the column containing the longitude (default: lng)
    Out:
        - df: same dataframe enriched with a geo-coordinate column
    '''
    assert pd.Series([lat, lng]).isin(df.columns).all(),\
        f'Cannot find columns "{lat}" and/or "{lng}" in the input dataframe.'
    return gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df.lng, df.lat))


#----------------------------Geospatially Creating Points------------------

customer_df_ms = add_geocoordinates(customer_df_ms)
facility_df_ms = add_geocoordinates(facility_df_ms)

#----------------------------------------------------------------------------

#-----------------------------Mapping Counties of CA (Our Base Map)------
counties = gpd.read_file('counties.shp')
counties.head()
counties.columns.values

counties.plot()

counties1  = counties.set_crs(epsg=4326,allow_override=True)

ax = counties1.plot(color='white', edgecolor='black', figsize=(10, 10))

# Plot customers as points
customer_df_ms.\
    plot(ax=ax, marker='X', color='red', markersize=30, alpha=0.5, label='Alfalfa Growing Parcel')

# Plot potential facility locations as points
facility_df_ms.\
    plot(ax=ax, marker='D', color='blue', markersize=30, alpha=0.5, label='Dairy')

# Add legend
plt.legend(facecolor='white', title='Location')

# Add title
plt.title('Customers (Alfalfa Parcels) and Depots (Dairies')

# Remove ticks from axis
plt.xticks([])
plt.yticks([])

# Show plot
plt.show()

customer_df_ms.columns.values

#---------------------------------------Regional Visualization County/ZIP/Region or whatever----------------

#Needs to be clusters mate. c'mon ya ejit think about it

df_ms.columns.values

region_df_ms = df_ms.\
             groupby(['county_name']).\
             agg({'lat': 'mean', 'lng': 'mean', 'demand': 'sum'}).\
             reset_index()
             
             
             # Add geo-coordinates
region_df_ms = add_geocoordinates(region_df_ms)

# Plot the shape of CA Counties
ax2 = counties1.plot(color='white', edgecolor='black', figsize=(10, 10))

# Plot county area colored based on demand
region_df_ms.\
    plot(ax=ax2, column='demand', marker='o', c='demand', cmap='plasma', markersize=2500, alpha=0.6)

# Add region 'center' as red dots
region_df_ms.\
    plot(ax=ax2, marker='o', c='red', markersize=25, alpha=0.8, label='Customer location')

# Add region name above the center
for i, row in region_df_ms.iterrows():
    plt.annotate(
        row.county_name, xy=(row.lng, row.lat+0.2), horizontalalignment='center')

# Add color bar with demand scale
plt.colorbar(ax2.get_children()[1], ax=ax2, label='Annual Demand', fraction=0.04, pad=0.04) 

# Add title
plt.title('Annual demand by region')

# Remove ticks from axis
plt.xticks([])
plt.yticks([])

# Show plot
plt.show()


#----------------------------------------------------------------------------

#--------------------------------------------Fixed Cost Assumptions-----------------------

#You must reset pandas index here or this step will not work***** Very Important****

# Dictionary of cutomer id (id) and demand (value)
demand_dict_ms = { customer : customer_df_ms['demand'][i] for i, customer in enumerate(customer_df_ms['customer_id']) }

# Assumptions: 
#    1. Each warehouse has an annual cost of 100,000.00 dollars: rent, electricity, ...
#    2. Each warehouse has a suppply capacity equal to its yearly feed demand "lbs per annum"
COST_PER_WAREHOUSE_ms = 100_000
SUPPLY_FACTOR_PER_WAREHOUSE_ms = 1

#--------------------------------Changed this from regional demand constraint to individual---------------

#SUPPLY_PER_WAREHOUSE_ms = region_df_ms.demand.mean() * SUPPLY_FACTOR_PER_WAREHOUSE_ms

SUPPLY_PER_WAREHOUSE_ms = facility_df_ms.Alfalfa_Req_Annum_lbs * SUPPLY_FACTOR_PER_WAREHOUSE_ms

list(facility_df_ms.columns)

type(SUPPLY_PER_WAREHOUSE_ms)
type(COST_PER_WAREHOUSE_ms)
type(SUPPLY_FACTOR_PER_WAREHOUSE_ms)
type(facility_df_ms.Alfalfa_Req_Annum_lbs)

SUPPLY_PER_WAREHOUSE_ms.head()

#apply the dtype attribute
#result = SUPPLY_PER_WAREHOUSE_ms.dtypes

#print("Output:")
#print(result)


#--------------------------------------------------------------------------------------------------------
# Warehouses list
facility_df_ms['warehouse_id'] = ['Warehouse ' + str(i) for i in range(1, 1 + facility_df_ms.shape[0])]

# Dictionary of warehouse id (id) and max supply (value)
annual_supply_dict_ms = { warehouse : SUPPLY_PER_WAREHOUSE_ms for warehouse in facility_df_ms['warehouse_id'] }

# Dictionary of warehouse id (id) and fixed costs (value)
annual_cost_dict_ms = { warehouse : COST_PER_WAREHOUSE_ms for warehouse in facility_df_ms['warehouse_id'] }

type(SUPPLY_PER_WAREHOUSE_ms)

SUPPLY_PER_WAREHOUSE_ms.head()

#-----------------------------------------------------------------------------------------

def haversine_distance(lat1, lon1, lat2, lon2):
    '''
    Calculate distance between two locations given latitude and longitude.

    Args:
       - lat1: latitude of the first location
       - lon1: longitude of the first location
       - lat2: latitude of the second location
       - lon2: longitude of the second location
    Out:
       - Distance in Km
    
    Ref: 
       - https://en.wikipedia.org/wiki/Haversine_formula
    '''
    return 6371.01 *\
            acos(sin(radians(lat1))*sin(radians(lat2)) +\
            cos(radians(lat1))*cos(radians(lat2))*cos(radians(lon1)-radians(lon2)))

#----------------------------------------------------------------------------------------------

def traveling_cost(distance_in_km):
    '''
    Return traveling cost in euros given a distance in Km.

    Args:
      - distance_in_km: travel distance in Km
    Out:
      - cost of the trip in euros
    '''
    return 0.36 * (distance_in_km * 0.621371)



# Dict to store the distances between all warehouses and customers
   
#---------------------------------MS Transportation Cost Evaluation-----------------------------
    
    
  # Dict to store the distances between all warehouses and customers
transport_costs_dict_ms = {}

# For each warehouse location
for i in range(0, facility_df_ms.shape[0]):
    
    # Dict to store the distances between the i-th warehouse and all customers
    warehouse_transport_costs_dict_ms = {}
    
    # For each customer location
    for j in range(0, customer_df_ms.shape[0]):
        
        # Distance in Km between warehouse i and customer j
        d =  haversine_distance(
            facility_df_ms.lat[i], facility_df_ms.lng[i], customer_df_ms.lat[j], customer_df_ms.lng[j])
        
        # Update costs for warehouse i
        warehouse_transport_costs_dict_ms.update({customer_df_ms.customer_id[j]: traveling_cost(d)})
    
    # Final dictionary with all costs for all warehouses
    transport_costs_dict_ms.update({facility_df_ms.warehouse_id[i]: warehouse_transport_costs_dict_ms})
      
    
#-------------Save this off Cache'd so that I don't have to re-run it if computer crashes------------


# open file for writing, "w" is writing
w = csv.writer(open("transport_cost_dictionary.csv", "w"))

# loop over dictionary keys and values
for key, val in transport_costs_dict_ms.items():

    # write every key and value to file
    w.writerow([key, val])
    
    
#-------------------------------------Lets do it in a text file too-------------------------------
# open file for writing
f = open("transport_cost_dictionary.txt","w")

# write file
f.write( str(transport_costs_dict_ms) )

# close file
f.close()

#-----------------------------------------------------------------------------------------------------   
    
#--------------------------------------- MS Linear Optimization---------------------------------

    # Define linear problem
lp_problem_ms = LpProblem('CFLP', LpMinimize)


# Variable: y_j (constraint: it is binary)
created_facility_ms = LpVariable.dicts(
    'Create_facility', facility_df_ms['warehouse_id'], 0, 1, LpBinary)

# Variable: x_ij
served_customer_ms = LpVariable.dicts(
    'Link', [(i,j) for i in customer_df_ms['customer_id'] for j in facility_df_ms['warehouse_id']], 0)


# Objective function 
objective_ms = lpSum(annual_cost_dict_ms[j]*created_facility_ms[j] for j in facility_df_ms['warehouse_id']) +\
            lpSum(transport_costs_dict_ms[j][i]*served_customer_ms[(i,j)] \
                  for j in facility_df_ms['warehouse_id'] for i in customer_df_ms['customer_id'])

lp_problem_ms += objective_ms

# Costraint: the demand must be met
for i in customer_df_ms['customer_id']:
    lp_problem_ms += lpSum(served_customer_ms[(i,j)] for j in facility_df_ms['warehouse_id']) == demand_dict_ms[i]

# Constraint: a warehouse cannot deliver more than its capacity limit
for j in facility_df_ms['warehouse_id']:
    lp_problem_ms += lpSum(served_customer_ms[(i,j)] for i in customer_df_ms['customer_id']) <= annual_supply_dict_ms[j] * created_facility_ms[j]

# Constraint: a warehouse cannot give a customer more than its demand
for i in customer_df_ms['customer_id']:
    for j in facility_df_ms['warehouse_id']:
        lp_problem_ms += served_customer_ms[(i,j)] <= demand_dict_ms[i] * created_facility_ms[j]
    
    
lp_problem_ms.solve()


print('Solution: ', LpStatus[lp_problem_ms.status])    


value(lp_problem_ms.objective)

#-----------------------------------Diagnostics For Why It Isn't Working if needed--------------

#PuLP.pulpTestAll()


#-------------------------------------------------------------------------------------------


def get_linked_customers_ms(input_warehouse):
    '''
    Find customer ids that are served by the input warehouse.
    
    Args:
        - input_warehouse: string (example: <Warehouse 21>)
    Out:
        - List of customers ids connected to the warehouse
    '''
    # Initialize empty list
    linked_customers_ms = []
    
    # Iterate through the xij decision variable
    for (k, v) in served_customer_ms.items():
            
            # Filter the input warehouse and positive variable values
            if k[1]==input_warehouse and v.varValue>0:
                
                # Customer is served by the input warehouse
                linked_customers_ms.append(k[0])

    return linked_customers_ms

#-----------------------------------MS Distribution Plotting---------------------------------

# Warehouses to establish
establish = facility_df_ms

#establish.warehouse

# Plot the shape of Italy
ax = counties1.plot(color='white', edgecolor='black', figsize=(30, 30))

# Plot sites to establish
establish.\
    plot(ax=ax, marker='o', c='#0059b3', markersize=100, label='Warehouse')

# Plot customers
customer_df_ms.\
    plot(ax=ax, marker='X', color='#990000', markersize=80, alpha=0.8, label='Customer')

# For each warehouse to build
for w in establish.warehouse_id:

    # Extract list of customers served by the warehouse
    linked_customers_ms = get_linked_customers_ms(w)

    # For each served customer
    for c in linked_customers_ms:
    
        # Plot connection between warehouse and the served customer
        ax.plot(
         [establish.loc[establish.warehouse_id==w].lng, customer_df_ms.loc[customer_df_ms.customer_id==c].lng],
         [establish.loc[establish.warehouse_id==w].lat, customer_df_ms.loc[customer_df_ms.customer_id==c].lat],
         linewidth=0.8, linestyle='--', color='#0059b3')

# Add title
plt.title('Optimized Customers Supply', fontsize = 35)

# Add legend
plt.legend(facecolor='white', fontsize=30)

# Remove ticks from axis
plt.xticks([])
plt.yticks([])

# Show plot
plt.show()


facility_df_ms.to_csv("facility_df_ms.csv")

customer_df_ms.to_csv("customer_df_ms.csv")

#-------------------------------------Now creating a leaflet map-----------------------------------

#-------------------------------Adding more info to the map--------------------------------

# Round to two decimal places
customer_df_ms['2021_Crop_Acres'] = round(customer_df_ms['2021_Crop_Acres'], 2)


# Create a Leaflet map centered at a location
m = folium.Map(location=[36.2077, -119.3473], zoom_start=10)

# Function to create popup content with additional information for warehouses
def create_warehouse_popup_content(row):
    return f"<strong>{row['VistaName']} ({row['warehouse_id']})</strong><br>County: {row['NAME']}<br>Alfalfa Lbs Req: {row['Alfalfa_Req_Annum_lbs']}<br>Herd Size: {row['herd_size']}"

# Add warehouse markers to the map with customized popups
for _, row in establish.iterrows():
    lat_lon = f"{row['lat']}, {row['lng']}"
    popup_content = create_warehouse_popup_content(row)
    folium.Marker(
        location=[row['lat'], row['lng']],
        popup=folium.Popup(popup_content, max_width=300),  # Customized popup
        icon=folium.Icon(icon='cloud', prefix='fa', color='blue')
    ).add_to(m)

# Function to create customer popup content with additional information
def create_customer_popup_content(row):
    return f"<strong>{row['owner']} ({row['customer_id']})</strong><br>County: {row['county_name']}<br>2021 Crop Acres: {row['2021_Crop_Acres']}<br>2021 Alfalfa Lbs: {row['demand']}"

# Add customer markers to the map with customized popups
for _, row in customer_df_ms.iterrows():
    lat_lon = f"{row['lat']}, {row['lng']}"
    popup_content = create_customer_popup_content(row)
    folium.Marker(
        location=[row['lat'], row['lng']],
        popup=folium.Popup(popup_content, max_width=300),  # Customized popup
        icon=folium.Icon(icon='user', prefix='fa', color='green')
    ).add_to(m)

# Add connections between warehouses and customers as lines
for _, warehouse in establish.iterrows():
    linked_customers_ms = get_linked_customers_ms(warehouse['warehouse_id'])
    for customer_id in linked_customers_ms:
        customer = customer_df_ms[customer_df_ms['customer_id'] == customer_id].iloc[0]
        folium.PolyLine(
            locations=[
                (warehouse['lat'], warehouse['lng']),
                (customer['lat'], customer['lng'])
            ],
            color='green',
            opacity=0.5
        ).add_to(m)

# Save the map as an HTML file
m.save("Alfalfa_map.html")

# Display the map in your default web browser
import webbrowser
webbrowser.open("Alfalfa_map.html")



