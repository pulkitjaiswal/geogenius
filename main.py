import streamlit as st
import requests
import folium
from streamlit_folium import folium_static
from math import radians, cos, sin, asin, sqrt, degrees, atan2
from shapely.geometry import Point, Polygon
import geopandas as gpd
import pandas as pd
import pygris
import shapely
import polyline
import logging
import json
from shapely.geometry import MultiPoint
from census import Census
from us import states
import base64
from llama_index import StorageContext, load_index_from_storage
from pathlib import Path
from llama_index.storage.docstore import SimpleDocumentStore
from llama_index.storage.index_store import SimpleIndexStore
from llama_index.vector_stores import SimpleVectorStore
from llama_index.llms import OpenAI
import openai
import os
import re
import auth_functions
from pymongo import MongoClient
import ssl
import time
from shapely.geometry import mapping
from shapely.geometry import Point, Polygon, shape  # Include 'shape' here

openai.api_key = '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'
os.environ["OPENAI_API_KEY"]= "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"
c = Census("9873cb96ddca9200a10b8c9f57c34fa09dc0ceaf")
MONGO_URI = 'mongodb+srv://overlord-one:sbNciWt8sf5KUkmU@asc-fin-data.oxz1gjj.mongodb.net/?retryWrites=true&w=majority'
client = MongoClient(MONGO_URI, tls=ssl.HAS_SNI, tlsAllowInvalidCertificates=True)
db = client['manhattan-project']
collection = db['demographics']

import numpy as np

def weighted_median(data, weights):
    """
    Calculate the weighted median of a dataset.
    
    data: list or array-like, the data values.
    weights: list or array-like, the weights corresponding to the data.
    """
    data, weights = np.array(data), np.array(weights)
    sorted_indices = np.argsort(data)
    sorted_data = data[sorted_indices]
    sorted_weights = weights[sorted_indices]
    cumulative_weights = np.cumsum(sorted_weights)
    midpoint = 0.5 * sum(sorted_weights)
    if any(cumulative_weights > midpoint):
        median_idx = np.where(cumulative_weights >= midpoint)[0][0]
        return sorted_data[median_idx]
    return data[len(data) // 2]


variable_codes = [
    'B01003_001E',  # Total Population
    'B19013_001E',  # Median Household Income
    'B25077_001E',  # Median Home Value
    'B01001_011E',  # Population by Age Group: Males 65 to 74
    'B01001_035E',  # Population by Age Group: Females 65 to 74
    'B01001_019E',  # Population by Age Group: Males 75 to 84
    'B01001_043E',  # Population by Age Group: Females 75 to 84
    'B01001_023E',  # Population by Age Group: Males 85+
    'B01001_047E'   # Population by Age Group: Females 85+
    # Additional income and age-related variables for calculating median income for 75+ households
    # would be required, but they are not specified here.
]


variable_names = {
    'B01003_001E': 'Total Population',
    'B19013_001E': 'Median Household Income',
    'B25077_001E': 'Median Home Value',
    'B01001_011E': 'Male Population Age 65-74',
    'B01001_035E': 'Female Population Age 65-74',
    'B01001_019E': 'Male Population Age 75-84',
    'B01001_043E': 'Female Population Age 75-84',
    'B01001_023E': 'Male Population Age 85+',
    'B01001_047E': 'Female Population Age 85+',
}
def download_link(object_to_download, download_filename, download_link_text):
    """
    Generates a link to download the given object_to_download.
    """
    if isinstance(object_to_download, pd.DataFrame):
        object_to_download = object_to_download.to_csv(index=False)

    # Create a byte representation of the object
    b64 = base64.b64encode(object_to_download.encode()).decode()

    # Create the download link
    href = f'<a href="data:file/txt;base64,{b64}" download="{download_filename}">{download_link_text}</a>'
    return href


def haversine(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 6371 # Radius of earth in kilometers. Use 3956 for miles
    return c * r



def geocode_address(address):
    """Geocode an address to lat/long coordinates using Google Maps API."""
    GOOGLE_MAPS_API_KEY = st.secrets["GOOGLE_MAPS_API_KEY"]
    geocode_url = f"https://maps.googleapis.com/maps/api/geocode/json?address={address}&key={GOOGLE_MAPS_API_KEY}"
    response = requests.get(geocode_url)
    if response.status_code == 200:
        results = response.json()['results']
        if results:
            location = results[0]['geometry']['location']
            state_code = next((component['short_name'] for component in results[0]['address_components'] if 'administrative_area_level_1' in component['types']), None)
            return location['lat'], location['lng'], state_code
    return None, None, None

def get_drive_time_polygon(lat, lon, minutes, api_key):
    """
    Get the drive time polygon coordinates from the given lat/long using Google Maps API.
    """
    average_speed_km_per_hour = 50
    distance_km = average_speed_km_per_hour * (minutes / 60.0)
    directions_base_url = "https://maps.googleapis.com/maps/api/directions/json"
    params = {"origin": f"{lat},{lon}", "mode": "driving", "key": api_key}
    gdf_points = gpd.GeoDataFrame(columns=['geometry'], geometry='geometry')

    for bearing_degrees in range(0, 360, 30):  # Increased step to reduce points
        bearing_radians = radians(bearing_degrees)
        destination_lat, destination_lon = get_destination_point(lat, lon, bearing_radians, distance_km)
        
        params['destination'] = f"{destination_lat},{destination_lon}"
        response = requests.get(directions_base_url, params=params)
        data = response.json()

        if data['routes']:
            for step in data['routes'][0]['legs'][0]['steps']:  # Consider only the first leg and step
                step_duration = step['duration']['value'] / 60
                if step_duration <= minutes:
                    polyline_str = step['polyline']['points']
                    decoded_points = decode_polyline(polyline_str)
                    # Add only a subset of points for efficiency
                    for pnt in decoded_points[::5]:  # Skip every 4 points
                        point_geom = Point(pnt[1], pnt[0])
                        gdf_points = pd.concat([gdf_points, gpd.GeoDataFrame({'geometry': [point_geom]})], ignore_index=True)

    if not gdf_points.empty:
        unified_geometry = gdf_points.unary_union
        if isinstance(unified_geometry, MultiPoint):
            polygon = unified_geometry.convex_hull
            if isinstance(polygon, Polygon):
                return list(zip(polygon.exterior.coords.xy[1], polygon.exterior.coords.xy[0])), gdf_points
            else:
                st.write("Collected points do not form a polygon.")
                return [], gdf_points
        else:
            st.write("Unified geometry is not a MultiPoint. Possibly a LineString.")
            return [], gdf_points
    else:
        st.write("No points were collected to form a polygon.")
        return [], gdf_points

def get_destination_point(lat, lon, bearing, distance_km):
    """
    Calculate a destination point given an origin, bearing, and distance.
    """
    try:
        if not (-90 <= lat <= 90 and -180 <= lon <= 180):
            raise ValueError("Invalid latitude or longitude.")
        if distance_km < 0:
            raise ValueError("Distance cannot be negative.")
        R = 6371.0
        d = distance_km / R
        lat1 = radians(lat)
        lon1 = radians(lon)
        lat2 = asin(sin(lat1) * cos(d) + cos(lat1) * sin(d) * cos(bearing))
        lon2 = lon1 + atan2(sin(bearing) * sin(d) * cos(lat1), cos(d) - sin(lat1) * sin(lat2))
        return degrees(lat2), degrees(lon2)
    except Exception as e:
        logging.error(f"Error in get_destination_point: {e}")
        return lat, lon

def decode_polyline(polyline_str):
    """
    Decodes a polyline that was encoded using the Google Maps Polyline Algorithm.
    """
    return polyline.decode(polyline_str)

def create_map(lat, lon, drive_time_polygon, gdf_points, census_tract_data):
    """
    Create a folium map with the drive time polygon overlay and markers for each point.
    """
    map_ = folium.Map(location=[lat, lon], zoom_start=12)
    
    if drive_time_polygon:
        folium.Polygon(locations=drive_time_polygon, color='blue', fill=True, fill_color='blue').add_to(map_)
        st.write(f"🌍 Generated a polygon representing a {len(drive_time_polygon)}-point drive time area.")

    # Add the intersecting census tracts to the map
    if not census_tract_data.empty:
        folium.GeoJson(
            census_tract_data.to_json(),
            style_function=lambda feature: {
                'fillColor': '#ffff00',
                'color': 'red',
                'weight': 3,
                'dashArray': '5, 5'
            }
        ).add_to(map_)
        st.write(f"📊 Found {len(census_tract_data)} census tracts intersecting with the drive time area.")
        st.write("🔍 Details of Intersecting Census Tracts:")
        # Renaming columns for better readability
        census_tract_data = census_tract_data.rename(columns={'NAME': 'Tract Name', 'GEOID': 'Tract ID', 'ALAND': 'Land Area (sq meters)', 'AWATER': 'Water Area (sq meters)'})
        st.dataframe(census_tract_data[['Tract Name', 'Tract ID', 'Land Area (sq meters)', 'Water Area (sq meters)']])
    else: 
        st.write("❗️ No census tracts intersect within the specified drive time area.")

    st.write("🔎 Here's the detailed map with the drive time area and intersecting census tracts:")
    return map_


def load_map_from_mongo(address, drive_minutes):
    try:
        # Query MongoDB for the stored data
        query_result = collection.find_one({"address": address})
        if query_result:
            # Find the item in census_tract_data with the matching drive_time
            matching_item = next((item for item in query_result.get("census_tract_data", []) if item.get("drive_time") == drive_minutes), None)
            if matching_item is None:
                st.write("❗️ No data found for the specified drive time.")
                return False, None
            else: 
                st.write("❗️ Pulling data directly from database.")
            # Get the stored census tracts data
            census_data_for_map = matching_item.get("tracts", [])

            # Get the stored drive_time_polygon
            drive_time_polygon = matching_item.get("drive_time_polygon", {})

            # Create map
            lat, lon, _ = geocode_address(address)  # Reusing the geocode_address function
            map_ = folium.Map(location=[lat, lon], zoom_start=12)

            # Add the census tracts to the map
            if not census_data_for_map:
                st.write("❗️ No census tracts intersect within the specified drive time area.")
            else:
                for tract in census_data_for_map:
                    # Convert GeoJSON back to Shapely Polygon
                    tract_polygon = shape(tract['geometry'])
                    folium.GeoJson(tract_polygon, style_function=lambda feature: {
                        'fillColor': '#ffff00',
                        'color': 'red',
                        'weight': 3,
                        'dashArray': '5, 5'
                    }).add_to(map_)
                st.write(f"📊 Found {len(census_data_for_map)} census tracts intersecting with the drive time area.")
                st.write("🔍 Details of Intersecting Census Tracts:")
                # Renaming columns for better readability
                census_data_for_map = pd.DataFrame(census_data_for_map).rename(columns={'name': 'Tract Name', 'geoid': 'Tract ID', 'aland': 'Land Area (sq meters)', 'awater': 'Water Area (sq meters)'})
                st.dataframe(census_data_for_map[['Tract Name', 'Tract ID', 'Land Area (sq meters)', 'Water Area (sq meters)']])
            geojson_polygon = {
                "type": "Polygon",
                "coordinates": [[
                    [item[1], item[0]] for item in drive_time_polygon["coordinates"][0]
                ]]
            }

            # Add the drive time polygon to the map
            folium.GeoJson(geojson_polygon, style_function=lambda feature: {
                'fillColor': '#0000ff',
                'color': 'blue',
                'weight': 3,
            }).add_to(map_)

            # Display the map
            st.write(f"🌍 Map for {address}, Drive Time: {drive_minutes} minutes")
            folium_static(map_)
            return True, census_data_for_map  # Return census data along with success status
        else:
            st.write("❗️ No data found for the specified address.")
            return False, None
    except Exception as e:
        st.write(f"❗️ Failed to load data from MongoDB: {str(e)}")
        return False, None
    

def get_census_tract_data(drive_time_polygon, state_code):
    """
    Fetch the census tract data within the given drive time polygon using PyGRIS.
    """
    # Load the US census tract data from PyGRIS
    gdf = pygris.tracts(state=state_code, year='2022')

    # Convert the drive_time_polygon object to a Shapely Polygon object
    drive_time_polygon_geom = gpd.GeoDataFrame(geometry=[shapely.geometry.Polygon([(lon, lat) for lat, lon in drive_time_polygon])], crs=gdf.crs)
    
    # Extract the data and return
    census_tract_data = gdf[gdf.geometry.within(drive_time_polygon_geom.geometry.unary_union)]
    #st.write(census_tract_data)
    return census_tract_data

# Function to load the index from the default storage
def load_llama_index():
    storage_context = StorageContext.from_defaults(
        docstore=SimpleDocumentStore.from_persist_dir(persist_dir="./storage"),
        vector_store=SimpleVectorStore.from_persist_dir(persist_dir="./storage"),
        index_store=SimpleIndexStore.from_persist_dir(persist_dir="./storage"),
    )
    # Assuming there's only one index in the storage context
    index = load_index_from_storage(storage_context)
    return index

def fetch_census_data(variables, state_code, county_code, tract_code, year=2021):
    
    data = {}
    for var in variables:
        try:
            # Query the ACS5 data for each variable
            query_result = c.acs5.state_county_tract(
                fields = (var, 'NAME'), 
                state_fips=states.lookup(state_code).fips, 
                county_fips=county_code, 
                tract=tract_code,
                year=year
            )
            if query_result:
                data[var] = query_result[0]
            else:
                data[var] = 'No data found'
        except Exception as e:
            data[var] = f"Error fetching data for variable {var}: {e}"
    return data

def generate_real_estate_insights(census_data):
    """
    Function to generate insights for the build-to-rent industry in real estate using GPT-3.5 Turbo,
    based on the provided census data.
    """
    # Initial system and user messages
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "I have census data related to housing and demographics. Can you analyze it and provide insights for the build-to-rent industry in real estate, focusing on market trends, demographic shifts, and potential opportunities?"}
    ]

    # Adding census data as user messages
    for key, value in census_data.items():
        messages.append({"role": "user", "content": f"Census variable {key} shows the following data: {value}"})

    # Call the GPT-3.5 Turbo API for insights
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages
        )
        return response.choices[0].message['content']
    except Exception as e:
        return str(e)

# Function to insert address into MongoDB
def insert_address_to_mongo(address, drive_minutes, census_tract_data, drive_time_polygon):
    try:
        # Convert Shapely Polygons in census_tract_data to GeoJSON
        census_data_for_mongo = [{
            'name': tract.NAME,
            'geoid': tract.GEOID,
            'aland': tract.ALAND,
            'awater': tract.AWATER,
            'geometry': mapping(tract.geometry)
        } for tract in census_tract_data.itertuples()]

       # Convert list of coordinates to a Shapely Polygon
        drive_time_polygon_geom = Polygon(drive_time_polygon)

        # Convert Shapely Polygon to GeoJSON
        drive_time_polygon_geojson = mapping(drive_time_polygon_geom)

        # Check if address already exists
        existing_record = collection.find_one({"address": address})
        if existing_record:
            # Update if new drive time data
            if not any(drive["drive_time"] == drive_minutes for drive in existing_record["census_tract_data"]):
                collection.update_one(
                    {"_id": existing_record["_id"]},
                    {"$push": {"census_tract_data": {"drive_time": drive_minutes, "tracts": census_data_for_mongo, "drive_time_polygon": drive_time_polygon_geojson}}}
                )
        else:
            # Insert new document
            collection.insert_one({
                "address": address,
                "census_tract_data": [{"drive_time": drive_minutes, "tracts": census_data_for_mongo, "drive_time_polygon": drive_time_polygon_geojson}]
            })
        st.success("Data stored in MongoDB.")
    except Exception as e:
        st.error(f"Failed to store data: {e}")


# Function to fetch matching addresses from the database
def fetch_matching_addresses(input_text):
    if input_text:
        regex_pattern = '^' + input_text  # Starts with the input text
        cursor = collection.find({"address": {"$regex": regex_pattern, "$options": "i"}})
        return [doc['address'] for doc in cursor]
    return []

# Callback function to handle address selection
def on_address_select():
    st.session_state.address_input = st.session_state.address_select


def fetch_census_data(variables, state_code, county_code, tract_code, year=2021):
    data = {}
    for var in variables:
        try:
            # Query the ACS5 data for each variable
            query_result = c.acs5.state_county_tract(
                fields = (var, 'NAME'), 
                state_fips=states.lookup(state_code).fips, 
                county_fips=county_code, 
                tract=tract_code,
                year=year
            )
            if query_result:
                data[var] = query_result[0]
            else:
                data[var] = 'No data found'
        except Exception as e:
            data[var] = f"Error fetching data for variable {var}: {e}"
    return data


def main():
    from datetime import datetime
    geo_id = None
    if 'user_info' not in st.session_state:
        col1,col2,col3 = st.columns([1,2,1])

        # Authentication form layout
        do_you_have_an_account = col2.selectbox(label='Do you have an account?',options=('Yes','No','I forgot my password'))
        auth_form = col2.form(key='Authentication form',clear_on_submit=False)
        email = auth_form.text_input(label='Email')
        password = auth_form.text_input(label='Password',type='password') if do_you_have_an_account in {'Yes','No'} else auth_form.empty()
        auth_notification = col2.empty()

        # Sign In
        if do_you_have_an_account == 'Yes' and auth_form.form_submit_button(label='Sign In',use_container_width=True,type='primary'):
            with auth_notification, st.spinner('Signing in'):
                auth_functions.sign_in(email,password)

        # Create Account
        elif do_you_have_an_account == 'No' and auth_form.form_submit_button(label='Create Account',use_container_width=True,type='primary'):
            with auth_notification, st.spinner('Creating account'):
                auth_functions.create_account(email,password)

        # Password Reset
        elif do_you_have_an_account == 'I forgot my password' and auth_form.form_submit_button(label='Send Password Reset Email',use_container_width=True,type='primary'):
            with auth_notification, st.spinner('Sending password reset link'):
                auth_functions.reset_password(email)

        # Authentication success and warning messages
        if 'auth_success' in st.session_state:
            auth_notification.success(st.session_state.auth_success)
            del st.session_state.auth_success
        elif 'auth_warning' in st.session_state:
            auth_notification.warning(st.session_state.auth_warning)
            del st.session_state.auth_warning

    ## -------------------------------------------------------------------------------------------------
    ## Logged in --------------------------------------------------------------------------------------
    ## -------------------------------------------------------------------------------------------------
    else:
        if 'response_str' not in st.session_state:
            st.session_state.response_str = ''
        # Load your logo
        st.sidebar.image("haystacks_logo.svg", use_column_width=True)
        # Show user information
        # Show user information
        st.sidebar.title("Navigation")
        option = st.sidebar.selectbox(
            'Select an option',
            ('Address Lookup', 'Reports')
        )

        # Show user information in the sidebar
        if 'user_info' in st.session_state:
            user_email = st.session_state.user_info.get("email", "No email")
            last_login = st.session_state.user_info.get("lastLoginAt", None)
            if last_login:
                # Convert from milliseconds to datetime
                last_login = datetime.fromtimestamp(int(last_login) / 1000).strftime('%Y-%m-%d %H:%M:%S')

            st.sidebar.subheader("User Information")
            st.sidebar.write(f"Email: {user_email}")
            st.sidebar.write(f"Last Login: {last_login}")

        # Sign out button in the sidebar
        st.sidebar.subheader('Sign out:')
        if st.sidebar.button(label='Sign Out', on_click=auth_functions.sign_out):
            # Perform sign out operations here
            pass

        


        # Use the selected option
        if option == 'Address Lookup':
            # Sample up to 5 random documents from MongoDB
            sampled_documents = list(collection.aggregate([{'$sample': {'size': 6}}]))

            # Get a list of addresses from the sampled documents
            addresses = [doc['address'] for doc in sampled_documents]
            # Define custom CSS to style the flash cards
            # Custom CSS for the flash cards
      

           
            st.session_state['clicked'] = False
            # Display the flash cards with addresses and drive times
            st.write("### Explore Prime Addresses & Insights")
            # Create a container for horizontal scrolling
            # Calculate the number of columns based on the number of sampled documents
            grid_cols = 3
            rows = (len(sampled_documents) + grid_cols - 1) // grid_cols
            
          
            # Define the custom CSS
            button_style = """
            <style>
            div.stButton > button:first-child {
                border: 2px solid #28a745 !important;
                color: #28a745;
            }
            </style>
            """

            # Render the custom CSS with the markdown
            st.markdown(button_style, unsafe_allow_html=True)

            for i in range(rows):
                cols = st.columns(grid_cols)  # Create a new row of columns
                for j in range(grid_cols):
                    idx = i * grid_cols + j
                    if idx < len(sampled_documents):
                        doc = sampled_documents[idx]
                        address = doc.get('address', 'No Address')
                        drive_time = doc.get('census_tract_data', [{}])[0].get('drive_time', 'No Drive Time')
                        with cols[j]:  # Use the column context
                            # Create a button for each address
                            if st.button(address, key=f"address_{idx}"):
                                st.session_state['address_input'] = address
                                st.session_state['drive_minutes'] = drive_time
                                st.session_state["trigger_map_generation"] = True
                                st.session_state["clicked"] = True
                                time.sleep(1)  # Delay for 1 second
                                st.rerun()
                            # Create the flash card with address and drive time within the column
                            st.markdown(f"""
                            <div style='border:2px solid #3DFF50; border-radius:10px;
                                        padding:10px; position: relative; background-color: transparent;
                                        min-width: 220px; min-height:150px; box-sizing: border-box;margin-bottom:10px'>
                                <h5 style='margin-bottom: 5px;'>{address}</h3>
                                <div style='position: absolute; bottom: 10px; right: 10px;
                                            background-color: #3DFF50; color: #000;
                                            padding: 2px 5px; border-radius: 5px;
                                            font-weight: bold;'>{drive_time} min</div>
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        with cols[j]:  # Create an empty column if no more documents
                            st.write("")

            # Check if a map should be generated based on a button click
            if st.session_state.get('trigger_map_generation', True):
                #st.write(st.session_state.get('address_input', ''), key='address_input')
                # Populate the address input and drive time slider with the selected values
                address_input = st.text_input("Enter an address:", value=st.session_state.get('address_input', ''), key='address_input')
                drive_minutes = st.slider("Drive time in minutes:", 5, 60, value=st.session_state.get('drive_minutes', 10), key='drive_minutes')
                st.session_state['clicked'] = True
                # Rerun the script to update the UI
                

                # Reset the trigger to avoid repeated map generation
            else:
                # Normal display of the widgets
                # Use the session state variable as the value for the text input
                address_input = st.text_input("Enter an address:", value=st.session_state.get('address_input', ''), key='address_input')
                drive_minutes = st.slider("Drive time in minutes:", 5, 60, 10, key='drive_minutes')
                st.session_state['clicked'] = False

            # If the "Generate Map" button is pressed, set the trigger for map generation
            if st.button("Generate Map"):
                st.session_state['clicked'] = True
                

            # Check if a map should be generated based on a button click or the session state trigger
            if st.session_state.get('clicked', True):
                st.session_state['clicked'] = False
                address = st.session_state.address_input
                if address:
                    data_loaded, census_tract_data = load_map_from_mongo(address, drive_minutes)
                
                    # Try to load map from MongoDB
                    if not data_loaded:
                        # Generate new map if data not in MongoDB
                        with st.spinner('Calculating drive time polygon...'):
                            lat, lon, state_code = geocode_address(address)
                            if lat and lon:
                                st.session_state.map, st.session_state.census_data = None, None  # Resetting the map and census data
                                drive_time_polygon, gdf_points = get_drive_time_polygon(lat, lon, drive_minutes, st.secrets["GOOGLE_MAPS_API_KEY"])

                                # Placeholder for progress updates
                                progress_placeholder = st.empty()

                                # Show progress and updates
                                for i in range(100):
                                    time.sleep(0.08)  # simulate processing time
                                    if i % 20 == 0:
                                        progress_placeholder.write(f"Processing...{i}% done")

                                # Clear the progress messages
                                progress_placeholder.empty()



                                census_tract_data = get_census_tract_data(drive_time_polygon, state_code)
                                if census_tract_data is not None:
                                    map_ = create_map(lat, lon, drive_time_polygon, gdf_points, census_tract_data)
                                    # Insert new data into MongoDB
                                    drive_time_polygon, gdf_points = get_drive_time_polygon(lat, lon, drive_minutes, st.secrets["GOOGLE_MAPS_API_KEY"])

                                    insert_address_to_mongo(address, drive_minutes, census_tract_data, drive_time_polygon)
                                    folium_static(map_)
                                    st.session_state.map = map_
                                    st.session_state.census_data = census_tract_data
                                else:
                                    st.error("Census tract data not found for the specified address and drive time.")
                    else:
                        # Get lat and lon from the MongoDB document
                        document = collection.find_one({"address": address, "drive_minutes": drive_minutes})
                        #lat, lon, state_code = geocode_address(address)
    
                        # Use the returned census_tract_data to create the map and update the session state
                        #drive_time_polygon, gdf_points = get_drive_time_polygon(lat, lon, drive_minutes, GOOGLE_MAPS_API_KEY)
                        #map_ = create_map(lat, lon, drive_time_polygon, gdf_points, census_tract_data)
                        #folium_static(map_)
                        #st.session_state.map = map_
                        st.session_state.census_data = census_tract_data

                        # Print the column names to verify if 'GEOID' exists
                        #st.write("Column names in census_tract_data:", census_tract_data.columns)
                        
                    # Initialize an empty dictionary for aggregated data
                    aggregated_data = {key: 0 for key in variable_codes}

                    # Initialize a progress bar
                    progress_bar = st.progress(0)
                    total_tracts = len(st.session_state.census_data)
                    current_tract = 0
                    # Create a placeholder for status messages
                    status_message = st.empty()
                    # Iterate over each tract in the drive time polygon
                    existing_record = collection.find_one(
                        {"address": address, "census_tract_data.drive_time": drive_minutes},
                        {"census_tract_data.$": 1}
                    )
                    if existing_record and 'aggregated_data' in existing_record['census_tract_data'][0]:
                        st.write("Aggregated data already exists for this address and drive time.")
                        aggregated_data = existing_record['census_tract_data'][0]['aggregated_data']
                    else:
                        st.write("No data cached yet, we're going to call API for each census tract")    
                        for idx, tract in st.session_state.census_data.iterrows():
                            try:
                                geo_id = str(tract.get('Tract ID') or tract.get('GEOID', ''))
                                state_fips = geo_id[:2]
                                county_fips = geo_id[2:5]
                                tract_code = geo_id[5:]

                                # Fetch data for each tract
                                tract_data = fetch_census_data(variable_codes, state_fips, county_fips, tract_code)

                                # Accumulate results and update progress
                                for key in tract_data:
                                    if key in aggregated_data:
                                        aggregated_data[key] += tract_data[key].get(key, 0)


                            except Exception as e:
                                st.error(f"Error processing tract {geo_id}: {e}")

                            # Update progress
                            current_tract += 1
                            progress_percentage = current_tract / total_tracts
                            progress_bar.progress(progress_percentage)

                            # Update status message
                            status_message.text(f"Processing tract {current_tract} of {total_tracts} ({geo_id})")
                        
                        
                        # Update the document with the new 'aggregated_data' field
                        result = collection.update_one(
                            {"address": address, "census_tract_data.drive_time": drive_minutes},
                            {"$set": {"census_tract_data.$.aggregated_data": aggregated_data}}
                        )

                        # Check if the update was successful
                        if result.matched_count > 0:
                            print(f"Document with address {address} updated successfully.")
                        else:
                            print(f"No document found with address {address} and drive time {drive_minutes}.")
                    # Create a dictionary to hold the transformed data with readable column names
                    readable_aggregated_data = {}
                    # Loop through the variable_names dictionary to map codes to readable names
                    for code, readable_name in variable_names.items():
                        data = aggregated_data.get(code, None)
                        
                        if data is not None and isinstance(data, list):
                            if 'Median' in readable_name:
                                # Extract the list of values and weights for the current code
                                try:
                                    values, weights = zip(*data)
                                    # Calculate the weighted median
                                    readable_aggregated_data[readable_name] = weighted_median(values, weights)
                                except TypeError as e:
                                    st.error(f"Error calculating weighted median for {readable_name}: {e}")
                                    st.error(f"Data received: {data}")
                            else:
                                # For total counts, sum the values
                                try:
                                    total = sum(value for value, weight in data)
                                    readable_aggregated_data[readable_name] = total
                                except TypeError as e:
                                    st.error(f"Error summing values for {readable_name}: {e}")
                                    st.error(f"Data received: {data}")
                        else:
                            # Handle the case where data is not a list of tuples
                            readable_aggregated_data[readable_name] = data if isinstance(data, (int, float)) else 0

                    # Create a pandas DataFrame using the transformed data
                    df = pd.DataFrame([readable_aggregated_data])

                    # Display the DataFrame in Streamlit
                    st.dataframe(df)
                    status_message.empty()
                    
            pass
        elif option == 'Reports':
            # Code for Reports
            pass
if __name__ == "__main__":
    main()