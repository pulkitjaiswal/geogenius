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
from llama_index import Document, VectorStoreIndex, StorageContext
from llama_index import VectorStoreIndex, get_response_synthesizer
from llama_index.indices.service_context import ServiceContext
from llama_index.llms import OpenAI
import pinecone
from llama_index.vector_stores import PineconeVectorStore

openai.api_key = st.secrets['OPENAI_API_KEY']
os.environ["OPENAI_API_KEY"]= st.secrets['OPENAI_API_KEY']
c = Census(st.secrets['CENSUS_KEY'])
MONGO_URI = st.secrets['MONGO_URI']
client = MongoClient(MONGO_URI, tls=ssl.HAS_SNI, tlsAllowInvalidCertificates=True)
db = client['manhattan-project']
collection = db['demographics']
# Initializing OpenAI and ServiceContext
llm = OpenAI(model="gpt-4")
service_context = ServiceContext.from_defaults(llm=llm)


# Initialize Pinecone connection
def initialize_pinecone_connection(api_key: str, environment: str) -> None:
    pinecone.init(api_key=api_key, environment=environment)


# Set up the Pinecone Vector Store with specified metadata filters
def setup_vector_store(index_name: str) -> PineconeVectorStore:
    # Connect to the existing Pinecone index
    pinecone_index = pinecone.Index(index_name)
    

    # Construct the Pinecone vector store with the specified metadata filters
    vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
    return vector_store





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
        # Construct a prompt for the GPT model
        # Assuming census_tract_data is a pandas DataFrame with the relevant data
        tract_data_summary = census_tract_data[['Tract Name', 'Tract ID', 'Land Area (sq meters)', 'Water Area (sq meters)']].to_dict(orient='records')

        # Construct a prompt for the GPT model
        prompt = "Generate a detailed narrative analysis based on the following census tract data:\n\n"
        for tract in tract_data_summary:
            prompt += f"Tract Name: {tract['Tract Name']}, Tract ID: {tract['Tract ID']}, Land Area: {tract['Land Area (sq meters)']} square meters, Water Area: {tract.get('Water Area (sq meters)', 'unreported')} square meters.\n"

        # Extract potential company name from the user's question using OpenAI
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {
                    "role": "user",
                    "content": f"Using the data from the intersecting census tracts within a specified drive time area, generate a detailed narrative that describes the land distribution, potential for development, and any notable features of the area. The tracts are numbered and include details about the land area in square feet. The description should highlight the diversity of the region, the potential for various land use applications, and provide a vivid image of the region’s potential for future investors or city planning developments."
                }
            ],
            stream=True,
        )

        # Initialize an empty string to accumulate the response
        accumulated_response = ""

        # Process each event as it arrives
        for event in response:
            event_text = event.get('choices', [{}])[0].get('message', {}).get('content', '')
            if event_text:
                accumulated_response += event_text
                # Use Streamlit's empty container to update the UI
                placeholder = st.empty()
                placeholder.write(accumulated_response)
                time.sleep(0.1)  # Adjust the sleep time if necessary

        # After the loop, display the final accumulated response
        st.write(accumulated_response)
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
                tract_data_summary = census_data_for_map[['Tract Name', 'Tract ID', 'Land Area (sq meters)', 'Water Area (sq meters)']].to_dict(orient='records')
                st.write("### Land Usage And Development Potential")
                prompt = "Using the data from the intersecting census tracts within a specified drive time area, generate a detailed narrative (5 point detailed overview) that describes the land distribution, potential for development, and any notable features of the area. The tracts are numbered and include details about the land area in square feet. The description should highlight the diversity of the region, the potential for various land use applications, and provide a vivid image of the region’s potential for future investors or city planning developments. Finish with a conclusion sentence at the end. Consider the following data points for each tract::\n\n"
                for tract in tract_data_summary:
                    prompt += f"Tract Name: {tract['Tract Name']}, Tract ID: {tract['Tract ID']}, Land Area: {tract['Land Area (sq meters)']} square meters, Water Area: {tract.get('Water Area (sq meters)', 'unreported')} square meters.\n"

                delay_time = 0.01 #  faster
                answer = ''
                message_placeholder = st.empty()
                full_response = ""
                try:
                    full_response = ""
                    for response in openai.ChatCompletion.create(
                        model="gpt-4",
                        messages=[
                            {
                                "role": "user",
                                "content": prompt
                            }
                        ],
                        stream=True
                    ):
                        full_response += (response.choices[0].delta.content or "")
                        message_placeholder.markdown(full_response + "▌")
                    message_placeholder.markdown(full_response)
                except openai.error.OpenAIError as e:
                    #st.error(f"An error occurred with the OpenAI API call: {str(e)}")
                    pass
                except Exception as e:
                    #st.error(f"An unexpected error occurred: {str(e)}")
                    pass
                
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
                st.write('### Cartography')
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


def fetch_census_data(variables, state_code, county_code, tract_code, start_year=2017, end_year=2021):
    data = {}
    for var in variables:
        data[var] = {}
        for year in range(start_year, end_year + 1):
            try:
                # Query the ACS5 data for each variable and year
                query_result = c.acs5.state_county_tract(
                    fields=(var, 'NAME'),
                    state_fips=states.lookup(state_code).fips,
                    county_fips=county_code,
                    tract=tract_code,
                    year=year
                )
                if query_result:
                    data[var][year] = query_result[0]
                else:
                    data[var][year] = 'No data found'
            except Exception as e:
                data[var][year] = f"Error fetching data for variable {var} in year {year}: {e}"
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
            
          

            for i in range(rows):
                cols = st.columns(grid_cols)  # Create a new row of columns
                for j in range(grid_cols):
                    idx = i * grid_cols + j
                    if idx < len(sampled_documents):
                        doc = sampled_documents[idx]
                        address = doc.get('address', 'No Address')
                        drive_time = doc.get('census_tract_data', [{}])[0].get('drive_time', 'No Drive Time')
                        with cols[j]:  # Use the column context
                            # Create the flash card with address and drive time within the column
                            st.markdown(f"""
                            <div style='border:2px solid #3DFF50; border-radius:10px;
                                        padding:10px; position: relative; background-color: transparent;
                                        min-width: 220px; min-height:150px; box-sizing: border-box;margin-bottom:10px' onclick='alert("hi")'>
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
                use_radius = st.checkbox("Use radius instead of drive time")
                # Add a checkbox to toggle between drive time and radius mode
                use_radius = st.checkbox("Use radius instead of drive time", value=st.session_state.get('use_radius', False), key='use_radius')

                # Conditional display based on the checkbox state
                if use_radius:
                    # If the checkbox is checked, show a slider for radius in miles
                    radius_miles = st.slider("Radius in miles:", min_value=5, max_value=60, step=5, value=st.session_state.get('radius_miles', 5), key='radius_miles')
                    # Set the mode to 'radius' in session state
                    st.session_state['mode'] = 'radius'
                    # Store the radius value in session state
                    st.session_state['radius_miles'] = radius_miles
                else:
                    # If the checkbox is not checked, show a slider for drive time in minutes
                    drive_minutes = st.slider("Drive time in minutes:", min_value=5, max_value=60, value=st.session_state.get('drive_minutes', 10), key='drive_minutes')
                    # Set the mode to 'drive_time' in session state
                    st.session_state['mode'] = 'drive_time'
                    # Store the drive time value in session state
                    st.session_state['drive_minutes'] = drive_minutes

                # Set the clicked state to False to indicate that the map should not be generated yet
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
                    )
                    st.write('### Demographics')
                    if existing_record and 'aggregated_demographics_data' in existing_record:
                        st.write("Aggregated data already exists for this address and drive time.")
                        # Use the existing aggregated data
                        aggregated_data = existing_record['aggregated_demographics_data']
                    else:
                        st.write("No data cached yet, we're going to call API for each census tract")
                        # Initialize a dictionary to hold the aggregated data for each year
                        aggregated_data_by_year = []

                        for idx, tract in st.session_state.census_data.iterrows():
                            try:
                                geo_id = str(tract.get('Tract ID') or tract.get('GEOID', ''))
                                state_fips = geo_id[:2]
                                county_fips = geo_id[2:5]
                                tract_code = geo_id[5:]

                                # Fetch data for each tract and each year
                                tract_data = fetch_census_data(variable_codes, state_fips, county_fips, tract_code)

                                # Append the tract_data to the list
                                aggregated_data_by_year.append(tract_data)
                                st.write()

                            except Exception as e:
                                st.error(f"Error processing tract {geo_id}: {e}")

                            # Update progress
                            current_tract += 1
                            progress_percentage = current_tract / total_tracts
                            progress_bar.progress(progress_percentage)

                            # Update status message
                            status_message.text(f"Processing tract {current_tract} of {total_tracts} ({geo_id})")
                        # Update the document with the new 'aggregated_data_by_year' field
                        # Convert year keys to strings before updating MongoDB
                        # Ensure the aggregated data by year is in the correct format with string keys
                        # Update the MongoDB document with the new 'aggregated_data_by_year' list
                        # Convert year keys to strings for each dictionary in the list

                    
             
                        
                        # Initialize a dictionary to hold the aggregated data for each year and variable
                        # Initialize the aggregated data structure
                        aggregated_data = {str(year): {var: 0 for var in variable_codes} for year in range(2017, 2022)}

                        # Loop through the list of tract data and sum the values for each variable and year
                        # Debug: Check the structure of aggregated_data_by_year
                        #st.write("Sample data from aggregated_data_by_year:", aggregated_data_by_year[:2])

                        # Loop through the list of tract data and sum the values for each variable and year
                        for tract_data in aggregated_data_by_year:
                            for var in variable_codes:
                                #st.write(f"Variable: {var}")
                                if var in tract_data:
                                    #st.write(f"Keys for {var}: {list(tract_data[var].keys())}")  # Print out the keys
                                    pass
                                else:
                                    st.write(f"Variable {var} not found in tract_data.")
                                    continue  # Skip to the next variable if the current one is not found

                                for year in range(2017, 2022):
                                    # Use integer for year when checking if it's in the dictionary
                                    if year in tract_data[var]:
                                        data_value = tract_data[var][year].get(var, 0)
                                        #st.write(f"Year: {year}, Data Value: {data_value}")
                                        aggregated_data[str(year)][var] += data_value
                                    else:
                                        st.write(f"Year {year} not found for variable {var} in tract_data.")

                        # Convert integer keys to strings in aggregated_data
                        aggregated_data_str_keys = {str(year): data for year, data in aggregated_data.items()}

                        # Convert integer keys to strings in aggregated_data_by_year
                        aggregated_data_by_year_str_keys = []
                        for data in aggregated_data_by_year:
                            data_str_keys = {}
                            for var, year_data in data.items():
                                year_data_str_keys = {str(year): values for year, values in year_data.items()}
                                data_str_keys[var] = year_data_str_keys
                            aggregated_data_by_year_str_keys.append(data_str_keys)

                        # Now update the MongoDB document with the modified dictionaries
                        result = collection.update_one(
                            {"address": address, "census_tract_data.drive_time": drive_minutes},
                            {"$set": {
                                "aggregated_demographics_data": aggregated_data_str_keys,
                                "aggregated_demographics_data_by_year": aggregated_data_by_year_str_keys
                            }},
                            upsert=True  # This creates a new document if one doesn't exist
                        )

                        # Check if the update was successful
                        if result.matched_count > 0:
                            st.success(f"Document with address {address} updated successfully.")
                        else:
                            st.error(f"No document found with address {address} and drive time {drive_minutes}.")

                            
                    # Create a DataFrame from the aggregated demographic data
                    list_of_dicts = []
                    for var in variable_codes:
                        var_dict = {'Variable': variable_names[var]}  # Use the actual name instead of the code
                        for year in range(2017, 2022):
                            year_str = str(year)
                            var_dict[year_str] = aggregated_data.get(year_str, {}).get(var, 0)
                        list_of_dicts.append(var_dict)

                    df_aggregated = pd.DataFrame(list_of_dicts)
                    df_aggregated.set_index('Variable', inplace=True)
                    df_aggregated = df_aggregated.T
                    df_aggregated.index.name = 'Year'

                    # Display the DataFrame in Streamlit
                    st.dataframe(df_aggregated)
                    
                    st.write('#### Quantitative Analysis of Demographics')
                
                    # Construct the prompt for the GPT model
                    prompt = "Identify and highlight key quantitative insights related to demographics based on the following data over the past five years:\n\n"
                    for var in variable_codes:
                        prompt += f"{variable_names[var]}:\n"
                        for year in range(2017, 2022):
                            year_str = str(year)
                            value = aggregated_data.get(year_str, {}).get(var, 0)
                            prompt += f" - {year_str}: {value}\n"
                        prompt += "\n"

                    prompt += "Please provide a summary of significant demographic trends, potential implications for the real estate market, and any notable changes or patterns."
                    st.write(prompt)
                    # Call the OpenAI API with the prompt
                    try:
                        response = openai.ChatCompletion.create(
                            model="gpt-3.5-turbo-16k",
                            messages=[
                                {
                                    "role": "user",
                                    "content": prompt
                                }
                            ],
                            # Removed stream=True to get the full response at once
                        )

                        # Check if the response has content and display it
                        if response.choices:
                            response_content = response.choices[0].message['content']
                            if response_content:
                                # Display the response content
                                st.write(response_content)
                            else:
                                st.write("The response from the model is empty.")
                        else:
                            st.write("No response received from the model.")

                    except openai.error.OpenAIError as e:
                        return f"An error occurred with the OpenAI API call: {str(e)}"
                    except Exception as e:
                        return f"An unexpected error occurred: {str(e)}"    
                    


                    st.write('#### Recommendations for RE Developers')
                    # Initialize an empty placeholder before the loop
                    demographics_summary_placeholder = st.empty()
                    # Construct a prompt for the GPT model
                    prompt = "Generate a detailed narrative analysis for real estate development based on the following demographic data over the past five years:\n\n"
                    for var, name in variable_names.items():
                        prompt += f"{name}:\n"
                        for year in range(2017, 2022):
                            year_str = str(year)
                            value = aggregated_data.get(year_str, {}).get(var, 0)
                            prompt += f" - {year_str}: {value}\n"
                        prompt += "\n"

                    prompt += "Based on these demographic trends, provide insights into potential real estate market demands, opportunities for residential or commercial development, and recommendations for real estate investors or city planners to address the needs of the community. present in 5 detailed bullets followed by a conclusion"
                    #st.write(prompt)
                    # Extract potential company name from the user's question using OpenAI
                    try:
                        response = openai.ChatCompletion.create(
                            model="gpt-4",
                            messages=[
                                {
                                    "role": "user",
                                    "content": prompt
                                }
                            ],
                            # Removed stream=True to get the full response at once
                        )

                        # Check if the response has content and display it
                        if response.choices:
                            response_content = response.choices[0].message['content']
                            if response_content:
                                # Display the response content
                                st.write(response_content)
                            else:
                                st.write("The response from the model is empty.")
                        else:
                            st.write("No response received from the model.")

                    except openai.error.OpenAIError as e:
                        st.error(f"An error occurred with the OpenAI API call: {str(e)}")
                    except Exception as e:
                        st.error(f"An unexpected error occurred: {str(e)}")
            
                    st.write('## Neighborhood Key Attractions')
                    query_str = (
                        "Provide a brief overview (in bullet points) of the following key convenience points in the area of interest "
                        "to assess its attractiveness for the build-to-rent industry. The audience is real estate professionals "
                        "who are focused on understanding the potential of the area for residential development. Highlight the "
                        "variety and richness of local amenities and attractions, including their proximity to the residential site, "
                        "quality, and any standout features that would appeal to potential residents:\n\n"
                        "- Restaurants: Variety of cuisines, notable dining experiences, and any award-winning establishments.\n"
                        "- Cafes: Availability of casual coffee shops and work-friendly spaces with high-quality brews.\n"
                        "- Grocery Stores & Supermarkets: Access to fresh produce, organic options, and international food selections.\n"
                        "- Schools: Reputation and performance of local educational institutions, from primary to high school.\n"
                        "- Childcare Centers: Quality and availability of daycare services for working families.\n"
                        "- Parks & Recreational Areas: Presence of green spaces, playgrounds, and facilities for outdoor activities.\n"
                        "- Medical Facilities: Range of healthcare services, including hospitals, clinics, and specialist centers.\n"
                        "- Public Transport: Connectivity and convenience of the public transportation network.\n"
                        "- Shopping Centers & Malls: Diversity of retail stores, presence of major brands, and shopping convenience.\n"
                        "- Fitness Centers & Gyms: Quality of fitness facilities, availability of classes, and personal training services.\n"
                        "- Entertainment Venues: Options for nightlife, cinemas, theaters, and cultural events.\n\n"
                        "Each point should provide a snapshot that captures the essence of what these amenities offer to residents, "
                        "emphasizing aspects that are particularly attractive for the build-to-rent sector."
                    )


                    # Constants for Pinecone connection
                    API_KEY = st.secrets['PINECONE_API_KEY']
                    ENVIRONMENT = st.secrets['PINECONE_ENVIRONMENT']
                    INDEX_NAME = st.secrets['PINECONE_INDEX_NAME']


                    # Initialize Pinecone with the new API key and index name
                    initialize_pinecone_connection(API_KEY, ENVIRONMENT)
                    vector_store = setup_vector_store(INDEX_NAME)

                    # Create the storage context
                    storage_context = StorageContext.from_defaults(vector_store=vector_store)


                    index = VectorStoreIndex.from_vector_store(vector_store)
                    query_engine = index.as_query_engine(
                        mode='tree_summarize',
                        top_k=50,
                    )
                    # Query the index using the query string
                    response = query_engine.query(query_str)

                    # Print the synthesized response which should contain trading ideas
                    st.write(str(response.response))
                    attractions_query = st.text_input("Enter your attractions related query:")
                    query_str = (
                            f"Respond to the user's query: {attractions_query}; regarding local attractions and amenities that impact the attractiveness "
                            f"of an area for the build-to-rent industry. Real estate professionals are interested in a comprehensive understanding "
                            f"of the neighborhood's potential for residential development. Address the variety and quality of the following points, "
                            f"emphasizing their appeal to homebuyers and renters:\n\n"
                        )
                    if st.button("Submit"):
                        query_engine = index.as_query_engine(streaming=True, similarity_top_k=1)
                        response_stream = query_engine.query(query_str)
                        # Display the accumulated response
                        st.markdown("### Answer:")
                        placeholder = st.empty()

                        accumulated_response = ""
                        for chunk in response_stream.response_gen:
                            accumulated_response += chunk
                            placeholder.markdown(accumulated_response)
                    pass
        elif option == 'Reports':
            # Code for Reports
            pass
if __name__ == "__main__":
    main()