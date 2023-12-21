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
GOOGLE_MAPS_API_KEY = "AIzaSyD0SiBZN7Rp9Gr8v86q69iuHRVWWDyv9VQ"
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

openai.api_key = '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'
os.environ["OPENAI_API_KEY"]= "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"

MONGO_URI = 'mongodb+srv://overlord-one:sbNciWt8sf5KUkmU@asc-fin-data.oxz1gjj.mongodb.net/?retryWrites=true&w=majority'
client = MongoClient(MONGO_URI, tls=ssl.HAS_SNI, tlsAllowInvalidCertificates=True)
db = client['manhattan-project']
collection = db['demographics']


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
    
    # Add the intersecting census tracts to the map
    if not census_tract_data.empty:
        st.write("I will try to show census tract data")
        st.write(census_tract_data)
        folium.GeoJson(
            census_tract_data.to_json(),
            style_function=lambda feature: {
                'fillColor': '#ffff00',
                'color': 'red',
                'weight': 3,
                'dashArray': '5, 5'
            }
        ).add_to(map_)
    else: 
        st.write("census_tract_data is empty")
    return map_
   
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
    c = Census("9873cb96ddca9200a10b8c9f57c34fa09dc0ceaf")
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
def insert_address_to_mongo(address, drive_minutes):
    try:
        collection.insert_one({"address": address, "drive_minutes": drive_minutes})
        st.success("Address stored in MongoDB.")
    except Exception as e:
        st.error(f"Failed to store address: {e}")

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



def main():
    from datetime import datetime

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
            # User types an address
            user_input = st.text_input("Enter an address:", key='address_input')

            # Fetch and display matching addresses
            matching_addresses = fetch_matching_addresses(user_input)
            selected_address = st.selectbox("Matching addresses:", matching_addresses, key='address_select', on_change=on_address_select)

            drive_minutes = st.slider("Drive time in minutes:", 5, 60, 10, key='drive_minutes')


            # If the "Generate Map" button is pressed
            if st.button("Generate Map"):
                with st.spinner('Calculating drive time polygon...'):
                    address = st.session_state.address_input
                    if address:
                        lat, lon, state_code = geocode_address(address)
                        if lat and lon:
                            
                            st.session_state.map, st.session_state.census_data = None, None  # Resetting the map and census data
                            drive_time_polygon, gdf_points = get_drive_time_polygon(lat, lon, drive_minutes, GOOGLE_MAPS_API_KEY)

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
                            map_ = create_map(lat, lon, drive_time_polygon, gdf_points, census_tract_data)
                            insert_address_to_mongo(address, drive_minutes)
                            # Use folium to render the map in Streamlit
                            folium_static(map_)
                            
                            # Store map and census data in the session state
                            st.session_state.map = map_
                            st.session_state.census_data = census_tract_data.describe()
                        else:
                            st.error("Could not geocode the address.")
                    else:
                        st.error("Please enter an address and API Key.")

                    # Always display the map and census data if they have been generated
                    if st.session_state.map is not None:
                        #folium_static(st.session_state.map)
                        st.write(st.session_state.census_data)

                    # Input for type of census data after the map is generated
                    data_type = st.text_input("Specify the type of Census data you'd like to retrieve:")
                    submit_button = st.button("Submit Census Query")

                    # Inject the user input into the prompt and get the response from the LLM when "Submit Census Query" is clicked
                    if submit_button and data_type:
                        index = load_llama_index()  # Load the index
                        # Assuming OpenAI and query_engine are defined and set up correctly here
                        query_str = (
                            "The JSON file contains keys 'label', 'concept', 'predicateType', 'group', 'limit', and 'attributes'. "
                            f"I want to focus on the 'label' and 'concept' keys for {data_type} "
                            "The output should list the variable names (from 'label') and their descriptions (from 'concept') in a clear and structured way. "
                            "Here is a sample format for the output I am expecting:"
                            "\n\n"
                            "- C17002_001E: Estimate!!Total:!!$150,000 to $199,999: Household Income in the Past 12 Months (in 2022 Inflation-Adjusted Dollars) (Black or African American Alone Householder)\n"
                            "- C17002_002E: Estimate!!Total:!!$200,000 or more: Household Income in the Past 12 Months (in 2022 Inflation-Adjusted Dollars) (Black or African American Alone Householder)\n"
                            "\n"
                            f"Please extract the variable names and their corresponding descriptions following this format, focusing on the ones relevant to {data_type} . Make sure to be exhaustive in your search and surface a diverse range of categories of data and not just repetition of the same type."
                        )
                        llm = OpenAI(model="gpt-4")
                        query_engine = index.as_query_engine(
                            mode='tree_summarize',
                            top_k=50,
                        )
                        # Here you would execute the query and store the result in session state
                        st.session_state.response_str = "LLM data"  # Placeholder for the actual LLM query result
                        
                    # Display the LLM Response if available
                    if st.session_state.response_str:
                        st.text("LLM Response:")

                        response_str = query_engine.query(query_str)
                        st.write(response_str.response)

                        # Adjust this pattern according to the actual format of variable codes in response_str
                        pattern = r'- (B\d{5}_\d{3}E):'

                        # Find all matches in the response string
                        variable_codes = re.findall(pattern, response_str.response)
                        fetch_census_data_button = st.button("Fetch Census Data")
                        if fetch_census_data_button:
                            if variable_codes:
                                # Fetch the census data using the variable codes
                                state_code = '26'  # California
                                county_code = '125'  # San Francisco County
                                tract_code = '168901'
                                census_data = fetch_census_data(variable_codes, state_code, county_code, tract_code)
                                st.write(census_data)
                            else:
                                st.error("No variable codes found in the response.")
            pass
        elif option == 'Reports':
            # Code for Reports
            pass
if __name__ == "__main__":
    main()