# README for Streamlit Application

## Introduction

Welcome to the Streamlit application! This document will guide you through setting up Git, understanding, and running the Streamlit application that integrates various technologies like Google Maps API, MongoDB, and OpenAI's GPT-3.5 Turbo API.

## Prerequisites

Before you start, ensure you have the following:

- Python installed on your computer.
- Basic understanding of Python programming.
- A code editor (like VSCode, PyCharm, etc.).
- A command-line interface (like Terminal on MacOS or CMD on Windows).

## Setting Up Git

1. **Install Git:**
   - **Windows:** Download and install from [Git for Windows](https://git-scm.com/download/win).
   - **MacOS:** Install via Homebrew with `brew install git` or download from [Git for MacOS](https://git-scm.com/download/mac).
   - **Linux:** Install via your distribution's package manager, for example, on Ubuntu use `sudo apt-get install git`.

2. **Configure Git:**
   - Set your name: `git config --global user.name "Your Name"`
   - Set your email: `git config --global user.email "your.email@example.com"`

3. **Clone the Repository:**
   - Use `git clone <repository-url>` to clone the remote repository to your local machine.
   - Navigate to the cloned directory using `cd <repository-name>`.

## Getting Started with Streamlit

1. **Install Streamlit:**
   - Run `pip install streamlit` in your command line.

2. **Run the Streamlit Application:**
   - Inside the project directory, run `streamlit run main.py`.
   - The application will open in your default web browser.

3. **Understanding the Code:**
   - `main.py` is the entry point of the application.
   - The code integrates various libraries and APIs:
     - **Folium:** For map rendering.
     - **GeoPandas:** For geographical data manipulation.
     - **PyGRIS:** For accessing census tract data.
     - **MongoDB:** For database operations.
     - **OpenAI:** For generating real estate insights using GPT-3.5 Turbo.

4. **MongoDB Integration:**
   - The app connects to a MongoDB database for storing and retrieving addresses.
   - Ensure the MongoDB URI is correctly set in the `MONGO_URI` variable.

5. **Google Maps API Integration:**
   - The app uses Google Maps API for geocoding addresses.
   - Ensure the Google Maps API key is set in the `GOOGLE_MAPS_API_KEY` variable.

6. **Function Descriptions:**
   - `download_link`: Generates a download link for data.
   - `haversine`: Calculates the distance between two points on Earth.
   - `geocode_address`: Converts an address into latitude and longitude.
   - `get_drive_time_polygon`: Fetches drive time polygon data.
   - `get_destination_point`: Calculates a destination point based on origin and distance.
   - `decode_polyline`: Decodes a polyline from the Google Maps API.
   - `create_map`: Creates a map using Folium.
   - `get_census_tract_data`: Fetches census tract data.
   - `load_llama_index`: Loads the index for querying with OpenAI.
   - `fetch_census_data`: Fetches data from the U.S. Census.
   - `generate_real_estate_insights`: Generates real estate insights using OpenAI.
   - `insert_address_to_mongo`: Inserts an address into the MongoDB database.
   - `fetch_matching_addresses`: Fetches matching addresses from MongoDB.
   - `on_address_select`: Handles address selection from a dropdown.

## Conclusion

This README should provide you with the basic setup and understanding to get started with the Streamlit application. Happy coding!
