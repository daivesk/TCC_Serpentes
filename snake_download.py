from pygbif import occurrences
import requests
import os

def fetch_snake_images(species, primary_country, limit=50):
    print(f"Fetching image URLs for {species} in {primary_country}...")
    primary_data = occurrences.search(scientificName=species, country=primary_country,
                                      limit=limit, mediaType='StillImage')
    primary_results = primary_data.get('results')
    image_urls = set()

    for result in primary_results:
        if 'media' in result:
            for media in result['media']:
                if 'identifier' in media:
                    image_urls.add(media['identifier'])

    if len(image_urls) < limit:
        print(f"Found {len(image_urls)} images in {primary_country}, searching globally for more...")
        global_data = occurrences.search(scientificName=species,
                                         limit=limit - len(image_urls), mediaType='StillImage')
        global_results = global_data.get('results')

        for result in global_results:
            if 'media' in result:
                for media in result['media']:
                    if 'identifier' in media:
                        image_urls.add(media['identifier'])

    print(f"Total {len(image_urls)} image URLs found for {species}.")
    return list(image_urls)[:limit]

def download_images(image_urls, species_name, parent_directory):
    species_folder = os.path.join(parent_directory, species_name.replace(' ', '_'))
    if not os.path.exists(species_folder):
        os.makedirs(species_folder)
        print(f"Created directory: {species_folder}")

    for i, url in enumerate(image_urls):
        try:
            response = requests.get(url)
            if response.status_code == 200:
                image_path = os.path.join(species_folder, f'{species_name}_{i}.jpg')
                with open(image_path, 'wb') as f:
                    f.write(response.content)
                print(f"Downloaded image {i+1} for {species_name}.")
        except Exception as e:
            print(f"Error downloading {url}: {e}")

def read_species_from_file(file_path):
    with open(file_path, 'r') as file:
        species_names = [line.strip() for line in file.readlines()]
    print(f"Read {len(species_names)} species names from the file.")
    return species_names

# Define the parent directory
parent_directory = 'images/'

# Ensure parent directory exists
if not os.path.exists(parent_directory):
    os.makedirs(parent_directory)
    print(f"Created parent directory: {parent_directory}")

# File path to the Markdown file with species names
file_path = 'lista de cobras do brasil.md'

# Read species names from the file
species_names = read_species_from_file(file_path)

# Iterate over each species and download images
for species_name in species_names:
    image_urls = fetch_snake_images(species_name, 'BR', limit=50)
    download_images(image_urls, species_name, parent_directory)
