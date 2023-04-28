import os
from google_images_search import GoogleImagesSearch

# Replace with your own API Key and Custom Search Engine ID
API_KEY = ''
CSE_ID = ''


def search_and_download_images(query, path, num_images=10):
    # Initialize the GoogleImagesSearch object
    gis = GoogleImagesSearch(API_KEY, CSE_ID)

    # Define search parameters
    search_params = {
        'q': query,
        'num': num_images,
        'imgSize': 'medium',
        'safe': 'off'
    }

    # Search for images
    gis.search(search_params)
    results = gis.results()

    # Create the directory if it does not exist
    if not os.path.exists(path):
        os.makedirs(path)

    # Download images
    for i, image in enumerate(results):
        try:
            image.download(os.path.join(path, f"{query}_{i + 1}.jpg"))
        except Exception as e:
            print(f"Error downloading image {i + 1}: {e}")


if __name__ == '__main__':
    search_queries = ['tiger', 'lion']
    output_folder = 'mix_images'

    for query in search_queries:
        path = os.path.join(output_folder, query)
        search_and_download_images(query, path)
