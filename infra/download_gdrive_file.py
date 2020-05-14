import sys
import requests

def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)    

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    print("Downloading, this may take some time")
    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)

def main():
    argc = len(sys.argv)
    if argc != 3:
        print("Usage: %s <id_from_shareable_link> <dest_file>")
        return 1
    
    file_id = sys.argv[1]
    destination = sys.argv[2]
    download_file_from_google_drive(file_id, destination)

if __name__ == "__main__":
    main()
