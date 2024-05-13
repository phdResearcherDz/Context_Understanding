import requests
import json

API_URL = 'http://localhost:11434/api'

# Function to Generate a Completion
def generate_completion(model, prompt, stream=False, format='json', images=None, options=None):
    data = {
        'model': model,
        'prompt': prompt,
        'stream': stream,
        'format': format,
        'images': images,
        'options': options
    }
    response = requests.post(f'{API_URL}/generate', json=data)
    return response.json()

# Function to Generate a Chat Completion
def generate_chat_completion(model, message, stream=False, format='json', options=None, keep_alive='30m'):
    chat_messages = [
        {
            "role": "user",
            "content": message
        }
    ]

    data = {
        'model': model,
        'messages': chat_messages,
        'stream': stream,
        'format': format,
        'options': options,
        'keep_alive': keep_alive
    }
    response = requests.post(f'{API_URL}/chat', json=data)
    return response.json()


# Function to Create a Model
def create_model(name, modelfile, path=None, stream=False):
    data = {
        'name': name,
        'modelfile': modelfile,
        'path': path,
        'stream': stream
    }
    response = requests.post(f'{API_URL}/create', json=data)
    return response.json()

# Function to List Local Models
def list_local_models():
    response = requests.get(f'{API_URL}/tags')
    return response.json()

# Function to Show Model Information
def show_model_information(name):
    data = {'name': name}
    response = requests.post(f'{API_URL}/show', json=data)
    return response.json()

# Function to Copy a Model
def copy_model(source, destination):
    data = {
        'source': source,
        'destination': destination
    }
    response = requests.post(f'{API_URL}/copy', json=data)
    return response.json()

# Function to Delete a Model
def delete_model(name):
    data = {'name': name}
    response = requests.delete(f'{API_URL}/delete', json=data)
    return response.json()

# Function to Pull a Model
def pull_model(name, stream=False):
    data = {
        'name': name,
        'stream': stream
    }
    response = requests.post(f'{API_URL}/pull', json=data)
    return response.json()

# Function to Push a Model
def push_model(name, stream=False):
    data = {
        'name': name,
        'stream': stream
    }
    response = requests.post(f'{API_URL}/push', json=data)
    return response.json()

# Function to Generate Embeddings
def generate_embeddings(model, prompt, options=None, keep_alive='5m'):
    data = {
        'model': model,
        'prompt': prompt,
        'options': options,
        'keep_alive': keep_alive
    }
    response = requests.post(f'{API_URL}/embeddings', json=data)
    return response.json()
