import json
from datetime import datetime

def process_json(input_file, output_file):
    '''
    Requires:
        input_file (str): A string representing the path to the input JSON file.
                          This file should contain data in the specified format, with
                          a "messages" key containing message objects.
        output_file (str): A string representing the path where the output JSON file
                           will be saved. This file will contain the transformed message
                           data.

    Modifies:
        Creates or overwrites the output_file with the transformed message data.

    Returns:
        None: It writes the output directly to the specified output_file. The output JSON contains a list of dictionaries with 'message' and 'timestamp' keys extracted and transformed from the input file.
    '''
    # Read the JSON file
    with open(input_file, 'r') as file:
        data = json.load(file)

    # Extract messages and keep the last 100
    messages = data.get('messages', [])[-100:]

    # Transform data
    transformed_data = []
    for message in messages:
        if message.get('_') == 'Message':
            text = message.get('message')
            timestamp = message.get('date')

            # Convert timestamp to a standard format
            try:
                timestamp = datetime.fromisoformat(timestamp).strftime('%Y-%m-%d %H:%M:%S')
            except ValueError:
                pass  # Keep the original format if conversion fails

            transformed_data.append({'message': text, 'timestamp': timestamp})

    # Write transformed data to a new JSON file
    with open(output_file, 'w') as file:
        json.dump(transformed_data, file, indent=4)
