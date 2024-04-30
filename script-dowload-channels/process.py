import os
import json

def transform_data_to_expected_format(data):
    # Transforma los datos a un formato de lista de diccionarios con 'message' y 'timestamp'
    transformed_data = [{"message": item["message"], "timestamp": item["timestamp"]} for item in data]
    return transformed_data

def process_messages(data) -> list:
    # Procesa cada mensaje para extraer 'message' y 'date'
    processed_messages = []
    for entry in data["messages"]:
        # Verifica que tanto 'message' como 'date' existan y que 'message' no esté vacío
        if "message" in entry and "date" in entry and entry["message"].strip() != "":
            processed_messages.append({"message": entry["message"], "timestamp": entry["date"]})
    return processed_messages

def main():
    print('> Processing data...')
    # Asegúrate de que la ruta 'output/data' sea correcta y exista en tu directorio
    for folder in os.listdir('output/data'):
        print('> Processing the folder: ' + folder)
        if os.path.isdir(f'output/data/{folder}'):
            for file in os.listdir(f'output/data/{folder}'):
                if file.endswith('_messages.json'):
                    with open(f'output/data/{folder}/{file}', 'r') as f:
                        try:
                            data = json.load(f)
                            processed_messages = process_messages(data)
                            transformed_data = transform_data_to_expected_format(processed_messages)
                            output_folder = 'Detoxigram/dataset/new_dataset'
                            if not os.path.exists(output_folder):
                                os.makedirs(output_folder)
                            output_file = f'{output_folder}/{file}'
                            with open(output_file, 'w') as f_out:
                                json.dump(transformed_data, f_out, indent=4)
                            print(f'> Folder: {folder} and file: {file} processed. Output: {output_file}')
                        except json.JSONDecodeError as e:
                            print(f"Error decoding JSON from the file: {file}, Error: {str(e)}")
    print('> Done.')

if __name__ == '__main__':
    main()
