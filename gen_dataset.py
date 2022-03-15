import json


with open('data.json') as json_file:
    data = json.load(json_file)


train_key = 'initial_bundle'
test_key = 'test_bundle'
vehicles_key = 'Vehicles'
plants_key = 'Plants'


# classes 0 - vehicles, 1 - plants, 2 - others
def gen_data(dataset_file, data):
    vehicles_count = 0
    plants_count = 0
    others_count = 0

    with open(dataset_file, 'w') as f:
        for item in data:
            file_path = item['file']
            if not item['category']:
                # other_files.append(item['file'])
                f.write(f'{file_path} 2\n')
                others_count += 1
                continue

            if item['category']['name'] == vehicles_key:
                # vehicles_files.append(item['file'])
                f.write(f'{file_path} 0\n')
                vehicles_count += 1
                continue

            if item['category']['name'] == plants_key:
                # plants_files.append(item['file'])
                f.write(f'{file_path} 1\n')
                plants_count += 1
                continue

            # other_files.append(item['file'])
            f.write(f'{file_path} 2\n')
            others_count += 1

    print(f'File {dataset_file} , classes representation 0 : {vehicles_count}, 1 : {plants_count}, 2 : {others_count}')

gen_data('train_data.txt', data[train_key])
gen_data('test_data.txt', data[test_key])
