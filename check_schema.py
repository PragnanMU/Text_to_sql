import json
import os

data = json.load(open('spider/train_spider.json'))
db_ids_in_dataset = set(e['db_id'] for e in data)

missing_schema = []
for db_id in sorted(db_ids_in_dataset):
    db_dir = f'spider/database/{db_id}'
    if os.path.exists(db_dir):
        sql_files = [f for f in os.listdir(db_dir) if f.endswith('.sql')]
        if not sql_files:
            missing_schema.append(db_id)

print(f'Total unique DBs in train_spider.json: {len(db_ids_in_dataset)}')
print(f'Missing schema SQL files: {len(missing_schema)}')
print(f'Examples: {missing_schema[:10]}')
