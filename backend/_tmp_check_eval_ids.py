import json, sqlite3

db = r'g:\hello-agent\hello-agents\code\my_agent\backend\data\app.db'
js = r'g:\hello-agent\hello-agents\code\my_agent\backend\test_offline.json'

with open(js, 'r', encoding='utf-8') as f:
    data = json.load(f)

conn = sqlite3.connect(db)
cur = conn.cursor()
cur.execute('select doc_id from documents')
ids = {r[0] for r in cur.fetchall()}

missing = []
for case in data.get('cases', []):
    for rid in case.get('relevant_doc_ids', []):
        if rid not in ids:
            missing.append((case.get('query_id'), rid))

print('cases=', len(data.get('cases', [])))
print('db_docs=', len(ids))
print('missing=', len(missing))
if missing:
    print(missing[:10])
