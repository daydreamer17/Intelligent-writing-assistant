import sqlite3

db = r'g:\hello-agent\hello-agents\code\my_agent\backend\data\app.db'
conn = sqlite3.connect(db)
cur = conn.cursor()
cur.execute("SELECT doc_id, title, substr(content,1,120) FROM documents ORDER BY rowid DESC LIMIT 20")
rows=cur.fetchall()
for r in rows:
    print('---')
    print(r[0])
    print(r[1])
    print(r[2].replace('\n',' '))
