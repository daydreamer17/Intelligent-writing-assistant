import sqlite3

db = r'g:\hello-agent\hello-agents\code\my_agent\backend\data\app.db'
conn = sqlite3.connect(db)
cur = conn.cursor()
cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
print([r[0] for r in cur.fetchall()])
