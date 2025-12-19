import mysql.connector

try:
    conn = mysql.connector.connect(
        host="mainline.proxy.rlwy.net",  # ✅ host ONLY
        port=55628,                      # ✅ port separately
        user="root",
        password="ZJgQgRkGmGkSoVqqhBnWdNPqtfCMnZsg",
        database="railway"
    )
    print("✅ Connected to  MySQL!")
    conn.close()
except mysql.connector.Error as err:
    print("❌ Connection failed:", err)
