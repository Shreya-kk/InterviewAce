import sqlite3

sqlite_db = "reg.db"
output_sql_file = "reg_mysql_dump.sql"

conn = sqlite3.connect(sqlite_db)
cursor = conn.cursor()

with open(output_sql_file, "w", encoding="utf-8") as f:
    f.write("SET FOREIGN_KEY_CHECKS=0;\n")

    # Get list of tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [row[0] for row in cursor.fetchall() if row[0] != 'sqlite_sequence']

    for table in tables:
        # Write DROP + CREATE TABLE
        cursor.execute(f"SELECT sql FROM sqlite_master WHERE type='table' AND name='{table}';")
        create_stmt = cursor.fetchone()[0]
        create_stmt = create_stmt.replace("AUTOINCREMENT", "AUTO_INCREMENT")
        create_stmt = create_stmt.replace("INTEGER PRIMARY KEY", "INT PRIMARY KEY")
        create_stmt = create_stmt.replace("TEXT", "VARCHAR(255)")
        create_stmt = create_stmt.replace("TIMESTAMP DEFAULT CURRENT_TIMESTAMP", "DATETIME DEFAULT CURRENT_TIMESTAMP")
        f.write(f"\nDROP TABLE IF EXISTS `{table}`;\n")
        f.write(create_stmt + ";\n")

        # Dump data
        cursor.execute(f"SELECT * FROM {table}")
        rows = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]

        for row in rows:
            values = []
            for value in row:
                if value is None:
                    values.append("NULL")
                else:
                    # Escape single quotes by doubling them
                    escaped_value = str(value).replace("'", "''")
                    values.append(f"'{escaped_value}'")
            f.write(f"INSERT INTO `{table}` ({', '.join(columns)}) VALUES ({', '.join(values)});\n")

    f.write("SET FOREIGN_KEY_CHECKS=1;\n")

conn.close()

print(f"✅ Exported to {output_sql_file}")
