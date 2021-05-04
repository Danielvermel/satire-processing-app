import pymysql
import os

# Connect to the database
password = 'Will@Power_1997'

connection = pymysql.connect(host='csmysql.cs.cf.ac.uk',
                             user='c1955887',
                             password=password,
                             db='c1955887_MED',
                             charset='utf8mb4',
                             cursorclass=pymysql.cursors.DictCursor)

def create_database():
    with connection.cursor() as cur:
        sql_1 = "DROP TABLE IF EXISTS satire_table"
    
        sql_2 = """CREATE TABLE IF NOT EXISTS `satire_table` (
            `id_text` INT NOT NULL AUTO_INCREMENT PRIMARY KEY,
            `sentiment_value` VARCHAR(45) NOT NULL,
            `phrase` VARCHAR(1000) NULL);"""

        sql_3 = "ALTER TABLE satire_table CONVERT TO CHARACTER SET utf8"
        cur.execute(sql_1)
        cur.execute(sql_2)
        cur.execute(sql_3)
        connection.commit()	


def data_to_database(dataset_folder):
    list_rows = []
    
    files = os.listdir(dataset_folder)
    for filenames in files:
        f = open(os.path.join(dataset_folder, filenames),encoding="utf8")
        for line in f:
            row = (filenames, line)
            list_rows.append(row)
            
    with connection.cursor() as cur:
        sql = "INSERT INTO satire_table(sentiment_value,phrase) VALUES (%s,%s)"
        cur.executemany(sql,list_rows)
        connection.commit()	




if __name__ == '__main__':
    
    dataset = 'satire'
    create_database()
    data_to_database(dataset)
    print("done")
