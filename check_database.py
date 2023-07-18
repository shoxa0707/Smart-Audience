import sqlite3

def correct_len(son, num):
    free = num - len(str(son))
    rig = lef = (free) // 2
    if free % 2:
        lef += 1
    return ' '*lef + str(son) + ' '*rig
    
mydb = sqlite3.connect("smart.db", check_same_thread=False)
cursor = mydb.cursor()
cursor.execute("Select * from situation")
data = cursor.fetchall()
print('|   People   |   Men   |   Women   |          Time          |')
print('-------------------------------------------------------------')
for i in data:
    satr = '|'+correct_len(i[0], 12)+'|'+correct_len(i[1], 9)+'|'+correct_len(i[2], 11)+'|'+correct_len(i[3], 10)+'|'
    print(satr)