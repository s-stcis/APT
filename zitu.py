import json
import math
import os
import csv

time_slot = 0.006
time_forgot = 0.003


def flatten_dict(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)



def transform(time):
    if time[1] == ':':
        hour = float(time[0:1])
        if time[3] == ':':
            minute = float(time[2:3])
            second = float(time[4:len(time)])
        else:
            minute = float(time[2:4])
            second = float(time[5:len(time)])
    else:
        hour = float(time[0:2])
        if time[4] == ':':
            minute = float(time[3:4])
            second = float(time[5:len(time)])
        else:
            minute = float(time[3:5])
            second = float(time[6:len(time)])
    return 3600 * hour + 60 * minute + second


with open("./new.json", "r", encoding="utf-8") as f:
    json_file = []
    time_relative = []
    json_content = f.read()
    lines = json_content.split('\n')  # 将内容按行分割
    for line in lines:
        if line.strip():  # 忽略空行
            content = json.loads(line)
            json_file.append(content)
            time_relative.append(transform(content["edge"]["time"])-transform(json_file[0]["edge"]["time"]))


number = math.ceil((time_relative[len(time_relative)-1] - time_slot) / time_forgot + 1)
number_file = []
for i in range(number) :
    json_name = './data/json/' + str(i) + '.json'
    csv_name = './data/csv/' + str(i) + '.csv'
    json_new_file = open(json_name, "w")
    time_min = i * time_forgot
    time_max = time_min + time_slot
    number_init = 0
    for j in range(len(json_file)) :
        if time_relative[j] < time_max and time_relative[j] >= time_min :
            number_init = number_init + 1
            json.dump(json_file[j], json_new_file)
            json_new_file.writelines("\n")
    if number_init == 0 :
        json_new_file.close()
        os.remove(json_name)
    else :
        json_new_file.close()
        # 初始化一个空列表来存储所有的数据
        all_data = []
        # 读取并解析json文件中的每一行
        with open(json_name, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                flat_data = flatten_dict(data)
                all_data.append(flat_data)
        # 获取所有数据的所有字段（即，csv文件的表头）
        fields = set().union(*all_data)
        # 写入csv文件
        with open(csv_name, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for row in all_data:
                writer.writerow(row)
    json_new_file.close()




