import json

f = open('retirement-home.json')
data = json.load(f)

output = open('output.txt', 'w', encoding='utf-8')

for i in data:
    if(type(i) != dict):
        continue
    if(i["content"] == None or i["author"] == None):
        continue
    if(i["content"] == '' or i["author"] == ''):
        continue
    str = i["author"]
    str += ': '
    str += i["content"]
    # str += '<|endofmessage|>\n'
    str += '\n'
    output.write(str)

output.close()
f.close()
print('done')