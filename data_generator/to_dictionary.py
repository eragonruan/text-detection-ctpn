

def to_dictionary(text_path='', code='utf-8'):
	with open(text_path, 'rb') as file:
		info_list = [part.decode(code, 'ignore').strip() for part in file.readlines()]
		string = ''.join(info_list)
		setting = set(string)
		dictionary = {key : value for key, value in enumerate(setting)}

	return dictionary

if __name__ == '__main__':
	#to_dictionary('')
	list_a = [3,4,5,8]
	list_b = [3,4,5,6,7]
	set_c = set(list_a) & set(list_b)
	list_c = list(set_c)
	print(list_c)
	set_a = set(list_a)
	set_b = set(list_b)
	set_a.difference_update(set_b)
	print(set_a)