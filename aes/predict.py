import pickle


file = open('grid_search', 'rb') 
result = pickle.load(file)
file.close()


def predict(paragraph):
	grid_search = result['grid_search']
	return grid_search.predict(paragraph)

def main():
	paragraph = ["This is my essay. I hope you like it."]
	y_pred = predict(paragraph)
	print(y_pred)

if __name__ == "__main__":  # if run as .py script
	main()