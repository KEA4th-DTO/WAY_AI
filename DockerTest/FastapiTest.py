from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def printHello():
	return "Hello World"

@app.get("/json")
def printJson():
	return {
		"Number" : 12345
	}