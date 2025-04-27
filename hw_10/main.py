import uvicorn
import pandas as pd
from sklearn.linear_model import LinearRegression
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from fastapi.responses import PlainTextResponse

def prepare_data():
    train = pd.read_csv("realty_data.csv")

    train["price"] = train["price"].fillna(0)
    train.loc[train["rooms"].isnull(), "rooms"] = 0
    train.loc[train["floor"].isnull(), "floor"] = 1

    train['rooms'] = train['rooms'].astype(int)
    train['floor'] = train['floor'].astype(int)


    train = train.drop(columns=["product_name", 'period', 'postcode', 'address_name', 'object_type','city','settlement','district','area','description','source','lat','lon'])


    train = pd.get_dummies(train, drop_first=False)

    return train

def predictjob(pdf):
    train_data = prepare_data()
    X, y = train_data.drop("price", axis=1), train_data['price']
    reg = LinearRegression()
    reg.fit(X, y)
    return reg.predict(pdf)[0]



app = FastAPI()

class ModelRequestData(BaseModel):
    total_square: float
    rooms: int
    floor: int


class Result(BaseModel):
    result: float

@app.get("/health")
def health():
    return JSONResponse(content={"message": "System is normal"}, status_code=200)

@app.post("/predict_post", response_model=Result)
def preprocess_data(data: ModelRequestData):
    input_data = data.dict()
    input_df = pd.DataFrame(input_data, index=[0])
    result = round(predictjob(input_df))
    return Result(result=result)

@app.get("/predict_get")
def preprocess_data_get(total_square, rooms, floor):
    input_df = pd.DataFrame(
    {
        "total_square": total_square,
        "rooms": rooms,
        "floor": floor,
    },
    index=[0],
    )

    result = round(predictjob(input_df))
    return {"The price of real estate is ":result}

if __name__ == "__main__":
    #train_data = prepare_data()
    #X, y = train_data.drop("price", axis=1), train_data['price']


    #reg = LinearRegression()
    #reg.fit(X, y)
    uvicorn.run(app, host="0.0.0.0", port=8000)

#preds = reg.predict(inputDF)[0]


