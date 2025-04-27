import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression

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

st.set_page_config(
    page_title="Real Estate App",
)



rooms = st.sidebar.number_input("How many rooms you need?", 1, 30, 1)
floor = st.sidebar.number_input(
    "Which floor are you preffer?",
    1, 20, 1,
)
total_square=st.sidebar.number_input("Which total square you need?", 30, 300, 30)

# create input DataFrame
inputDF = pd.DataFrame(
    {
        "total_square": total_square,
        "rooms": rooms,
        "floor": floor,
    },
    index=[0],
)

train_data = prepare_data()
X, y = train_data.drop("price", axis=1), train_data['price']

reg = LinearRegression()
reg.fit(X, y)

preds = reg.predict(inputDF)[0]


st.image("realestate.jpg", use_container_width=True)
st.write(f"The price of your real estate is: {round(preds)} roubles")
