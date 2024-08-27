import json
import pickle

__locations=None
__data_columns=None
__model=None
def get_location_names():
    pass

def load_saved_artifacts():
    print("Loading saved artifacts...start") 
    global __data_columns
    global __location

    with open("./artifacts/columns.json",'r') as f:
        __data_columns=json.load(f)['data_columns']
        __locations=__data_columns[3:]
        
    with open("./artifacts/banglore_home_prices_model.pickle",'rb') as f:
        __model=pickle.load(f)

if __name__=="__main__":
    print(get_location_names())