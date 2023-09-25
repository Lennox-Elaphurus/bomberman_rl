import joblib
import pickle
import sklearn

print(sklearn.__version__)
print(joblib.__version__)


model = joblib.load('./models/random_forest_model.joblib')

# Assuming 'model' is your scikit-learn model
with open('./models/random_forest_model.pkl', 'wb') as f:
    pickle.dump(model, f, protocol=4)

with open('./models/random_forest_model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)
    print(loaded_model)