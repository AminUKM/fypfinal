from pycaret.classification import load_model
import pandas as pd

# Load PyCaret model (no pickle)
model = load_model("fake_news_model")

test_text = "Breaking news about economy."
df = pd.DataFrame([{"text": test_text}])

try:
    pred = model.predict(df)
    print(pred)
except Exception as e:
    print("Error during prediction:", e)
