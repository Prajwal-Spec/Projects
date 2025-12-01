
import os
import joblib
import pandas as pd

def model_fn(model_dir):
    """Load and return the model"""
    # CRITICAL FIX: Matches the filename used in train.py
    model_file_name = "model.joblib"
    pipeline_model = joblib.load(os.path.join(model_dir, model_file_name))
    
    return pipeline_model

def input_fn(request_body, request_content_type):
    """Process the input json data"""
    if request_content_type == "application/json":
        # Reads JSON Lines (e.g., {"feat1": 1}\n{"feat1": 2})
        input_object = pd.read_json(request_body, lines=True)
        return input_object
    else:
        raise ValueError(f"Only application/json content type supported! Received: {request_content_type}")

def predict_fn(input_object, pipeline_model):
    """Make predictions on processed input data"""
    predictions = pipeline_model.predict(input_object)
    pred_probs = pipeline_model.predict_proba(input_object)
    
    prediction_object = pd.DataFrame(
        {
            "prediction": predictions.tolist(),
            "pred_prob_class0": pred_probs[:, 0].tolist(),
            "pred_prob_class1": pred_probs[:, 1].tolist()
        }
    )
    
    return prediction_object

def output_fn(prediction_object, request_content_type):
    """Post process the predictions and return as json"""
    # Return as JSON Lines to match input style
    return_object = prediction_object.to_json(orient="records", lines=True)
    
    return return_object
