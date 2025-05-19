from flask import Flask, render_template, request, jsonify
import pandas as pd
import pickle
import os

app = Flask(__name__)

# Load the saved artifacts
def load_artifacts():
    with open('intrusion_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('feature_selector.pkl', 'rb') as f:
        selector = pickle.load(f)
    with open('label_encoders.pkl', 'rb') as f:
        label_encoders = pickle.load(f)
    return model, selector, label_encoders

# Load artifacts at startup
model, selector, label_encoders = load_artifacts()

@app.route('/')
def home():
    return render_template('welcome.html')

@app.route('/input')
def input_form():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        data = request.form.to_dict()
        
        # Define the expected feature order
        feature_order = [
            'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
            'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
            'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
            'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login',
            'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
            'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
            'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
            'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
            'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
            'dst_host_serror_rate', 'dst_host_srv_serror_rate',
            'dst_host_rerror_rate', 'dst_host_srv_rerror_rate'
        ]
        
        # Create DataFrame with ordered features
        ordered_data = {feature: data[feature] for feature in feature_order}
        sample_df = pd.DataFrame([ordered_data])
        
        # Convert numeric columns
        numeric_columns = [
            'duration', 'src_bytes', 'dst_bytes', 'land', 'wrong_fragment', 
            'urgent', 'hot', 'num_failed_logins', 'logged_in', 'num_compromised',
            'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
            'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login',
            'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
            'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
            'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
            'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
            'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
            'dst_host_serror_rate', 'dst_host_srv_serror_rate',
            'dst_host_rerror_rate', 'dst_host_srv_rerror_rate'
        ]
        
        for col in numeric_columns:
            sample_df[col] = pd.to_numeric(sample_df[col])
        
        # Encode categorical features
        categorical_cols = ['protocol_type', 'service', 'flag']
        for col in categorical_cols:
            if sample_df[col].iloc[0] in label_encoders[col].classes_:
                sample_df[col] = label_encoders[col].transform([sample_df[col].iloc[0]])
            else:
                sample_df[col] = -1
        
        # Select features and predict
        sample_sel = selector.transform(sample_df)
        prediction = model.predict(sample_sel)
        
        result = "Intrusion Detected: Yes" if prediction[0] == 1 else "Intrusion Detected: No"
        return jsonify({'prediction': result})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True) 