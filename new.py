import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif
import pickle

# Load and prepare data
df = pd.read_csv(r"C:\Users\DELL\Desktop\train dataset.csv")

# Create binary target variable
df['is_attack'] = df['label'].apply(lambda x: 0 if x == 'normal' else 1)

# Encode categorical features
categorical_cols = ['protocol_type', 'service', 'flag']
label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Feature selection
X = df.drop(['is_attack', 'label'], axis=1)
y = df['is_attack']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42, 
    stratify=y
)

# Feature selection
selector = SelectKBest(f_classif, k=14)
X_train_sel = selector.fit_transform(X_train, y_train)
X_test_sel = selector.transform(X_test)

# Train model
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
model.fit(X_train_sel, y_train)

# Evaluate
y_pred = model.predict(X_test_sel)
print("Model Performance:")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Normal', 'Attack']))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Save artifacts
with open('intrusion_model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('feature_selector.pkl', 'wb') as f:
    pickle.dump(selector, f)

with open('label_encoders.pkl', 'wb') as f:
    pickle.dump(label_encoders, f)

print("\nModel artifacts saved successfully!")

# Sample prediction function
def detect_intrusion(sample_data):
    # Load artifacts
    with open('intrusion_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('feature_selector.pkl', 'rb') as f:
        selector = pickle.load(f)
    with open('label_encoders.pkl', 'rb') as f:
        label_encoders = pickle.load(f)
    
    # Create DataFrame
    sample_df = pd.DataFrame([sample_data])
    
    # Encode categorical features
    for col in categorical_cols:
        if sample_df[col].iloc[0] in label_encoders[col].classes_:
            sample_df[col] = label_encoders[col].transform([sample_df[col].iloc[0]])
        else:
            sample_df[col] = -1  # Handle unknown categories
    
    # Select features and predict
    sample_sel = selector.transform(sample_df)
    prediction = model.predict(sample_sel)
    
    return "Intrusion Detected: Yes" if prediction[0] == 1 else "Intrusion Detected: No"

# Example usage
sample_input = {
    'duration': 0,
    'protocol_type': 'tcp',
    'service': 'http',
    'flag': 'SF',
    'src_bytes': 232,
    'dst_bytes': 8153,
    'land': 0,
    'wrong_fragment': 0,
    'urgent': 0,
    'hot': 0,
    'num_failed_logins': 0,
    'logged_in': 1,
    'num_compromised': 0,
    'root_shell': 0,
    'su_attempted': 0,
    'num_root': 0,
    'num_file_creations': 0,
    'num_shells': 0,
    'num_access_files': 0,
    'num_outbound_cmds': 0,
    'is_host_login': 0,
    'is_guest_login': 0,
    'count': 5,
    'srv_count': 5,
    'serror_rate': 0.2,
    'srv_serror_rate': 0.2,
    'rerror_rate': 0,
    'srv_rerror_rate': 0,
    'same_srv_rate': 1,
    'diff_srv_rate': 0,
    'srv_diff_host_rate': 0.09,
    'dst_host_count': 30,
    'dst_host_srv_count': 255,
    'dst_host_same_srv_rate': 1,
    'dst_host_diff_srv_rate': 0,
    'dst_host_same_src_port_rate': 0.03,
    'dst_host_srv_diff_host_rate': 0.04,
    'dst_host_serror_rate': 0.03,
    'dst_host_srv_serror_rate': 0.01,
    'dst_host_rerror_rate': 0,
    'dst_host_srv_rerror_rate': 0.01
}

print("\nSample Prediction:")
print(detect_intrusion(sample_input))

# Use the sample input you provided
sample_input = {
    'duration': 0,
    'protocol_type': 'tcp',
    'service': 'netbios_ns',
    'flag': 'S0',
    'src_bytes': 0,
    'dst_bytes': 0,
    'land': 0,
    'wrong_fragment': 0,
    'urgent': 0,
    'hot': 0,
    'num_failed_logins': 0,
    'logged_in': 0,
    'num_compromised': 0,
    'root_shell': 0,
    'su_attempted': 0,
    'num_root': 0,
    'num_file_creations': 0,
    'num_shells': 0,
    'num_access_files': 0,
    'num_outbound_cmds': 0,
    'is_host_login': 0,
    'is_guest_login': 0,
    'count': 96,
    'srv_count': 16,
    'serror_rate': 1,
    'srv_serror_rate': 1,
    'rerror_rate': 0,
    'srv_rerror_rate': 0,
    'same_srv_rate': 0.17,
    'diff_srv_rate': 0.05,
    'srv_diff_host_rate': 0,
    'dst_host_count': 255,
    'dst_host_srv_count': 2,
    'dst_host_same_srv_rate': 0.01,
    'dst_host_diff_srv_rate': 0.06,
    'dst_host_same_src_port_rate': 0,
    'dst_host_srv_diff_host_rate': 0,
    'dst_host_serror_rate': 1,
    'dst_host_srv_serror_rate': 1,
    'dst_host_rerror_rate': 0,
    'dst_host_srv_rerror_rate': 0
}

# Run detection
print("\nPrediction for your sample:")
print(detect_intrusion(sample_input))