"""
M505 – Predicting Data Scientist Job Postings Using Machine Learning
Student: Masoud Piryaie Yarahmadi | StudentNumber: GH1043129
Professor: Dr. Narjes Nikzad

This script:
1) Loads and cleans a dataset of job postings
2) Preprocesses features (numeric, categorical, text)
3) Trains and evaluates two MLP models:
   - Keras / TensorFlow
   - PyTorch
4) Computes metrics and plots confusion matrices
"""

# --- Imports ---
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.utils import resample

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# --- Load dataset ---
DATA_PATH = Path('Uncleaned_DS_jobs.csv')
if not DATA_PATH.exists():
    # Create demo dataset if file missing
    demo = pd.DataFrame([
        {'Job Title':'Data Scientist','Salary Estimate':'$75K-$131K (Glassdoor est.)','Job Description':'Work on ML models','Rating':'4.0','Company Name':'Acme','Location':'New York, NY','Headquarters':'New York, NY','Size':'51 to 200 employees','Founded':'2010'},
        {'Job Title':'Senior Data Engineer','Salary Estimate':'$120K-$160K (Glassdoor est.)','Job Description':'Build data pipelines','Rating':'3.8','Company Name':'BetaCorp','Location':'San Francisco, CA','Headquarters':'San Francisco, CA','Size':'501 to 1000 employees','Founded':'2005'},
        {'Job Title':'Machine Learning Engineer','Salary Estimate':np.nan,'Job Description':'Deploy models <br> and monitor','Rating':'-1','Company Name':'Gamma','Location':'Boston, MA','Headquarters':'Boston, MA','Size':'1001 to 5000 employees','Founded':'-1'},
        {'Job Title':'Data Scientist','Salary Estimate':'$90K-$150K (Glassdoor est.)','Job Description':'Analysis and modeling','Rating':'4.2','Company Name':'Delta Inc.','Location':'Newark, NJ','Headquarters':'Newark, NJ','Size':'51 to 200 employees','Founded':'2012'},
        {'Job Title':None,'Salary Estimate':'$50K-$90K (Glassdoor est.)','Job Description':'Junior role','Rating':'3.5','Company Name':None,'Location':'Remote','Headquarters':'Remote','Size':'Other (433)','Founded':'2018'}
    ])
    demo.to_csv(DATA_PATH, index=False)
    print('Demo dataset created as Uncleaned_DS_jobs.csv')

df = pd.read_csv(DATA_PATH)
print('Initial shape:', df.shape)

# --- Cleaning ---
df = df.drop_duplicates().copy()

def clean_text(x):
    if pd.isnull(x):
        return ''
    x = re.sub(r'<.*?>', '', str(x))
    return x.strip()

text_cols = ['Job Title','Job Description','Company Name','Location','Headquarters','Size']
for c in text_cols:
    if c in df.columns:
        df[c] = df[c].apply(clean_text)

def clean_salary(s):
    if pd.isnull(s) or str(s).strip()=='' or 'nan' in str(s).lower():
        return np.nan
    s = str(s).replace('(Glassdoor est.)','').replace('$','').replace('K','').replace(',','')
    s = s.replace(' - ','-').replace('—','-').replace(' to ', '-')
    try:
        low, high = s.split('-')
        return (float(low.strip()) + float(high.strip()))/2
    except:
        try:
            return float(s)
        except:
            return np.nan

if 'Salary Estimate' in df.columns:
    df['Salary'] = df['Salary Estimate'].apply(clean_salary)

for col in ['Rating','Founded']:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df.loc[df[col] < 0, col] = np.nan

if 'Size' in df.columns:
    df['Size'] = df['Size'].replace({'51 to 200 employees':'51-200','1001 to 5000 employees':'1001-5000',
                                    '5001 to 10000 employees':'5001-10000','501 to 1000 employees':'501-1000'})

if 'Location' in df.columns:
    df['City'] = df['Location'].apply(lambda x: x.split(',')[0].strip() if ',' in x else x)

df['is_data_scientist'] = df['Job Title'].str.contains('Data Scientist', case=False, na=False).astype(int)

# Drop rows without Job Title or Company Name
df = df.dropna(subset=['Job Title','Company Name'], how='any')

for col in ['Salary','Rating','Founded']:
    if col in df.columns:
        df[col] = df[col].fillna(df[col].median())

# Save cleaned dataset
clean_path = Path('Cleaned_DS_jobs.csv')
df.to_csv(clean_path, index=False)
print('Saved cleaned dataset to', clean_path)

# --- Feature Engineering ---
df_majority = df[df.is_data_scientist==0]
df_minority = df[df.is_data_scientist==1]

# Upsample minority
df_minority_upsampled = resample(df_minority, replace=True, n_samples=len(df_majority), random_state=42)
df_balanced = pd.concat([df_majority, df_minority_upsampled]).sample(frac=1, random_state=42).reset_index(drop=True)

# Text features
texts = df_balanced['Job Description'].fillna('').astype(str).values
tf = TfidfVectorizer(max_features=500, stop_words='english')
X_text_tfidf = tf.fit_transform(texts)
n_svd = min(50, X_text_tfidf.shape[1])
svd = TruncatedSVD(n_components=n_svd, random_state=42)
X_text = svd.fit_transform(X_text_tfidf)

# Numeric features
numeric_cols = [c for c in ['Salary','Rating','Founded'] if c in df_balanced.columns]
X_num = df_balanced[numeric_cols].values if numeric_cols else np.zeros((df_balanced.shape[0],0))

# Categorical features
cat_cols = []
if 'Size' in df_balanced.columns:
    cat_cols.append('Size')
if 'City' in df_balanced.columns:
    top_cities = df_balanced['City'].value_counts().index[:20].tolist()
    df_balanced['City_top'] = df_balanced['City'].apply(lambda x: x if x in top_cities else 'Other')
    cat_cols.append('City_top')

if cat_cols:
    ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    X_cat = ohe.fit_transform(df_balanced[cat_cols])
else:
    X_cat = np.zeros((df_balanced.shape[0],0))

# Combine features
X = np.hstack([X_num, X_cat, X_text])
y = df_balanced['is_data_scientist'].values

# --- Train-Test Split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale numeric features
scaler = StandardScaler()
if numeric_cols:
    X_train[:, :len(numeric_cols)] = scaler.fit_transform(X_train[:, :len(numeric_cols)])
    X_test[:, :len(numeric_cols)] = scaler.transform(X_test[:, :len(numeric_cols)])

# --- Keras MLP ---
input_dim = X_train.shape[1]
keras_model = models.Sequential([
    layers.Input(shape=(input_dim,)),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(1, activation='sigmoid')
])
keras_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
es = callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
history = keras_model.fit(X_train, y_train, validation_split=0.15, epochs=50, batch_size=32, callbacks=[es], verbose=1)
keras_pred_proba = keras_model.predict(X_test).ravel()
keras_pred = (keras_pred_proba >= 0.5).astype(int)

# --- PyTorch MLP ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class SimpleMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.reshape(-1,1), dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.reshape(-1,1), dtype=torch.float32)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# --- Define PyTorch Model ---
class SimpleMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64,1),
            nn.Sigmoid()
        )
    def forward(self,x):
        return self.net(x)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
model = SimpleMLP(X_train.shape[1]).to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# --- Training ---
epochs = 20
for epoch in range(1, epochs+1):
    model.train()
    total_loss = 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        out = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * xb.size(0)
    print(f'Epoch {epoch}/{epochs}, Loss: {total_loss/len(train_loader.dataset):.4f}')

# --- Evaluation ---
model.eval()
y_true, y_proba = [], []
with torch.no_grad():
    for xb, yb in test_loader:
        xb = xb.to(device)
        out = model(xb).cpu().numpy().ravel()
        y_true.extend(yb.numpy().ravel().tolist())
        y_proba.extend(out.tolist())
y_true = np.array(y_true)
y_proba = np.array(y_proba)
y_pred = (y_proba >= 0.5).astype(int)

# --- Metrics ---
acc = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred)
rec = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
auc = roc_auc_score(y_true, y_proba)

print("\nEvaluation Metrics:")
print(f"Accuracy: {acc:.3f}")
print(f"Precision: {prec:.3f}")
print(f"Recall: {rec:.3f}")
print(f"F1 Score: {f1:.3f}")
print(f"AUC: {auc:.3f}")

# --- Confusion Matrix ---
cm = confusion_matrix(y_true, y_pred, labels=[0,1])
disp = ConfusionMatrixDisplay(cm, display_labels=['Other','Data Scientist'])
disp.plot(cmap='Blues')
plt.title('Confusion Matrix')
plt.show()


import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# --- Use balanced dataset ---
X = np.hstack([X_num, X_cat, X_text])
y = df_balanced['is_data_scientist'].values

# --- Train-test split with stratification ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale numeric part
if len(numeric_cols) > 0:
    scaler = StandardScaler()
    X_train[:, :len(numeric_cols)] = scaler.fit_transform(X_train[:, :len(numeric_cols)])
    X_test[:, :len(numeric_cols)] = scaler.transform(X_test[:, :len(numeric_cols)])

# --- Keras Model ---
input_dim = X_train.shape[1]
model = models.Sequential([
    layers.Input(shape=(input_dim,)),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# Early stopping
es = callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model.fit(
    X_train, y_train, validation_split=0.15,
    epochs=50, batch_size=32, callbacks=[es], verbose=1
)

# --- Predictions ---
keras_pred_proba = model.predict(X_test).ravel()
keras_pred = (keras_pred_proba >= 0.5).astype(int)

# --- Plots ---
plt.figure(figsize=(12,5))

# Loss
plt.subplot(1,2,1)
plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training & Validation Loss')
plt.legend()

# Accuracy
plt.subplot(1,2,2)
plt.plot(history.history['accuracy'], label='train_acc')
plt.plot(history.history['val_accuracy'], label='val_acc')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training & Validation Accuracy')
plt.legend()

plt.show()

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

# --- Device Setup ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

# --- Define Model ---
class SimpleMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

# --- Prepare Data ---
torch.manual_seed(42)

train_ds = TensorDataset(
    torch.tensor(X_train, dtype=torch.float32),
    torch.tensor(y_train, dtype=torch.float32)
)
test_ds = TensorDataset(
    torch.tensor(X_test, dtype=torch.float32),
    torch.tensor(y_test, dtype=torch.float32)
)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

# --- Initialize Model, Loss, Optimizer ---
model_t = SimpleMLP(X_train.shape[1]).to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model_t.parameters(), lr=1e-3)

# --- Training Function ---
def train_epoch(model, loader, opt, crit):
    model.train()
    running_loss = 0.0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device).unsqueeze(1)
        opt.zero_grad()
        out = model(xb)
        loss = crit(out, yb)
        loss.backward()
        opt.step()
        running_loss += loss.item() * xb.size(0)
    return running_loss / len(loader.dataset)

# --- Evaluation Function ---
def evaluate(model, loader):
    model.eval()
    ys, yps = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            out = model(xb).cpu().numpy().ravel()
            ys.extend(yb.numpy().ravel().tolist())
            yps.extend(out.tolist())
    return np.array(ys), np.array(yps)

# --- Training Loop ---
best_loss = np.inf

for epoch in range(1, 51):
    loss = train_epoch(model_t, train_loader, optimizer, criterion)
    ys_val, yps_val = evaluate(model_t, test_loader)
    val_loss = criterion(
        torch.tensor(yps_val, dtype=torch.float32),
        torch.tensor(ys_val, dtype=torch.float32)
    ).item() if len(ys_val) > 0 else 0

    print(f'Epoch {epoch}: train_loss={loss:.4f}, val_loss={val_loss:.4f}')

    # Save best model
    if val_loss < best_loss:
        best_loss = val_loss
        torch.save(model_t.state_dict(), 'best_torch_model.pt')

# --- Load Best Model ---
model_t.load_state_dict(torch.load('best_torch_model.pt'))
y_true_t, y_proba_t = evaluate(model_t, test_loader)
torch_pred_proba = y_proba_t
torch_pred = (torch_pred_proba >= 0.5).astype(int)

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# --- Ensure y arrays are 1D ---
y_test = y_test.ravel() if y_test.ndim > 1 else y_test
y_true_t = y_true_t.ravel() if y_true_t.ndim > 1 else y_true_t

# --- Metrics computation function ---
def compute_metrics(y_true, y_pred, y_proba):
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'auc': roc_auc_score(y_true, y_proba) if len(np.unique(y_true)) > 1 else np.nan
    }

# --- Optional safeguard: ensure target exists ---
if 'is_data_scientist' not in df_balanced.columns:
    df_balanced['is_data_scientist'] = (df_balanced['Job Title']
                                        .str.contains('Data Scientist', case=False, na=False)).astype(int)

# --- Compute metrics for both models ---
keras_metrics = compute_metrics(y_test, keras_pred, keras_pred_proba)
torch_metrics = compute_metrics(y_true_t, torch_pred, torch_pred_proba)

# --- Comparison table ---
results = pd.DataFrame([keras_metrics, torch_metrics], index=['Keras_MLP','PyTorch_MLP'])
results = results.round(3)
print('Model comparison metrics:')
display(results)

# --- Confusion matrices ---
fig, axes = plt.subplots(1,2, figsize=(12,5))

cmk = confusion_matrix(y_test, keras_pred, labels=[0,1])
cmt = confusion_matrix(y_true_t, torch_pred, labels=[0,1])

sns.heatmap(cmk, annot=True, fmt='d', ax=axes[0], cmap='Blues')
axes[0].set_title('Keras Confusion Matrix')
axes[0].set_xlabel('Predicted')
axes[0].set_ylabel('True')

sns.heatmap(cmt, annot=True, fmt='d', ax=axes[1], cmap='Greens')
axes[1].set_title('PyTorch Confusion Matrix')
axes[1].set_xlabel('Predicted')
axes[1].set_ylabel('True')

plt.tight_layout()
plt.show()
