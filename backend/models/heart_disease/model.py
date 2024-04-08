import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import torch
import warnings
from torch import nn, Tensor
from torch.optim import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report


def plot_data():
    """This function plots all of the different aspects of this dataset"""
    # Load the dataset
    data = pd.read_csv('heart_disease_dataset.csv')
        
    # Initialize counters for male and female patients
    y = data['HeartDisease'].values
    gender = data['Sex']
    male_count = 0
    female_count = 0
    male_diagnosis_count = 0
    female_diagnosis_count = 0
    
    if len(gender) == len(y):
        for i in range(len(gender)):
            if gender[i] == 1:
                male_count += 1
                if y[i] == 1:
                    male_diagnosis_count += 1
            else:
                female_count += 1
                if y[i] == 1:
                    
                    female_diagnosis_count += 1
    else: 
        assert (len(gender) == len(y)), f'The number of heart disease patients and diagnosis count are not the same.'
        
    # Print the counts
    print(f'Male Count: {male_count}, Male Diagnosis Count: {male_diagnosis_count}. Thats {(male_diagnosis_count / male_count * 100):.1f}% of male patients')
    print(f'Female Count: {female_count}, Female Diagnosis Count: {female_diagnosis_count}. Thats {(female_diagnosis_count / female_count * 100):.1f}% of female patients.')
    
    categories = ['Male', 'Female']
    counts = [male_count, female_count]
    diagnosis_counts = [male_diagnosis_count, female_diagnosis_count]

    # Plot
    plt.figure(figsize=(10, 6))

    plt.bar(categories, counts, color='blue', label='Total Patients')
    plt.bar(categories, diagnosis_counts, color='red', label='Patients with Diagnosis')

    plt.xlabel('Gender')
    plt.ylabel('Count')
    plt.title('Heart Disease Patients by Gender')
    plt.legend()

    plt.show()
    
    # Filter data for males and females
    male_data = data[data['Sex'] == 1]
    female_data = data[data['Sex'] == 0]

    # Selecting features for visualization
    features = ['Age', 'RestingBP', 'Cholesterol']

    # Plotting for males
    plt.figure(figsize=(12, 6))
    plt.title('Data Visualization for Males')
    for feature in features:
        plt.plot(male_data.index, male_data[feature], label=feature)
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plotting for females
    plt.figure(figsize=(12, 6))
    plt.title('Data Visualization for Females')
    for feature in features:
        plt.plot(female_data.index, female_data[feature], label=feature)
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Set style and context for seaborn
    sns.set(style="ticks")

    # Create pair plot for males
    sns.pairplot(male_data, vars=['Cholesterol', 'HeartDisease'], hue='HeartDisease', diag_kind='kde', markers=['o', 's'], palette='husl')
    plt.title('Pair Plot for Cholesterol vs Diagnosis (Males)')
    plt.tight_layout()
    plt.show()

    # Create pair plot for females
    sns.pairplot(female_data, vars=['Cholesterol', 'HeartDisease'], hue='HeartDisease', diag_kind='kde', markers=['o', 's'], palette='husl')
    plt.title('Pair Plot for Cholesterol vs Diagnosis (Females)')
    plt.show()
    
    # Separate data for male and female
    male_data = data[data['Sex'] == 1]
    female_data = data[data['Sex'] == 0]

    # Create subplots with 1 row and 2 columns
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    # Plot for males
    male_heart_disease_color = male_data['HeartDisease'].apply(lambda x: 'red' if x == 1 else 'blue')
    axs[0].scatter(male_data['Age'], male_data['Cholesterol'], color=male_heart_disease_color, alpha=0.5)
    axs[0].set_title('Male')

    # Plot for females
    female_heart_disease_color = female_data['HeartDisease'].apply(lambda x: 'red' if x == 1 else 'blue')
    axs[1].scatter(female_data['Age'], female_data['Cholesterol'], color=female_heart_disease_color, alpha=0.5)
    axs[1].set_title('Female')

    # Add common labels
    for ax in axs:
        ax.set_xlabel('Patient Age')
        ax.set_ylabel('Cholesterol Level')

    # Show plot
    plt.tight_layout()
    plt.show()
    
    # Visualize data before training (e.g., histogram of a feature)
    # plt.figure(figsize=(8, 6))
    # plt.hist(data['Age'], bins=20, color='blue', alpha=0.7)
    # plt.xlabel('Age')
    # plt.ylabel('Frequency')
    # plt.title('Distribution of Age')
    # plt.show()

    # # Visualize training process (e.g., plot training loss over epochs)
    # plt.figure(figsize=(8, 6))
    # plt.plot(model.losses, label='Training Loss')
    # plt.xlabel('Epochs')
    # plt.ylabel('Loss')
    # plt.title('Training Loss over Epochs')
    # plt.legend()
    # plt.show()

    # # Histogram of numerical features
    # data.hist()
    
    # # Heat map
    # plt.figure(figsize=(10, 8))
    # sns.heatmap(data.corr(), annot=True, cmap='coolwarm', fmt=".2f")
    # plt.title("Correlation Heatmap")
    # plt.show()   


class HeartDiseaseModel(nn.Module):
    def __init__(self, input_dim=10):
        super(HeartDiseaseModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
            # nn.Linear(16, 1),
            # nn.Sigmoid(),
        )
        self.optimizer = Adam(self.parameters(), lr=0.001)
        self.loss_fn = nn.BCELoss()
        self.losses = []
        
    def forward(self, x):
        return self.model(x)
    
    def fit(self, x, y, epochs):
        x_tensor = torch.tensor(x, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)
        for epoch in range(epochs):
            self.train()
            self.optimizer.zero_grad()
            outputs = self.forward(x_tensor)
            loss = self.loss_fn(outputs, y_tensor)
            loss.backward()
            self.optimizer.step()
            self.losses.append(loss.item()) 
            if epoch % 100 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.6f}')
                
    def save_model(self, filename):
        torch.save(self.state_dict(), filename)
        
    def load_model(self, filename):
        self.load_state_dict(torch.load(filename))
        self.eval()
        
    def predict(self, x):
        with torch.no_grad():
            x_tensor = torch.tensor(x, dtype=torch.float32)
            self.eval()
            model = self.load_model('heart_disease_model.pth')
            predictions = self.model(x_tensor)
            return predictions
        

def test_prediction(fil):
    data = pd.read_csv('test_data.csv')
    columns = ['HeartDisease']
    
    x = data.drop(columns=columns).values
    y = data['HeartDisease'].values
    
    model = HeartDiseaseModel()
    scaler = StandardScaler()
    
    x = scaler.fit_transform(x)
    
    predictions = model.predict(x)
    predictions = predictions.round().flatten()
    predictions = (predictions.numpy() > 0.5).astype(int)
    
    # Evaluate the model
    accuracy = accuracy_score(y, predictions)
    print(classification_report(y, predictions))
    print(f'Accuracy: {accuracy * 100:.1f}%')
    

def main(*args, **kwargs):
    # Load the dataset
    data = pd.read_csv('heart_disease_dataset.csv')
    columns = ['HeartDisease']
    
    x = data.drop(columns=columns).values
    y = data['HeartDisease'].values
    
    # Split the data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # Standardize features
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    # Instantiate the model
    shape = x_train.shape[1]
    model = HeartDiseaseModel()

    # Train the model
    loss = model.fit(x_train, y_train, epochs=2500)

    # Save the model
    model.save_model('heart_disease_model.pth')
    
    # Evaluate the model
    predictions = model.predict(x_test)
    predictions = predictions.round().flatten() # Round predictions to 0 or 1 
    
    # Convert predictions to binary (0 or 1)
    predictions = (predictions.numpy() > 0.5).astype(int)
    
    # Classification report
    print(classification_report(y_test, predictions))
    accuracy = accuracy_score(y_test, predictions)
    print(f'Accuracy: {accuracy * 100:.1f}%')
    
    # Convert the data into DMatrix format for XGBoost
    dtrain = xgb.DMatrix(x_train, label=y_train)
    dtest = xgb.DMatrix(x_test, label=y_test)

    # Define parameters for XGBoost
    param = {
        'max_depth': 8,
        'eta': 0.5,
        'objective': 'binary:logistic',
        'eval_metric': 'logloss'
    }

    # Train the XGBoost model
    num_round = 120
    model = xgb.train(param, dtrain, num_round)
    
    # Predict on the test set
    y_pred_proba = model.predict(dtest)
    y_pred = np.round(y_pred_proba)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(classification_report(y_test, y_pred))
    print(f'Accuracy: {accuracy * 100:.1f}%')
    
    # Calculate accuracy
    accuracy = (predictions == y_test).mean()
    # print(f'Accuracy of testing outputs vs. expected output: {accuracy * 100:.2f}%')
    

if __name__ == '__main__':
    # main()
    test_prediction()
    
    # prepare_ACG()
    # plot_data()