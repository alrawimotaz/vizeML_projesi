import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import math
import joblib
import streamlit as st
import matplotlib.pyplot as plt

# Veri setini oku
file_path = "../veritoplama/processed_matches.csv"
df = pd.read_csv(file_path)

# "team_B" sütununu Label Encoding yap
label_encoder = LabelEncoder()
df['team_B_encoded'] = label_encoder.fit_transform(df['team_B'])

# Load the trained model
best_model = joblib.load('../algoritmalar/model.pkl')

# Streamlit arayüzü
st.title("Real Madrid Win Prediction Model")

# Kullanıcıdan takım seçimini al
selected_team = st.selectbox("Select a team:", df['team_B'].unique())

# Label Encoding uygula
selected_team_encoded = label_encoder.transform([selected_team])

# Tahmin olasılıklarını al
probabilities = best_model.predict_proba([[selected_team_encoded[0]]])[0]

# Olasılıkları göster
st.write(f"{selected_team} Team Next Match Probabiliteis:")
st.write(f"Probability of Real Madrid winning: {probabilities[1]:.2%}")
st.write(f"Draw Probability: {probabilities[2]:.2%}")
st.write(f"Probability of {selected_team} winning: {probabilities[0]:.2%}")

# Maç istatistikleri
team_A_wins = len(df[(df['team_B'] == selected_team) & (df['score_A'] > df['score_B'])])
team_B_wins = len(df[(df['team_B'] == selected_team) & (df['score_A'] < df['score_B'])])
draws = len(df[(df['team_B'] == selected_team) & (df['score_A'] == df['score_B'])])

# Maç istatistikleri
st.subheader("Match Statistics")
st.write(f"Real Madrid Wins: {team_A_wins} times")
st.write(f"{selected_team} Wins: {team_B_wins} times")
st.write(f"Draws: {draws} times")

# Sidebar menü
st.sidebar.header('Graph Menu')
st.sidebar.title('Statistics Chart')

# Calculate the maximum probability for the selected team's future match outcome
prediction_probability = max(probabilities)

# Sidebar'da çubuk grafik gösterimi
with st.sidebar.expander('Match Statistics'):
    fig, ax = plt.subplots()
    labels = ['Real Madrid Wins', f'{selected_team} Wins', 'Draws']
    values = [team_A_wins, team_B_wins, draws]

    ax.bar(labels, values)
    ax.set_ylabel('Number of Occurrences')
    ax.set_title('Match Statistics')

    st.pyplot(fig)

# Sidebar menü
st.sidebar.title('Prediction Chart')

# Sidebar'da pasta grafiği gösterimi
with st.sidebar.expander('Match Prediction'):
    labels = ['Real Madrid Wins', f'{selected_team} Wins', 'Draws']
    sizes = [probabilities[1], probabilities[0], probabilities[2]]
    colors = ['lightcoral', 'lightskyblue', 'lightgreen']
    explode = (0.1, 0, 0.1)

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=140)
    ax.axis('equal')
    ax.set_title('Match Prediction')

    st.pyplot(fig)



    ###------  modeli * https://matchresultguess.streamlit.app *  bu sitede diploy ettim -----###
