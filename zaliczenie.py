import pandas as pd
import datetime
import csv
import random
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from rapidfuzz import process

# Wczytanie danych z pliku imdbpl.csv
df = pd.read_csv("imdbpl.csv")

feature_cols = [
    'imdbRating', 'ratingCount', 'duration', 'year',
] + [
    'Action','Adult','Adventure','Animation','Biography','Comedy','Crime','Documentary','Drama',
    'Family','Fantasy','FilmNoir','GameShow','History','Horror','Music','Musical','Mystery',
    'News','RealityTV','Romance','SciFi','Short','Sport','TalkShow','Thriller','War','Western'
]

df_clean = df.dropna(subset=['title'] + feature_cols).reset_index(drop=True)
X = df_clean[feature_cols]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

knn = NearestNeighbors(n_neighbors=6, metric='euclidean')
knn.fit(X_scaled)

# Filtry
min_rating = 8.0
genre_filter = None

def get_genres(row):
    return ', '.join([g for g in feature_cols[4:] if row[g] == 1])

def find_best_match(title_input):
    titles = df_clean['title'].tolist()
    match_data = process.extractOne(title_input, titles)
    if match_data:
        match, score, _ = match_data
        if score < 85:
            print(f"⚠️ Znaleziono podobny tytuł: {match} (trafność: {score}%)")
            confirm = input("Czy chodziło Ci o ten film? (t/n): ").strip().lower()
            if confirm != 't':
                return None
        return match
    return None

def save_history_to_file():
    if not user_history_log:
        return
    with open("historia_uzytkownika.csv", "a", newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        for record in user_history_log:
            writer.writerow(record)

def surprise_me():
    candidates = df_clean[df_clean['imdbRating'] >= min_rating]
    if genre_filter:
        candidates = candidates[candidates[genre_filter] == 1]
    candidates = candidates[~candidates['title'].isin(user_history)]  # unikaj powtórzeń

    if candidates.empty:
        print("❗ Brak filmów do zaproponowania w trybie 'surprise me'.")
        return

    film = candidates.sample(1).iloc[0]
    print("\n🎁 Niespodzianka! Spróbuj tego filmu:")
    print(f"👉 {film['title']} (rok: {int(film['year'])}, ocena: {film['imdbRating']}, gatunki: {get_genres(film)})")

    # dodaj do logu historii
    user_history.append(film['title'])
    user_history_log.append([film['title'], datetime.datetime.now().isoformat(), 'surprise'])

def matches_filters(film):
    global min_rating, genre_filter
    if film['imdbRating'] < min_rating:
        return False
    if genre_filter:
        return film[genre_filter] == 1
    return True

def recommend_similar(title_input):
    title_match = find_best_match(title_input)
    if not title_match:
        print(f"❌ Nie znaleziono filmu podobnego do: {title_input}")
        return
    idx = df_clean[df_clean['title'] == title_match].index[0]
    distances, indices = knn.kneighbors([X_scaled[idx]])
    print(f"\n🎬 Rekomendacje dla: {title_match} (ocena: {df_clean.iloc[idx]['imdbRating']})")
    found = False
    for i in indices[0][1:]:
        film = df_clean.iloc[i]
        if matches_filters(film):
            print(f"👉 {film['title']} (rok: {int(film['year'])}, ocena: {film['imdbRating']}, gatunki: {get_genres(film)})")
            found = True
    if not found:
        print("❗ Brak rekomendacji spełniających aktywne filtry.")

def recommend_from_history():
    if not user_history:
        print("📭 Brak historii wyszukiwań.")
        return
    print(f"\n📚 Twoja historia: {', '.join(user_history)}")
    user_vectors = []
    for title in user_history:
        idx = df_clean[df_clean['title'] == title].index[0]
        user_vectors.append(X_scaled[idx])
    import numpy as np
    avg_vector = np.mean(user_vectors, axis=0).reshape(1, -1)
    distances, indices = knn.kneighbors(avg_vector)
    shown = set(user_history)
    found = False
    print("\n🎯 Rekomendacje na podstawie historii:")
    for i in indices[0]:
        film = df_clean.iloc[i]
        if film['title'] not in shown and matches_filters(film):
            print(f"👉 {film['title']} (rok: {int(film['year'])}, ocena: {film['imdbRating']}, gatunki: {get_genres(film)})")
            shown.add(film['title'])
            found = True
    if not found:
        print("❗ Brak rekomendacji spełniających aktywne filtry.")

def list_genres():
    print("\n🎭 Dostępne gatunki:")
    for g in feature_cols[4:]:
        print(f"• {g}")

def interactive_loop():
    global min_rating, genre_filter
    print("\n🟢 Witaj w systemie rekomendacji filmów!")
    print("Wpisz nazwę filmu, lub komendę:")
    print("🔸 recommend — rekomenduj na podstawie historii")
    print("🔸 surprise me — zaproponuj film losowo")
    print("🔸 minrating [ocena] — ustaw minimalną ocenę (np. minrating 7.5)")
    print("🔸 filter genre [GATUNEK] — filtruj tylko po gatunku (np. Drama)")
    print("🔸 clear filters — usuń wszystkie filtry")
    print("🔸 genres — pokaż dostępne gatunki")
    print("🔸 exit — zakończ")

    while True:
        inp = input("\n📝 Komenda lub tytuł: ").strip()
        if inp.lower() == "exit":
            save_history_to_file()
            print("📁 Historia zapisana do: historia_uzytkownika.csv")
            break
        elif inp.lower() == "recommend":
            recommend_from_history()
        elif inp.lower().startswith("minrating"):
            try:
                min_rating = float(inp.split()[1])
                print(f"📊 Ustawiono minimalną ocenę: {min_rating}")
            except:
                print("❗ Użycie: minrating [liczba]")
        elif inp.lower().startswith("filter genre"):
            try:
                genre_input = inp.split(" ", 2)[2].strip()
                if genre_input in feature_cols:
                    genre_filter = genre_input
                    print(f"🎭 Filtr gatunku ustawiony na: {genre_filter}")
                else:
                    print("❗ Nieprawidłowy gatunek.")
            except:
                print("❗ Użycie: filter genre [nazwa gatunku]")
        elif inp.lower() == "clear filters":
            min_rating = 0.0
            genre_filter = None
            print("🧹 Filtry wyczyszczone.")
        elif inp.lower() == "genres":
            list_genres()
        elif inp.lower() in ["surprise me", "suprise me", "surprize me"]:
            surprise_me()
        else:
            match = find_best_match(inp)
            if match:
                user_history.append(match)
                user_history_log.append([match, datetime.datetime.now().isoformat(), 'search'])
                recommend_similar(match)
            else:
                print("❌ Nie znaleziono filmu.")

# Historia użytkownika
user_history_log = []
user_history = []

# Test początkowy
print("\n🔧 Testowe rekomendacje:")
for t in ["Toy Story", "Batman Begins", "Star Wars Episode III"]:
    recommend_similar(t)

# Interaktywny interfejs
interactive_loop()
