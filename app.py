from flask import Flask, render_template, url_for, request
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
import numpy as np
from sklearn.neighbors import NearestNeighbors

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/svdIndex')
def svdIndex():
    return render_template('svd.html')

class Books:
    def __init__(self):
        # Reading CSV files
        try:
            self.books = pd.read_csv('./Book/Books.csv')
            self.users = pd.read_csv('./Book/Users.csv')
            self.ratings = pd.read_csv('./Book/Ratings.csv')
            print("Files successfully loaded.")
        except Exception as e:
            print(f"Error loading files: {e}")
            raise

        # Splitting Explicit and Implicit user ratings
        self.ratings_explicit = self.ratings[self.ratings.bookRating != 0]
        self.ratings_implicit = self.ratings[self.ratings.bookRating == 0]

        # Each Book's Mean ratings and Total Rating Count
        self.average_rating = pd.DataFrame(
            self.ratings_explicit.groupby('ISBN')['bookRating'].mean())
        self.average_rating['ratingCount'] = pd.DataFrame(
            self.ratings_explicit.groupby('ISBN')['bookRating'].count())
        self.average_rating = self.average_rating.rename(
            columns={'bookRating': 'MeanRating'})

        # Filter users who have rated at least 50 books
        counts1 = self.ratings_explicit['userID'].value_counts()
        self.ratings_explicit = self.ratings_explicit[
            self.ratings_explicit['userID'].isin(counts1[counts1 >= 50].index)]

        # Explicit Books and ISBN
        self.explicit_ISBN = self.ratings_explicit.ISBN.unique()
        self.explicit_books = self.books.loc[self.books['ISBN'].isin(
            self.explicit_ISBN)]

        # Look up dict for Book and BookID
        def normalize_title(title):
            return re.sub(r'\s+', ' ', re.sub(r'[^\w\s]', '', title)).strip().lower()

        self.Book_lookup = dict(
            zip(self.explicit_books["ISBN"], self.explicit_books["bookTitle"]))
        self.ID_lookup = dict(
            zip(self.explicit_books["bookTitle"].str.lower(), self.explicit_books["ISBN"]))

    def Top_Books(self, n=10, RatingCount=100, MeanRating=3):
        BOOKS = self.books.merge(self.average_rating, how='right', on='ISBN')
        M_Rating = BOOKS.loc[BOOKS.ratingCount >= RatingCount].sort_values(
            'MeanRating', ascending=False).head(n)
        H_Rating = BOOKS.loc[BOOKS.MeanRating >= MeanRating].sort_values(
            'ratingCount', ascending=False).head(n)
        return M_Rating, H_Rating

class KNN(Books):
    def __init__(self, n_neighbors=5):
        super().__init__()
        self.n_neighbors = n_neighbors
        self.ratings_mat = self.ratings_explicit.pivot(
            index="ISBN", columns="userID", values="bookRating").fillna(0)
        self.uti_mat = csr_matrix(self.ratings_mat.values)
        self.model_knn = NearestNeighbors(metric='cosine', algorithm='brute')
        self.model_knn.fit(self.uti_mat)

    def Recommend_Books(self, book, n_neighbors=5):
        book = book.lower()
        if book not in self.ID_lookup:
            return f"Error: The book '{book}' was not found in the dataset.", None, None

        bID = self.ID_lookup[book]
        query_index = self.ratings_mat.index.get_loc(bID)
        KN = self.ratings_mat.iloc[query_index, :].values.reshape(1, -1)
        distances, indices = self.model_knn.kneighbors(
            KN, n_neighbors=n_neighbors + 1)

        Rec_books = [self.ratings_mat.index[indices.flatten()[i]] for i in range(1, len(distances.flatten()))]
        Book_dis = [distances.flatten()[i] for i in range(1, len(distances.flatten()))]

        Book = self.Book_lookup[bID]
        Recommended_Books = self.books[self.books['ISBN'].isin(Rec_books)]
        return Book, Recommended_Books, Book_dis

class SVD(Books):
    def __init__(self, n_latent_factor=50):
        super().__init__()
        self.n_latent_factor = n_latent_factor
        self.ratings_mat = self.ratings_explicit.pivot(
            index="userID", columns="ISBN", values="bookRating").fillna(0)

        self.uti_mat = self.ratings_mat.values
        self.user_ratings_mean = np.mean(self.uti_mat, axis=1)
        self.mat = self.uti_mat - self.user_ratings_mean.reshape(-1, 1)

        self.explicit_users = np.sort(self.ratings_explicit.userID.unique())
        self.User_lookup = dict(
            zip(range(1, len(self.explicit_users)), self.explicit_users))

        self.predictions = None

    def scipy_SVD(self):
        U, S, Vt = svds(self.mat, k=self.n_latent_factor)
        S_diag_matrix = np.diag(S)
        X_pred = np.dot(np.dot(U, S_diag_matrix), Vt) + self.user_ratings_mean.reshape(-1, 1)
        self.predictions = pd.DataFrame(
            X_pred, columns=self.ratings_mat.columns, index=self.ratings_mat.index)

    def Recommend_Books(self, userID, num_recommendations=5):
        user_row_number = self.User_lookup[userID]
        sorted_user_predictions = self.predictions.loc[user_row_number].sort_values(ascending=False)
        user_data = self.ratings_explicit[self.ratings_explicit.userID == (
            self.User_lookup[userID])]
        user_full = (user_data.merge(self.books, how='left', left_on='ISBN', right_on='ISBN').
                     sort_values(['bookRating'], ascending=False))

        recom = (self.books[~self.books['ISBN'].isin(user_full['ISBN'])].
                 merge(pd.DataFrame(sorted_user_predictions).reset_index(), how='left',
                       left_on='ISBN', right_on='ISBN'))
        recom = recom.rename(columns={user_row_number: 'Predictions'})
        recommend = recom.sort_values(by=['Predictions'], ascending=False)
        recommendations = recommend.iloc[:num_recommendations, :-1]
        return user_full, recommendations

@app.route('/predict', methods=['POST'])
def predict():
    global KNN_Recommended_Books
    if request.method == 'POST':
        ICF = KNN()
        book = request.form['book']
        Book, KNN_Recommended_Books, _ = ICF.Recommend_Books(book)

        if KNN_Recommended_Books is None:
            return render_template('result.html', prediction=None)

        KNN_Recommended_Books = KNN_Recommended_Books.merge(
            ICF.average_rating, how='left', on='ISBN')
        KNN_Recommended_Books = KNN_Recommended_Books.rename(
            columns={'bookRating': 'MeanRating'})

        df = pd.DataFrame(KNN_Recommended_Books, columns=['bookTitle', 'bookAuthor', 'MeanRating'])

    return render_template('result.html', prediction=df)

@app.route('/svd', methods=['POST'])
def svd():
    global SVD_Recommended_Books
    if request.method == 'POST':
        userCollaborativeFiltering = SVD()
        userCollaborativeFiltering.scipy_SVD()
        userId = request.form['svd']
        data = int(userId)

        Rated_Books, SVD_Recommended_Books = userCollaborativeFiltering.Recommend_Books(userID=data)

        pd.set_option('display.max_colwidth', -1)

        SVD_Recommended_Books = SVD_Recommended_Books.merge(
            userCollaborativeFiltering.average_rating, how='left', on='ISBN')
        SVD_Recommended_Books = SVD_Recommended_Books.rename(
            columns={'bookRating': 'MeanRating'})

    return render_template('resultSvd.html', predictionB=SVD_Recommended_Books[['bookTitle']],
                           predictionA=SVD_Recommended_Books[['bookAuthor']],
                           predictionR=SVD_Recommended_Books[['MeanRating']],
                           prediction=SVD_Recommended_Books)

if __name__ == '__main__':
    app.run(debug=True)
