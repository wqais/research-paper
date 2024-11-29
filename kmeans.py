import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import string
import numpy as np
import PyPDF2
import json
from kneed import KneeLocator
from feedback import feedback


class CustomEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, bool):
            return str(obj)  # Convert boolean to string
        try:
            # Attempt to convert the object to a serializable format
            return super().default(obj)
        except TypeError:
            # Handle objects that are not serializable
            return str(obj)  # You could also return a custom string representation here


def pdf_to_text(pdf_file_path):
    with open(pdf_file_path, "rb") as file:
        pdf_reader = PyPDF2.PdfReader(file)
        syllabus_text = ""
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            syllabus_text += page.extract_text()  # Corrected from `page.extraimport`
    return syllabus_text


def download_nltk_dependencies():
    """Download required NLTK packages with error handling."""
    required_packages = ["punkt", "stopwords", "wordnet"]
    for package in required_packages:
        try:
            nltk.download(package, quiet=True)
        except Exception as e:
            print(f"Error downloading {package}: {str(e)}")
            return False
    return True


def preprocess_text(texts):
    try:
        lemmatizer = WordNetLemmatizer()
        stop_words = set(stopwords.words("english"))

        processed_texts = []
        all_tokens = []

        for text in texts:
            tokens = word_tokenize(text.lower())
            tokens = [
                lemmatizer.lemmatize(token)
                for token in tokens
                if token not in string.punctuation
                and token not in stop_words
                and token.isalnum()
            ]

            all_tokens.append(tokens)
            processed_texts.append(" ".join(tokens))

        print("\nTokenized Preprocessed Feedback is ready")
        # for idx, tokens in enumerate(all_tokens):
        #     print(f"Feedback {idx + 1}: {tokens}")

        return processed_texts, all_tokens
    except Exception as e:
        print(f"Error in text preprocessing: {str(e)}")
        return None, None


def perform_clustering(texts, n_clusters):
    try:
        vectorizer = TfidfVectorizer(max_features=100)
        X = vectorizer.fit_transform(texts)

        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(X)

        feature_names = vectorizer.get_feature_names_out()
        return X, labels, feature_names, kmeans, vectorizer
    except Exception as e:
        print(f"Error in clustering: {str(e)}")
        return None, None, None, None


def plot_clusters(X, labels, title="Feedback Clusters"):
    try:
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X.toarray())

        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap="viridis")
        plt.title(title)
        plt.xlabel("First Principal Component")
        plt.ylabel("Second Principal Component")
        plt.colorbar(scatter, label="Cluster")
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Error in plotting: {str(e)}")


def plot_elbow_method(X):
    try:
        distortions = []
        K = range(1, 21)
        for k in K:
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(X)
            distortions.append(kmeans.inertia_)

        plt.figure(figsize=(8, 5))
        plt.plot(K, distortions, "bo-")
        plt.xlabel("Number of clusters")
        plt.ylabel("Distortion")
        plt.title("Elbow Method For Optimal k")
        plt.show()

        # Find the "elbow" point
        knee_locator = KneeLocator(
            K, distortions, curve="convex", direction="decreasing"
        )
        optimal_k = knee_locator.knee
        print(f"Optimal number of clusters: {optimal_k}")
        return optimal_k
    except Exception as e:
        print(f"Error in elbow method plotting: {str(e)}")
        return None


def analyze_clusters(texts, labels, feature_names, kmeans):
    try:
        cluster_texts = {i: [] for i in range(max(labels) + 1)}
        for text, label in zip(texts, labels):
            cluster_texts[label].append(text)

        print("\nCluster Analysis:")
        print("-" * 50)

        for cluster_id in cluster_texts:
            print(f"\nCluster {cluster_id}:")
            print(f"Number of items: {len(cluster_texts[cluster_id])}")
            print("Sample feedback:")
            for text in cluster_texts[cluster_id][:3]:
                print(f"- {text}")
            print()
    except Exception as e:
        print(f"Error in cluster analysis: {str(e)}")


def extract_top_keywords_per_cluster(X_tfidf, kmeans_model, vectorizer, top_n=10):
    try:
        terms = vectorizer.get_feature_names_out()
        clusters_keywords = {}
        for i in range(kmeans_model.n_clusters):
            cluster_indices = kmeans_model.labels_ == i
            cluster_center = X_tfidf[cluster_indices].mean(axis=0)
            sorted_indices = np.argsort(cluster_center.A[0])[::-1][:top_n]
            top_keywords = [terms[i] for i in sorted_indices]
            clusters_keywords[f"Cluster {i}"] = top_keywords
        return clusters_keywords
    except Exception as e:
        print(f"Error extracting top keywords: {str(e)}")
        return {}


def find_matching_syllabus_topics(clusters_keywords, syllabus_topics):
    try:
        syllabus_vectorizer = TfidfVectorizer(stop_words="english")
        syllabus_tfidf = syllabus_vectorizer.fit_transform(syllabus_topics)

        matching_results = {}
        for cluster, keywords in clusters_keywords.items():
            keywords_text = " ".join(keywords)
            cluster_tfidf = syllabus_vectorizer.transform([keywords_text])
            similarities = cosine_similarity(cluster_tfidf, syllabus_tfidf)
            top_match_index = similarities.argmax()
            top_match_score = similarities[0, top_match_index]
            matching_results[cluster] = {
                "matched_topic": syllabus_topics[top_match_index],
                "similarity_score": top_match_score,
            }
        return matching_results
    except Exception as e:
        print(f"Error matching syllabus topics: {str(e)}")
        return {}


def suggest_syllabus_modifications(matches, threshold=0.3):
    suggestions = []
    try:
        for cluster, match in matches.items():
            if match["similarity_score"] < threshold:
                suggestions.append(
                    f"Consider adding or expanding content on {cluster} (low similarity with syllabus)."
                )
            else:
                suggestions.append(
                    f"Topic '{match['matched_topic']}' sufficiently covers {cluster}."
                )
        return suggestions
    except Exception as e:
        print(f"Error suggesting syllabus modifications: {str(e)}")
        return []


def main():
    if not download_nltk_dependencies():
        return

    processed_texts, tokens = preprocess_text(feedback)
    if processed_texts is None:
        return

    # Convert processed texts to TF-IDF matrix
    vectorizer = TfidfVectorizer(max_features=100)
    X = vectorizer.fit_transform(processed_texts)

    # Now pass the TF-IDF matrix to the elbow method
    optimal_clusters = plot_elbow_method(X)
    if optimal_clusters is not None:
        X, labels, feature_names, kmeans_model, vectorizer = perform_clustering(
            processed_texts, n_clusters=optimal_clusters
        )
    else:
        print("Could not determine the optimal number of clusters.")
        return

    analyze_clusters(processed_texts, labels, feature_names, kmeans_model)
    plot_clusters(X, labels)

    clusters_keywords = extract_top_keywords_per_cluster(X, kmeans_model, vectorizer)

    with open("syllabus.json", "r") as file:
        syllabus_data = json.load(file)

    syllabus_topics = [subject["subjectName"] for subject in syllabus_data]

    matches = find_matching_syllabus_topics(clusters_keywords, syllabus_topics)
    suggestions = suggest_syllabus_modifications(matches)

    updated_syllabus = syllabus_data.copy()
    for subject in updated_syllabus:
        matched_cluster = next(
            (
                match
                for cluster, match in matches.items()
                if match["matched_topic"] == subject["subjectName"]
            ),
            None,
        )
        if matched_cluster:
            if matched_cluster["similarity_score"] < 0.3:
                subject["modification_suggestion"] = (
                    f"Consider adding or expanding content on {matched_cluster['matched_topic']} (low similarity with syllabus)."
                )
            else:
                subject["modification_suggestion"] = (
                    f"Topic '{matched_cluster['matched_topic']}' sufficiently covers {subject['subjectName']}."
                )
        else:
            subject["modification_suggestion"] = (
                "No matching topic found for this subject. Consider reviewing the syllabus content."
            )

    with open("updatedsyllabus.json", "w") as outfile:
        json.dump(updated_syllabus, outfile, indent=4, cls=CustomEncoder)

    print("\nSyllabus Modification Suggestions:")
    for suggestion in suggestions:
        print(f"- {suggestion}")


if __name__ == "__main__":
    main()