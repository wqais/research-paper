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

        print("\nTokenized Preprocessed Feedback:")
        for idx, tokens in enumerate(all_tokens):
            print(f"Feedback {idx + 1}: {tokens}")

        return processed_texts, all_tokens
    except Exception as e:
        print(f"Error in text preprocessing: {str(e)}")
        return None, None


def perform_clustering(texts, n_clusters=3):
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
        K = range(1, 11)
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
    except Exception as e:
        print(f"Error in elbow method plotting: {str(e)}")


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
    feedback = [
        "Students need stronger problem-solving skills, especially in coding challenges.",
        "Effective communication and teamwork abilities are essential for success.",
        "Hands-on experience with real-world projects in cybersecurity would be valuable.",
        "There is a lack of understanding of data structures and algorithms among students.",
        "Practical knowledge of cloud computing, such as AWS or Azure, is highly desired.",
        "Students should focus more on time management and organizational skills.",
        "A deeper understanding of machine learning concepts and applications is required.",
        "More emphasis on leadership and project management skills is necessary.",
        "Programming languages like Python and Java should be mastered by students.",
        "Critical thinking and analytical reasoning need improvement in student performance.",
        "Experience with DevOps tools like Docker and Kubernetes would be beneficial.",
        "The curriculum lacks focus on artificial intelligence and deep learning techniques.",
        "Students should gain better proficiency in software testing and debugging practices.",
        "Soft skills like negotiation, conflict resolution, and presentation skills need attention.",
        "Familiarity with cybersecurity practices, such as encryption and secure coding, is essential.",
        "Practical experience with front-end frameworks like React or Angular is in demand.",
        "A stronger focus on database management systems like SQL and NoSQL is needed.",
        "The curriculum should include more case studies and real-world business scenarios.",
        "Students should be more proficient in using Git for version control and collaboration.",
        "An understanding of agile methodologies and working in a sprint-based environment is crucial.",
    ]

    if not download_nltk_dependencies():
        return

    processed_texts, tokens = preprocess_text(feedback)
    if processed_texts is None:
        return

    X, labels, feature_names, kmeans_model, vectorizer = perform_clustering(
        processed_texts, n_clusters=3
    )

    if X is None:
        return

    plot_elbow_method(X)
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
        updated_syllabus = syllabus_data.copy()
    for subject in updated_syllabus:
        matched_cluster = next((match for cluster, match in matches.items() if match['matched_topic'] == subject['subjectName']), None)
        if matched_cluster:
            if matched_cluster['similarity_score'] < 0.3:
                subject['modification_suggestion'] = f"Consider adding or expanding content on {matched_cluster['matched_topic']} (low similarity with syllabus)."
            else:
                subject['modification_suggestion'] = f"Topic '{matched_cluster['matched_topic']}' sufficiently covers {subject['subjectName']}."
        else:
            subject['modification_suggestion'] = "No matching topic found for this subject. Consider reviewing the syllabus content."


    with open("updatedsyllabus.json", "w") as outfile:
        json.dump(updated_syllabus, outfile, indent=4, cls=CustomEncoder)

    print("\nSyllabus Modification Suggestions:")
    for suggestion in suggestions:
        print(f"- {suggestion}")


if __name__ == "__main__":
    main()
