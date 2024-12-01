import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import json
from sentence_transformers import SentenceTransformer
from kneed import KneeLocator
import numpy as np
from feedback import feedback

class CustomEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, bool):
            return str(obj)  # Convert boolean to string
        try:
            return super().default(obj)
        except TypeError:
            return str(obj)  # Handle non-serializable objects

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
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words("english"))

    processed_texts = []
    for text in texts:
        tokens = word_tokenize(text.lower())
        tokens = [
            lemmatizer.lemmatize(token)
            for token in tokens
            if token.isalnum() and token not in stop_words
        ]
        processed_texts.append(" ".join(tokens))

    # print("\nTokenized Preprocessed Text is ready")
    return processed_texts

def check_embeddings(embeddings):
    """Check for NaN values in embeddings and return valid embeddings."""
    if np.any(np.isnan(embeddings)):
        print("Found NaN values in embeddings. Cleaning up...")
        # Remove rows with NaN values
        embeddings = embeddings[~np.isnan(embeddings).any(axis=1)]
    return embeddings

def perform_clustering(texts, n_clusters):
    try:
        model = SentenceTransformer('all-MiniLM-L6-v2')
        embeddings = model.encode(texts)

        # Check for NaN values
        embeddings = check_embeddings(embeddings)

        # Ensure embeddings are valid
        if embeddings.ndim != 2:
            print("Embeddings are not in the correct shape. Expected 2D array.")
            return None, None, None

        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(embeddings)

        return embeddings, labels, kmeans
    except Exception as e:
        print(f"Error in clustering: {str(e)}")
        return None, None, None

def apply_pca(embeddings, n_components=2):
    pca = PCA(n_components=n_components)
    reduced_embeddings = pca.fit_transform(embeddings)
    return reduced_embeddings

def plot_pca_results(X_reduced, labels):
    plt.figure(figsize=(10, 7))
    unique_labels = np.unique(labels)
    
    for label in unique_labels:
        # Select the points that belong to the current cluster
        cluster_points = X_reduced[labels == label]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {label}')
    
    plt.title('PCA of Clusters')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend()
    plt.grid()
    plt.show()

def plot_elbow_method(embeddings):
    try:
        distortions = []
        K = range(1, 21)
        for k in K:
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(embeddings)
            distortions.append(kmeans.inertia_)

        plt.figure(figsize=(8, 5))
        plt.plot(K, distortions, "bo-")
        plt.xlabel("Number of clusters")
        plt.ylabel("Distortion")
        plt.title("Elbow Method For Optimal k")
        plt.show()

        # Find the "elbow" point
        knee_locator = KneeLocator(K, distortions, curve="convex", direction="decreasing")
        optimal_k = knee_locator.knee
        print(f"Optimal number of clusters: {optimal_k}")
        return optimal_k
    except Exception as e:
        print(f"Error in elbow method plotting: {str(e)}")
        return None

def analyze_clusters(texts, labels):
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
        


def find_matching_syllabus_topics(clusters_keywords, syllabus_topics):
    try:
        model = SentenceTransformer('all-MiniLM-L6-v2')
        syllabus_embeddings = model.encode(syllabus_topics)

        matching_results = {}
        for cluster, keywords in clusters_keywords.items():
            keywords_text = " ".join(keywords)
            cluster_embedding = model.encode([keywords_text])
            similarities = cosine_similarity(cluster_embedding, syllabus_embeddings)
            top_match_index = similarities.argmax()
            top_match_score = similarities[0, top_match_index]
            matching_results[cluster] = {
                "matched_topic": syllabus_topics[top_match_index],
                "similarity_score": top_match_score,
                "missing_keywords": list(set(keywords) - set(syllabus_topics[top_match_index].split())),
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
                missing_keywords = ", ".join(match["missing_keywords"])
                suggestions.append(
                    f"Consider adding or expanding content on {cluster} (low similarity with syllabus). Missing keywords: {missing_keywords}."
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

    # Load syllabus data
    with open("syllabus.json", "r") as file:
        syllabus_data = json.load(file)

    syllabus_topics = [subject["subjectName"] for subject in syllabus_data]

    # Preprocess syllabus topics
    processed_syllabus = preprocess_text(syllabus_topics)
                
    # Perform clustering on syllabus topics
    optimal_clusters = plot_elbow_method(
        SentenceTransformer('all-MiniLM-L6-v2').encode(processed_syllabus)
    )
    if optimal_clusters is not None:
        (
            X_syllabus,
            labels_syllabus,
            kmeans_model_syllabus,
        ) = perform_clustering(processed_syllabus, n_clusters=optimal_clusters)
    else:
        print("Could not determine the optimal number of clusters for syllabus.")
        return

    # Preprocess feedback
    processed_feedback = preprocess_text(feedback)
    
    # Perform clustering on feedback
    (
        X_feedback,
        labels_feedback,
        kmeans_model_feedback,
    ) = perform_clustering(processed_feedback, n_clusters=optimal_clusters)

    # Apply PCA to the feedback embeddings
    if X_feedback is not None:
        X_feedback_reduced = apply_pca(X_feedback)

        #Plot PCA Results
        plot_pca_results(X_feedback_reduced, labels_feedback)
        
        # Extract top keywords for feedback clusters
        clusters_keywords = {i: [] for i in range(max(labels_feedback) + 1)}
        for text, label in zip(processed_feedback, labels_feedback):
            clusters_keywords[label].append(text)

        # Match feedback clusters to syllabus topics
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
                if matched_cluster["similarity_score"] < 0.5:  # Adjusted threshold
                    subject["modification_suggestion"] = (
                        f"Consider adding or expanding content on {matched_cluster['matched_topic']} (low similarity with syllabus). Missing keywords: {', '.join(matched_cluster['missing_keywords'])}."
                    )
                else:
                    subject["modification_suggestion"] = (
                        f"Topic '{matched_cluster['matched_topic']}' sufficiently covers {subject['subjectName']}."
                    )
            else:
                subject["modification_suggestion"] = (
                    "No matching topic found for this subject. Consider reviewing the syllabus content."
                )
    else:
        print("Feedback embeddings are invalid. Cannot proceed with PCA.")
        return

    # Write the updated syllabus to a JSON file
    with open("updatedsyllabus.json", "w") as outfile:
        json.dump(updated_syllabus, outfile, indent=4, cls=CustomEncoder)

    print("\nUpdated syllabus has been saved to 'updatedsyllabus.json'.")
    # print("\nSyllabus Modification Suggestions:")

if __name__ == "__main__":
    main()