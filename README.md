#  An Analytical Study on Adaptive Curriculum Design: Utilizing Employer Feedback from Campus Recruitment Processes to Optimize Course Syllabi

  

##  Overview

  

This project aims to optimize course syllabi by analyzing employer feedback gathered during campus recruitment processes. By leveraging natural language processing (NLP) and clustering techniques, the study seeks to identify gaps in the existing curriculum and suggest modifications based on real-world employer expectations.

  

##  Table of Contents

  

-  [Introduction](#introduction)

-  [Requirements](#requirements)

-  [Installation](#installation)

-  [Usage](#usage)

-  [Methodology](#methodology)

-  [Results](#results)

-  [Conclusion](#conclusion)

  

##  Introduction

  

As industries evolve, so do the skills and competencies required from graduates. This research focuses on adapting academic curricula to better align with employer needs by utilizing feedback from campus recruitment processes. The goal is to ensure that course syllabi remain relevant and effective in preparing students for the job market.

  

##  Requirements

  

To run this project, you will need the following packages:

  

-  `nltk`

-  `scikit-learn`

-  `matplotlib`

-  `numpy`

-  `sentence-transformers`

-  `kneed`

-  `json`

-  `feedback` (custom module)

  

##  Installation

  

You can install the required packages using pip:
```
pip install nltk scikit-learn matplotlib numpy sentence-transformers kneed
```
Additionally, make sure to download the necessary NLTK dependencies:

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
```

## Usage
1.  Place your syllabus data in a JSON file named  `syllabus.json`.
2.  Ensure that the  `feedback`  module is properly defined and contains the relevant employer feedback.
3.  Run the main script:
```
python main.py
```
4.  The output will be saved in a file named  `updatedsyllabus.json`, containing the modified syllabus with suggestions based on the analysis.

## Methodology

1.  **Data Preprocessing**: The syllabus topics and feedback are preprocessed using tokenization, lemmatization, and removal of stopwords.
2.  **Clustering**: K-means clustering is applied to group similar topics and feedback. The optimal number of clusters is determined using the elbow method.
3.  **PCA Visualization**: Principal Component Analysis (PCA) is used to reduce dimensionality and visualize the clustering results.
4.  **Matching Topics**: The feedback clusters are matched with syllabus topics using cosine similarity to identify gaps and suggest modifications.
5.  **Output Generation**: The updated syllabus is saved to a JSON file, including suggestions for curriculum improvements.

## Results
The analysis results in an updated syllabus that includes:

-   Suggested modifications for topics with low similarity to employer feedback.
-   Recommendations for topics that sufficiently cover the necessary skills and competencies.

## Conclusion
This project demonstrates the potential of utilizing employer feedback to inform and adapt academic curricula. By systematically analyzing feedback and aligning it with course syllabi, educational institutions can enhance their programs and better prepare students for the workforce.

## Connect With Us
Feel free to reach out to us at our emails: qais.warekar23@spit.ac.in and lalit.chandora23@spit.ac.in
