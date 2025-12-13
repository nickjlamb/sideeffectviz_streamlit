# SideEffectViz 💊📊

An interactive visualization and analysis tool that uses machine learning to identify patterns in medication side effects from FDA Adverse Event Reporting System (FAERS) data.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://www.pharmatools.ai/sideeffectviz)
![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

![SideEffectViz Demo](https://img.shields.io/badge/Live_Demo-PharmaTools.AI-FF4B4B?style=for-the-badge&logo=streamlit)

## 🎯 Overview

SideEffectViz demonstrates the application of machine learning techniques beyond simple generative AI, combining data visualization, clustering algorithms, and dimensionality reduction to extract meaningful insights from pharmaceutical safety data.

**[Try the Live Demo →](https://www.pharmatools.ai/sideeffectviz)**

## ✨ Features

- **Interactive Network Visualization** — Explore medication–side effect relationships through a dynamic network graph with color-coded categories
- **Real FAERS Data Integration** — Access and analyze actual pharmaceutical adverse event reports from the FDA's database via the OpenFDA API
- **Frequency Analysis** — Examine side effect prevalence across medications through interactive bar charts
- **Heatmap Visualization** — View medication–side effect relationships through an intuitive color-coded matrix
- **ML-Powered Clustering** — Discover patterns in medication side effect profiles using K-means clustering and PCA

## 🔬 How It Works

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   OpenFDA API   │────▶│  Data Processing │────▶│  Visualization  │
│  (FAERS Data)   │     │  & Categorization│     │    & Analysis   │
└─────────────────┘     └──────────────────┘     └─────────────────┘
                                                          │
                                                          ▼
                        ┌──────────────────┐     ┌─────────────────┐
                        │   2D Projection  │◀────│  ML Clustering  │
                        │      (PCA)       │     │   (K-means)     │
                        └──────────────────┘     └─────────────────┘
```

1. Retrieve adverse event reports from the OpenFDA API
2. Categorize side effects and calculate frequencies
3. Generate interactive visualizations for data exploration
4. Apply K-means clustering to identify medications with similar side effect profiles
5. Use PCA to reduce dimensionality and visualize clusters in 2D space

## 🧠 Machine Learning Components

| Component | Purpose |
|-----------|---------|
| **K-means Clustering** | Unsupervised learning algorithm that groups medications with similar side effect profiles |
| **Principal Component Analysis (PCA)** | Dimensionality reduction technique for visualizing high-dimensional side effect data in 2D |
| **Silhouette Score** | Clustering quality metric (ranges -1 to 1; higher = better-defined clusters) |

## 🛠️ Tech Stack

| Category | Technologies |
|----------|-------------|
| **Frontend** | Streamlit |
| **Visualization** | Plotly, NetworkX |
| **Machine Learning** | scikit-learn (K-means, PCA) |
| **Data Source** | OpenFDA API |
| **Deployment** | Streamlit Community Cloud |

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/nickjlamb/sideeffectviz_streamlit.git
cd sideeffectviz_streamlit

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

## 📈 Future Enhancements

- [ ] Time-series analysis of adverse event trends
- [ ] Additional clustering algorithms (DBSCAN, Hierarchical)
- [ ] Natural language processing of adverse event descriptions
- [ ] Predictive modeling of potential drug interactions

## 👤 Author

**Nick Lamb, PhD, CMPP**  
Senior Medical Copywriter & Creative Technologist

- 🌐 [PharmaTools.AI](https://www.pharmatools.ai)
- 💼 [LinkedIn](https://www.linkedin.com/in/nickjlamb/)
- ✍️ [Medium](https://medium.com/@nickjlamb)

---

*Part of the [PharmaTools.AI](https://www.pharmatools.ai) suite of AI-powered tools for pharmaceutical professionals.*
