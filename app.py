import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
import requests
import io
import zipfile
from datetime import datetime
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Set page configuration
st.set_page_config(
    page_title="SideEffectViz",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Add this function for clustering analysis
def perform_clustering_analysis(df, n_clusters=3):
    """
    Perform clustering analysis on medication side effect profiles.
    
    Parameters:
    df (pandas.DataFrame): DataFrame containing medication side effect data
    n_clusters (int): Number of clusters to create
    
    Returns:
    tuple: (clustered_df, pca_df, silhouette_avg, cluster_profiles)
    """
    # Create a pivot table of medications vs side effects
    pivot_df = df.pivot_table(
        index='drug_name', 
        columns='side_effect', 
        values='frequency',
        fill_value=0
    )
    
    # Check if we have enough data for clustering
    if len(pivot_df) < 3:
        return None, None, None, None
    
    if n_clusters >= len(pivot_df):
        n_clusters = len(pivot_df) - 1
    n_clusters = max(2, n_clusters)
    
    # Standardize the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(pivot_df)
    
    # Perform K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(scaled_data)
    
    # Calculate silhouette score
    silhouette_avg = silhouette_score(scaled_data, cluster_labels)
    
    # Add cluster labels to the pivot table
    clustered_df = pivot_df.copy()
    clustered_df['cluster'] = cluster_labels
    
    # Perform PCA for visualization
    n_pca = min(2, len(pivot_df), len(pivot_df.columns))
    pca = PCA(n_components=n_pca)
    principal_components = pca.fit_transform(scaled_data)
    
    # Create DataFrame with PCA results
    if n_pca == 1:
        pca_df = pd.DataFrame(data=principal_components, columns=['PC1'])
        pca_df['PC2'] = 0
    else:
        pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
    pca_df['medication'] = pivot_df.index
    pca_df['cluster'] = cluster_labels
    
    # Calculate cluster profiles (average side effect frequencies per cluster)
    cluster_profiles = {}
    for cluster in range(n_clusters):
        # Get medications in this cluster
        cluster_meds = clustered_df[clustered_df['cluster'] == cluster].index
        
        # Calculate average side effect profile for this cluster
        cluster_profile = df[df['drug_name'].isin(cluster_meds)].groupby('side_effect')['frequency'].mean()
        
        # Get top side effects for this cluster
        top_side_effects = cluster_profile.nlargest(5)
        
        cluster_profiles[cluster] = top_side_effects
    
    return clustered_df, pca_df, silhouette_avg, cluster_profiles

# Add this function to visualize clustering results
def visualize_clusters(pca_df, cluster_profiles, silhouette_avg):
    """
    Create visualizations for clustering results.
    
    Parameters:
    pca_df (pandas.DataFrame): DataFrame with PCA results and cluster assignments
    cluster_profiles (dict): Dictionary of cluster profiles
    silhouette_avg (float): Silhouette score for clustering quality
    
    Returns:
    tuple: (scatter_fig, profiles_fig)
    """
    # Create scatter plot of medications in PCA space
    scatter_fig = px.scatter(
        pca_df,
        x='PC1',
        y='PC2',
        color='cluster',
        hover_name='medication',
        title=f'Medication Clusters (Silhouette Score: {silhouette_avg:.3f})',
        labels={'PC1': 'Principal Component 1', 'PC2': 'Principal Component 2'},
        color_continuous_scale=px.colors.qualitative.G10
    )
    
    # Create bar charts for cluster profiles
    profiles_fig = go.Figure()
    
    for cluster, profile in cluster_profiles.items():
        profiles_fig.add_trace(go.Bar(
            x=profile.index,
            y=profile.values,
            name=f'Cluster {cluster}',
            text=[f'{val:.1f}%' for val in profile.values],
            textposition='auto'
        ))
    
    profiles_fig.update_layout(
        title='Top Side Effects by Cluster',
        xaxis_title='Side Effect',
        yaxis_title='Average Frequency (%)',
        barmode='group',
        legend_title='Cluster'
    )
    
    return scatter_fig, profiles_fig

# Add this function to download and process FAERS data
def get_faers_data(limit=1000):
    """
    Download and process a sample of FAERS data using the OpenFDA API.
    
    Parameters:
    limit (int): Maximum number of records to retrieve
    
    Returns:
    pandas.DataFrame: Processed FAERS data
    """
    st.write("Fetching FAERS data from OpenFDA API...")
    
    # Use OpenFDA API to get adverse event data
    base_url = "https://api.fda.gov/drug/event.json"
    
    # Query parameters
    params = {
        "limit": limit,
        "search": "receivedate:[20230101 TO 20231231]" # 2023 data
    }
    
    try:
        # Make API request
        response = requests.get(base_url, params=params) 
        response.raise_for_status()  # Raise exception for HTTP errors
        
        data = response.json()
        
        if 'results' not in data:
            st.error("No results found in the API response")
            return generate_sample_data()  # Fallback to sample data
        
        # Process the results
        processed_data = []
        
        for result in data['results']:
            # Extract drug information
            if 'patient' in result and 'drug' in result['patient']:
                for drug in result['patient']['drug']:
                    if 'medicinalproduct' not in drug:
                        continue
                        
                    drug_name = drug['medicinalproduct']
                    
                    # Extract reaction information
                    if 'reaction' in result['patient']:
                        for reaction in result['patient']['reaction']:
                            if 'reactionmeddrapt' not in reaction:
                                continue
                                
                            side_effect = reaction['reactionmeddrapt']
                            
                            # Determine severity (if available)
                            severity = 'Moderate'  # Default
                            if 'seriousnessdeath' in result and result['seriousnessdeath'] == '1':
                                severity = 'Severe'
                            elif 'seriousnesslifethreatening' in result and result['seriousnesslifethreatening'] == '1':
                                severity = 'Severe'
                            elif 'seriousnesshospitalization' in result and result['seriousnesshospitalization'] == '1':
                                severity = 'Severe'
                            elif 'seriousnessdisabling' in result and result['seriousnessdisabling'] == '1':
                                severity = 'Severe'
                            
                            # Categorize side effects (simplified)
                            category_keywords = {
                                'Neurological': ['headache', 'dizz', 'pain', 'neuro', 'seizure', 'cerebral', 'brain'],
                                'Gastrointestinal': ['nausea', 'vomit', 'diarrhoea', 'stomach', 'gastro', 'intestin', 'bowel'],
                                'Dermatological': ['rash', 'skin', 'dermat', 'itch', 'urticaria'],
                                'Cardiovascular': ['heart', 'cardio', 'rhythm', 'vascular', 'thromb', 'embol'],
                                'Respiratory': ['breath', 'lung', 'respir', 'pulmonary', 'cough'],
                                'General': ['fatigue', 'fever', 'malaise', 'weakness']
                            }
                            
                            category = 'Other'
                            side_effect_lower = side_effect.lower()
                            
                            for cat, keywords in category_keywords.items():
                                if any(keyword in side_effect_lower for keyword in keywords):
                                    category = cat
                                    break
                            
                            # Add to processed data
                            processed_data.append({
                                'drug_name': drug_name,
                                'side_effect': side_effect,
                                'frequency': 1,  # Count occurrences later
                                'severity': severity,
                                'category': category
                            })
        
        # Convert to DataFrame
        df = pd.DataFrame(processed_data)
        
        # Calculate frequency as percentage of total for each drug-side effect pair
        if not df.empty:
            # Count occurrences of each drug-side effect pair
            freq_df = df.groupby(['drug_name', 'side_effect']).size().reset_index(name='count')
            
            # Calculate total occurrences for each drug
            drug_totals = freq_df.groupby('drug_name')['count'].sum().reset_index(name='total')
            
            # Merge to get totals
            freq_df = freq_df.merge(drug_totals, on='drug_name')
            
            # Calculate frequency as percentage
            freq_df['frequency'] = (freq_df['count'] / freq_df['total'] * 100).round(1)
            
            # Merge frequency back to original data
            df = df.drop('frequency', axis=1).drop_duplicates(['drug_name', 'side_effect'])
            df = df.merge(freq_df[['drug_name', 'side_effect', 'frequency']], on=['drug_name', 'side_effect'])
        
        # Limit to top medications by frequency for better visualization
        top_meds = df.groupby('drug_name')['frequency'].sum().nlargest(15).index.tolist()
        df = df[df['drug_name'].isin(top_meds)]
        
        st.success(f"Successfully processed {len(df)} FAERS records")
        return df
        
    except Exception as e:
        st.error(f"Error fetching FAERS data: {str(e)}")
        st.write("Falling back to sample data...")
        return generate_sample_data()  # Fallback to sample data
    
# Function to generate sample data with the correct structure
def generate_sample_data():
    # Sample medications
    medications = ['Ibuprofen', 'Aspirin', 'Loratadine', 'Amoxicillin', 'Lisinopril']
    
    # Sample side effects
    side_effects = ['Headache', 'Nausea', 'Dizziness', 'Rash', 'Fatigue']
    
    # Side effect categories
    categories = {
        'Headache': 'Neurological',
        'Nausea': 'Gastrointestinal',
        'Dizziness': 'Neurological',
        'Rash': 'Dermatological',
        'Fatigue': 'General'
    }
    
    # Severity levels
    severity_levels = ['Mild', 'Moderate', 'Severe']
    
    # Generate random associations
    data = []
    for med in medications:
        for se in side_effects:
            # Not all medications have all side effects
            if np.random.random() > 0.3:  # 70% chance of having the side effect
                frequency = round(np.random.uniform(1, 25), 1)
                severity = np.random.choice(severity_levels)
                category = categories[se]
                
                data.append({
                    'drug_name': med,
                    'side_effect': se,
                    'frequency': frequency,
                    'severity': severity,
                    'category': category
                })
    
    return pd.DataFrame(data)

# Function to create network graph visualization
def create_network_graph(df, selected_medication=None, selected_category=None, selected_severity=None):
    # Filter data based on selections
    filtered_df = df.copy()
    
    if selected_medication != 'All Medications':
        filtered_df = filtered_df[filtered_df['drug_name'] == selected_medication]
    
    if selected_category != 'All Categories':
        filtered_df = filtered_df[filtered_df['category'] == selected_category]
        
    if selected_severity != 'All Levels':
        filtered_df = filtered_df[filtered_df['severity'] == selected_severity]
    
    # Create a graph
    G = nx.Graph()
    
    # Add medication nodes
    for med in filtered_df['drug_name'].unique():
        G.add_node(med, type='medication')
    
    # Add side effect nodes
    for se in filtered_df['side_effect'].unique():
        G.add_node(se, type='side_effect')
    
    # Add edges
    for _, row in filtered_df.iterrows():
        G.add_edge(row['drug_name'], row['side_effect'], weight=row['frequency'])
    
    # Create positions for nodes
    pos = nx.spring_layout(G, seed=42)
    
    # Create edge trace
    edge_x = []
    edge_y = []
    
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')
    
    # Create node trace for medications
    med_node_x = []
    med_node_y = []
    med_node_text = []
    
    for node in G.nodes():
        if G.nodes[node]['type'] == 'medication':
            x, y = pos[node]
            med_node_x.append(x)
            med_node_y.append(y)
            
            # Count associated side effects
            side_effects_count = len(list(G.neighbors(node)))
            med_node_text.append(f"{node}<br>Associated Side Effects: {side_effects_count}")
    
    med_node_trace = go.Scatter(
        x=med_node_x, y=med_node_y,
        mode='markers',
        hoverinfo='text',
        text=med_node_text,
        marker=dict(
            showscale=False,
            color='#1A365D',
            size=20,
            line_width=2))
    
    # Create node trace for side effects
    se_node_x = []
    se_node_y = []
    se_node_text = []
    se_node_color = []
    
    color_map = {
        'Neurological': '#FF6B35',
        'Gastrointestinal': '#00A3B4',
        'Dermatological': '#7A28CB',
        'General': '#2D936C'
    }
    
    for node in G.nodes():
        if G.nodes[node]['type'] == 'side_effect':
            x, y = pos[node]
            se_node_x.append(x)
            se_node_y.append(y)
            
            # Get category for color
            category = 'Other'
            for _, row in filtered_df.iterrows():
                if row['side_effect'] == node:
                    category = row['category']
                    break
                    
            se_node_color.append(color_map.get(category, '#888'))
            
            # Count associated medications
            meds_count = len(list(G.neighbors(node)))
            se_node_text.append(f"{node}<br>Category: {category}<br>Associated Medications: {meds_count}")
    
    se_node_trace = go.Scatter(
        x=se_node_x, y=se_node_y,
        mode='markers',
        hoverinfo='text',
        text=se_node_text,
        marker=dict(
            showscale=False,
            color=se_node_color,
            size=15,
            line_width=2))
    
    # Create figure
    fig = go.Figure()
    
    # Add traces
    fig.add_trace(edge_trace)
    fig.add_trace(med_node_trace)
    fig.add_trace(se_node_trace)
    
    # Update layout with compatible properties
    fig.update_layout(
        title=dict(text='Medication - Side Effect Network', font=dict(size=16)),
        showlegend=False,
        hovermode='closest',
        margin=dict(b=20, l=5, r=5, t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=600,
        plot_bgcolor='#f8f9fa'
    )
    
    return fig



# Add a toggle in the sidebar to switch between sample and real data
st.sidebar.header("Data Source")
use_real_data = st.sidebar.checkbox("Use real FAERS data", value=True)

if use_real_data:
    df = get_faers_data(limit=1000)  # Start with 1000 records
else:
    df = generate_sample_data()

# Header
st.title("SideEffectViz")
st.subheader("Interactive Visualization of Medication Side Effects")

# Sidebar filters
st.sidebar.header("Filters")

# Medication filter
medication_options = ['All Medications'] + sorted(df['drug_name'].unique().tolist())
selected_medication = st.sidebar.selectbox("Select Medication", medication_options)

# Category filter
category_options = ['All Categories'] + sorted(df['category'].unique().tolist())
selected_category = st.sidebar.selectbox("Select Side Effect Category", category_options)

# Severity filter
severity_options = ['All Levels'] + sorted(df['severity'].unique().tolist())
selected_severity = st.sidebar.selectbox("Select Severity Level", severity_options)

# Apply filters
filtered_df = df.copy()
if selected_medication != 'All Medications':
    filtered_df = filtered_df[filtered_df['drug_name'] == selected_medication]
if selected_category != 'All Categories':
    filtered_df = filtered_df[filtered_df['category'] == selected_category]
if selected_severity != 'All Levels':
    filtered_df = filtered_df[filtered_df['severity'] == selected_severity]

# Display metrics
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Medications", len(df['drug_name'].unique()))
with col2:
    st.metric("Unique Side Effects", len(df['side_effect'].unique()))
with col3:
    st.metric("Average Frequency", f"{df['frequency'].mean():.1f}%")
with col4:
    st.metric("Data Points", len(df))

# Create tabs for different visualizations
tab1, tab2, tab3, tab4 = st.tabs(["Network View", "Frequency Analysis", "Heatmap View", "Clustering Analysis"])

with tab1:
    # Network graph visualization
    st.plotly_chart(create_network_graph(filtered_df, selected_medication, selected_category, selected_severity), use_container_width=True)
    
    # Add legend for network graph
    legend_col1, legend_col2, legend_col3, legend_col4 = st.columns(4)
    with legend_col1:
        st.markdown("🔵 **Medications**")
    with legend_col2:
        st.markdown("🟠 **Neurological**")
    with legend_col3:
        st.markdown("🟢 **Gastrointestinal**")
    with legend_col4:
        st.markdown("🟣 **Dermatological**")

with tab2:
    # Simple bar chart of side effect frequencies
    if selected_medication == 'All Medications':
        # Group by side effect and calculate mean frequency
        grouped_df = filtered_df.groupby('side_effect')['frequency'].mean().reset_index()
        grouped_df = grouped_df.sort_values('frequency', ascending=False)
        title = 'Average Side Effect Frequency Across All Medications'
    else:
        # Sort by frequency
        grouped_df = filtered_df.sort_values('frequency', ascending=False)
        title = f'Side Effect Frequencies for {selected_medication}'
    
    fig = px.bar(
        grouped_df,
        x='side_effect',
        y='frequency',
        color='side_effect',
        title=title,
        labels={'side_effect': 'Side Effect', 'frequency': 'Frequency (%)'}
    )
    
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    # Create a pivot table of medications vs side effects
    pivot_df = filtered_df.pivot_table(
        index='drug_name', 
        columns='side_effect', 
        values='frequency',
        fill_value=0
    )
    
    # Create heatmap
    fig = px.imshow(
        pivot_df,
        labels=dict(x="Side Effect", y="Medication", color="Frequency (%)"),
        x=pivot_df.columns,
        y=pivot_df.index,
        color_continuous_scale='Viridis',
        title='Medication-Side Effect Frequency Heatmap'
    )
    
    st.plotly_chart(fig, use_container_width=True)

# Add the clustering tab
with tab4:
    st.subheader("Medication Clustering by Side Effect Profiles")
    
    # Add slider for number of clusters
    n_clusters = st.slider("Number of Clusters", min_value=2, max_value=6, value=3)
    
    # Perform clustering
    result = perform_clustering_analysis(filtered_df, n_clusters)
    
    if result[0] is None:
        st.info("Not enough medications in the current selection for clustering analysis. Select 'All Medications' or a broader filter to use this feature.")
    else:
        clustered_df, pca_df, silhouette_avg, cluster_profiles = result
        
        # Display clustering metrics
        st.metric("Silhouette Score", f"{silhouette_avg:.3f}", 
                  help="Measures how similar an object is to its own cluster compared to other clusters. Values range from -1 to 1, with higher values indicating better clustering.")
        
        # Create two columns for visualizations
        cluster_col1, cluster_col2 = st.columns(2)
        
        # Visualize clustering results
        scatter_fig, profiles_fig = visualize_clusters(pca_df, cluster_profiles, silhouette_avg)
        
        with cluster_col1:
            st.plotly_chart(scatter_fig, use_container_width=True)
            
        with cluster_col2:
            st.plotly_chart(profiles_fig, use_container_width=True)
        
        # Display medications in each cluster
        st.subheader("Medications by Cluster")
        
        for cluster in range(n_clusters):
            cluster_meds = clustered_df[clustered_df['cluster'] == cluster].index.tolist()
            
            with st.expander(f"Cluster {cluster} ({len(cluster_meds)} medications)"):
                st.write(", ".join(cluster_meds))
                
                # Get top side effects for this cluster
                if cluster in cluster_profiles:
                    st.write("**Characteristic Side Effects:**")
                    for side_effect, freq in cluster_profiles[cluster].items():
                        st.write(f"- {side_effect}: {freq:.1f}%")

# About section
with st.expander("About SideEffectViz"):
    st.write("""
    **SideEffectViz** is an interactive visualization and analysis tool that uses machine learning to identify patterns 
    in medication side effects from FDA Adverse Event Reporting System (FAERS) data.
    
    **Features:**
    - **Interactive Network Visualization**: Explore the relationships between medications and their side effects through an 
      interactive network graph with color-coded categories.
    - **Real FAERS Data Integration**: Access and analyze actual pharmaceutical adverse event reports from the FDA's database.
    - **Frequency Analysis**: Examine the prevalence of different side effects across medications through interactive bar charts.
    - **Heatmap Visualization**: View medication-side effect relationships through an intuitive color-coded heatmap.
    - **Machine Learning Clustering**: Discover patterns in medication side effect profiles using K-means clustering and 
      Principal Component Analysis (PCA).
    
    **How It Works:**
    1. The application retrieves adverse event reports from the OpenFDA API
    2. Side effects are categorized and frequencies are calculated
    3. Interactive visualizations allow exploration of relationships
    4. K-means clustering identifies medications with similar side effect profiles
    5. PCA reduces dimensionality to visualize clusters in 2D space
    
    This tool demonstrates the application of machine learning techniques beyond simple GenAI, 
    combining data visualization, clustering algorithms, and dimensional reduction to extract 
    meaningful insights from pharmaceutical data.
    """)

# Add technical details as a separate expander
with st.expander("Technical Details"):
    st.write("""
    **Technologies Used:**
    - **Streamlit**: Web application framework
    - **Plotly**: Interactive visualizations
    - **NetworkX**: Network graph creation
    - **scikit-learn**: K-means clustering and PCA
    - **OpenFDA API**: Real adverse event data
    
    **Machine Learning Components:**
    - **K-means Clustering**: Unsupervised learning algorithm that groups medications with similar side effect profiles
    - **Principal Component Analysis**: Dimensionality reduction technique for visualizing high-dimensional data
    - **Silhouette Score**: Metric for evaluating clustering quality (ranges from -1 to 1, with higher values indicating better clustering)
    
    **Future Enhancements:**
    - Time-series analysis of adverse event trends
    - Additional clustering algorithms (DBSCAN, Hierarchical)
    - Natural language processing of adverse event descriptions
    - Predictive modeling of potential drug interactions
    """)
