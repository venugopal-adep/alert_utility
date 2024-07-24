import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import random
import shap

# Set page config
st.set_page_config(page_title="AIOps Alert System", layout="wide")

# Custom color palette
COLOR_PALETTE = px.colors.qualitative.Bold

@st.cache_data
def generate_synthetic_dataset(num_incidents=1000):
    """Generate a synthetic dataset of IT incidents"""
    categories = ['Network', 'Server', 'Application', 'Database', 'Security']
    actions = [
        'Restart service', 'Update software', 'Reconfigure settings',
        'Replace hardware', 'Add resources', 'Run diagnostics',
        'Apply security patch', 'Restore from backup', 'Clear cache',
        'Reset user permissions'
    ]
    
    descriptions = [
        'High CPU usage detected', 'Network connectivity issues',
        'Application crashes frequently', 'Database query timeout',
        'Suspicious login attempts', 'Disk space running low',
        'Memory leak detected', 'Service unresponsive',
        'Slow response time', 'Data integrity issues'
    ]
    
    data = {
        'incident_id': range(1, num_incidents + 1),
        'category': [random.choice(categories) for _ in range(num_incidents)],
        'description': [
            f"{random.choice(descriptions)} on {random.choice(categories)} system"
            for _ in range(num_incidents)
        ],
        'action': [random.choice(actions) for _ in range(num_incidents)],
        'severity': [random.choice(['Low', 'Medium', 'High']) for _ in range(num_incidents)],
        'resolution_time': [random.randint(10, 180) for _ in range(num_incidents)],
        'impact_score': [random.uniform(1, 10) for _ in range(num_incidents)],
        'priority': [random.choice(['P1', 'P2', 'P3', 'P4']) for _ in range(num_incidents)]
    }
    
    df = pd.DataFrame(data)
    
    # Ensure all text fields are strings
    text_columns = ['category', 'description', 'action', 'severity', 'priority']
    for col in text_columns:
        df[col] = df[col].astype(str)
    
    return df

@st.cache_resource
def prepare_ml_models(df):
    """Prepare machine learning models"""
    # TF-IDF Vectorizer
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['description'])
    
    # K-means clustering
    kmeans = KMeans(n_clusters=5, random_state=42)
    df['cluster'] = kmeans.fit_predict(tfidf_matrix)
    
    # Random Forest Classifier
    X = tfidf_matrix
    y = df['action']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_classifier.fit(X_train, y_train)
    
    return tfidf, kmeans, rf_classifier, X_train, df

def get_recommendations(alert, df, tfidf, rf_classifier, top_n=5):
    """Get utility recommendations based on the alert description"""
    alert_vector = tfidf.transform([alert['description']])
    
    # Use Random Forest to predict the action
    predicted_action = rf_classifier.predict(alert_vector)[0]
    
    # Find similar incidents
    similarities = rf_classifier.predict_proba(alert_vector)[0]
    similar_indices = similarities.argsort()[-top_n-1:-1][::-1]
    recommendations = df.iloc[similar_indices]
    
    return predicted_action, recommendations[['incident_id', 'category', 'description', 'action', 'severity', 'resolution_time']]

def create_wordcloud(text):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    return fig

def main():
    st.title("🚨 AIOps Alert to Utility Recommendation System")
    st.markdown("""
    <style>
    .big-font {
        font-size:20px !important;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)
    st.markdown('<p class="big-font">Empowering IT Operations with AI-driven Insights and Recommendations</p>', unsafe_allow_html=True)

    # Generate dataset and prepare models
    df = generate_synthetic_dataset()
    tfidf, kmeans, rf_classifier, X_train, df = prepare_ml_models(df)

    # Create tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["🎯 Alert Simulation", "📊 Data Exploration", "🧠 Model Insights", "🔍 Explainability", "📁 Dataset"])

    with tab1:
        st.header("🎯 Alert Simulation")
        st.info("""
        This section simulates new IT incidents and provides AI-driven recommendations based on historical data.
        Click the button below to generate a new alert and see the system's intelligent suggestions.
        """)
        if st.button("🔄 Generate New Alert", key="generate_alert"):
            alert = df.sample(1).iloc[0]
            st.subheader("🚨 New Alert:")
            col1, col2 = st.columns(2)
            with col1:
                st.success(f"**Incident ID:** {alert['incident_id']}")
                st.success(f"**Category:** {alert['category']}")
                st.success(f"**Severity:** {alert['severity']}")
            with col2:
                st.warning(f"**Description:** {alert['description']}")
            
            predicted_action, recommendations = get_recommendations(alert, df, tfidf, rf_classifier)
            
            st.subheader("🎯 Predicted Action:")
            st.info(predicted_action, icon="🔍")
            
            st.subheader("📋 Recommended Actions:")
            st.markdown("""
            These are similar incidents from our historical data, providing additional context and alternative solutions.
            """)
            st.dataframe(recommendations.style.background_gradient(cmap='YlOrRd'))

    with tab2:
        st.header("📊 Data Exploration")
        st.info("""
        Dive into the visual insights of our incident data to uncover patterns and distributions.
        These charts provide a quick overview of the incident landscape.
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🍰 Incident Categories")
            category_counts = df['category'].value_counts()
            fig = px.pie(values=category_counts.values, names=category_counts.index, title="Incident Categories", color_discrete_sequence=COLOR_PALETTE)
            fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig)
            st.markdown("This chart illustrates the distribution of incidents across different categories, helping identify the most common areas of concern.")
        
        with col2:
            st.subheader("📊 Severity Distribution")
            severity_counts = df['severity'].value_counts()
            fig = px.bar(x=severity_counts.index, y=severity_counts.values, title="Severity Distribution", color=severity_counts.index, color_discrete_sequence=COLOR_PALETTE)
            fig.update_layout(xaxis_title="Severity", yaxis_title="Number of Incidents")
            st.plotly_chart(fig)
            st.markdown("This chart showcases the distribution of incident severities, allowing for quick assessment of overall system health.")
        
        st.subheader("⏱️ Resolution Time by Category")
        fig = px.box(df, x="category", y="resolution_time", title="Resolution Time by Category", color="category", color_discrete_sequence=COLOR_PALETTE)
        fig.update_layout(xaxis_title="Category", yaxis_title="Resolution Time (minutes)")
        st.plotly_chart(fig)
        st.markdown("This box plot reveals the distribution of resolution times for each incident category, highlighting potential areas for improvement in response times.")

    with tab3:
        st.header("🧠 Model Insights")
        st.info("""
        Explore the inner workings of our machine learning models used for clustering and classification.
        Gain insights into how the AI system categorizes and analyzes incidents.
        """)
        
        st.subheader("🌐 3D Cluster Visualization")
        cluster_columns = ['resolution_time', 'impact_score', 'severity']
        x_column = st.selectbox("Select X-axis:", options=df.columns, index=df.columns.get_loc(cluster_columns[0]))
        y_column = st.selectbox("Select Y-axis:", options=df.columns, index=df.columns.get_loc(cluster_columns[1]))
        z_column = st.selectbox("Select Z-axis:", options=df.columns, index=df.columns.get_loc(cluster_columns[2]))
        
        # Handle categorical columns
        df_plot = df.copy()
        for col in [x_column, y_column, z_column]:
            if df_plot[col].dtype == 'object':
                df_plot[col] = pd.Categorical(df_plot[col]).codes
        
        fig = px.scatter_3d(df_plot, x=x_column, y=y_column, z=z_column, color='cluster',
                            title="3D Incident Clusters", color_continuous_scale=px.colors.sequential.Viridis)
        fig.update_layout(scene=dict(xaxis_title=x_column, yaxis_title=y_column, zaxis_title=z_column))
        st.plotly_chart(fig)
        st.markdown("""
        This 3D scatter plot visualizes the clusters of incidents based on the selected features.
        Each point represents an incident, and colors indicate different clusters identified by the AI.
        You can rotate and zoom the plot to explore the clusters from different angles.
        """)
        
        st.subheader("🔍 Explore Cluster Contents")
        cluster_names = {0: "Network Issues", 1: "Security Alerts", 2: "Performance Problems", 3: "Software Errors", 4: "Hardware Failures"}
        selected_cluster = st.selectbox("Select a cluster to explore:", options=sorted(df['cluster'].unique()), format_func=lambda x: f"Cluster {x}: {cluster_names[x]}")
        
        cluster_data = df[df['cluster'] == selected_cluster]
        st.markdown(f"### Cluster {selected_cluster}: {cluster_names[selected_cluster]}")
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📊 Common Actions")
            cluster_actions = cluster_data['action'].value_counts()
            fig = px.bar(x=cluster_actions.index, y=cluster_actions.values, title=f"Common Actions in Cluster {selected_cluster}", color_discrete_sequence=COLOR_PALETTE)
            fig.update_layout(xaxis_title="Action", yaxis_title="Frequency")
            st.plotly_chart(fig)
        
        with col2:
            st.subheader("☁️ Word Cloud")
            cluster_text = " ".join(cluster_data['description'].astype(str))
            st.pyplot(create_wordcloud(cluster_text))
        
        st.subheader("📋 Cluster Data Sample")
        st.dataframe(cluster_data[['incident_id', 'category', 'description', 'action', 'severity']].head(10).style.background_gradient(cmap='YlOrRd'))

    with tab4:
        st.header("🔍 Model Explainability")
        st.info("""
        Understand how our AI model makes decisions using SHAP (SHapley Additive exPlanations) values.
        This section provides insights into which features have the most impact on the model's predictions.
        """)

        # Prepare SHAP values
        explainer = shap.TreeExplainer(rf_classifier)
        
        # Convert sparse matrix to dense array for SHAP
        X_train_dense = X_train.toarray()
        
        # Limit the number of features for SHAP analysis to avoid memory issues
        max_features = 1000
        if X_train_dense.shape[1] > max_features:
            feature_importance = rf_classifier.feature_importances_
            top_features = feature_importance.argsort()[-max_features:][::-1]
            X_train_dense = X_train_dense[:, top_features]
            feature_names = np.array(tfidf.get_feature_names_out())[top_features]
        else:
            feature_names = tfidf.get_feature_names_out()
        
        # Sample a subset of the data for SHAP analysis to reduce computation time
        sample_size = min(1000, X_train_dense.shape[0])
        X_train_sample = X_train_dense[np.random.choice(X_train_dense.shape[0], sample_size, replace=False)]
        
        shap_values = explainer.shap_values(X_train_sample)

        st.subheader("🏆 Feature Importance (SHAP)")
        fig, ax = plt.subplots(figsize=(10, 6))
        shap.summary_plot(shap_values, X_train_sample, plot_type="bar", feature_names=feature_names, show=False)
        st.pyplot(fig)
        st.markdown("""
        This chart shows the average impact of each feature on the model output.
        Features are ranked by their absolute SHAP values, indicating their overall importance in the model's decisions.
        Red bars indicate features that increase the likelihood of a particular action, while blue bars decrease it.
        """)

        st.subheader("🎯 SHAP Value Distribution")
        fig, ax = plt.subplots(figsize=(10, 8))
        shap.summary_plot(shap_values, X_train_sample, feature_names=feature_names, show=False)
        st.pyplot(fig)
        st.markdown("""
        This plot shows the distribution of SHAP values for each feature.
        Each point represents a single prediction, with color indicating the feature value (red high, blue low).
        Features are ranked by their overall importance, with the most impactful features at the top.
        """)

    with tab5:
        st.header("📁 Dataset Overview")
        st.info("""
        Explore a sample of the synthetic dataset used in this application.
        This data forms the foundation of our AI-driven recommendations and insights.
        """)
        st.dataframe(df.head(20).style.background_gradient(cmap='YlOrRd'))
        
        st.markdown("""
        **📋 Column Explanations:**
        - **incident_id**: Unique identifier for each incident
        - **category**: The general category of the IT issue (e.g., Network, Server, Application)
        - **description**: A brief description of the incident
        - **action**: The action taken to resolve the incident
        - **severity**: The urgency or impact level of the incident (Low, Medium, High)
        - **resolution_time**: The time taken to resolve the incident (in minutes)
        - **impact_score**: A numerical score representing the impact of the incident (1-10)
        - **priority**: The priority level assigned to the incident (P1, P2, P3, P4)
        - **cluster**: The cluster assigned by the K-means algorithm
        """)

        st.subheader("📊 Dataset Statistics")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Incidents", len(df))
        with col2:
            st.metric("Unique Categories", df['category'].nunique())
        with col3:
            st.metric("Avg. Resolution Time", f"{df['resolution_time'].mean():.2f} minutes")

if __name__ == "__main__":
    main()