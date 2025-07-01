import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
import io
import base64
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Health Cluster Predictor",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Main styling */
    .main {
        font-family: 'Inter', sans-serif;
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    }
    
    .main-header h1 {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    .main-header p {
        font-size: 1.2rem;
        opacity: 0.9;
        margin: 0;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #f8fafc 0%, #e2e8f0 100%);
    }
    
    .sidebar-header {
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        text-align: center;
        font-weight: 600;
    }
    
    /* Card styling */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        border: 1px solid #e2e8f0;
        margin-bottom: 1rem;
        transition: transform 0.2s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 30px rgba(0,0,0,0.12);
    }
    
    .cluster-result {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        margin: 2rem 0;
        text-align: center;
        box-shadow: 0 10px 30px rgba(16, 185, 129, 0.2);
    }
    
    .cluster-result h2 {
        font-size: 2rem;
        margin-bottom: 1rem;
        font-weight: 700;
    }
    
    .cluster-description {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #10b981;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    }
    
    .recommendation-box {
        background: linear-gradient(135deg, #fbbf24 0%, #f59e0b 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 4px 20px rgba(251, 191, 36, 0.2);
    }
    
    .tip-box {
        background: #f0f9ff;
        border: 1px solid #0ea5e9;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 4px solid #0ea5e9;
    }
    
    .warning-text {
        color: #dc2626;
        font-weight: 500;
        font-size: 0.9rem;
        margin-top: 0.5rem;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    }
    
    /* Download button */
    .download-btn {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        text-decoration: none;
        display: inline-block;
        margin: 0.5rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(16, 185, 129, 0.3);
    }
    
    .download-btn:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(16, 185, 129, 0.4);
        text-decoration: none;
        color: white;
    }
    
    /* Expandable sections */
    .streamlit-expanderHeader {
        background: #f8fafc;
        border-radius: 10px;
        border: 1px solid #e2e8f0;
        font-weight: 600;
    }
    
    /* Comparison styling */
    .comparison-good {
        color: #10b981;
        font-weight: 600;
    }
    
    .comparison-bad {
        color: #ef4444;
        font-weight: 600;
    }
    
    .comparison-neutral {
        color: #6b7280;
        font-weight: 600;
    }
    
    /* Tooltip styling */
    .tooltip {
        position: relative;
        display: inline-block;
        cursor: help;
        color: #6b7280;
        margin-left: 5px;
    }
    
    .tooltip .tooltiptext {
        visibility: hidden;
        width: 200px;
        background-color: #1f2937;
        color: #fff;
        text-align: center;
        border-radius: 6px;
        padding: 5px 8px;
        position: absolute;
        z-index: 1;
        bottom: 125%;
        left: 50%;
        margin-left: -100px;
        opacity: 0;
        transition: opacity 0.3s;
        font-size: 0.8rem;
    }
    
    .tooltip:hover .tooltiptext {
        visibility: visible;
        opacity: 1;
    }
</style>
""", unsafe_allow_html=True)

# Cluster icons and colors
CLUSTER_ICONS = {
    #0: "🏃‍♀️",  # Active/Healthy
    #1: "⚖️",   # Balanced
    #2: "😴",   # Sedentary/Stressed
    #3: "💪",   # Fitness Enthusiast
    #4: "🧘‍♀️"   # Wellness Focused
}

CLUSTER_COLORS = {
    0: "#10b981",  # Green
    1: "#3b82f6",  # Blue
    2: "#ef4444",  # Red
    3: "#8b5cf6",  # Purple
    4: "#f59e0b"   # Orange
}

@st.cache_data
def load_data():
    """Load and cache the dataset"""
    try:
        with open("clustered_health_lifestyle_data.pkl", "rb") as f:
            df = pickle.load(f)
        return df
    except FileNotFoundError:
        st.error("Data file 'clustered_health_lifestyle_data.pkl' not found. Please ensure the file is in the same directory.")
        st.stop()

@st.cache_data
def prepare_model_and_pca(df):
    """Prepare the model and PCA for clustering"""
    features = ["Exercise_Freq", "Diet_Quality", "Stress_Level", "Sleep_Hours"]
    X = df[features]
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    # Convert back to DataFrame to preserve feature names
    X_scaled = pd.DataFrame(X_scaled, columns=features)
    # Fit KMeans
    n_clusters = df["Cluster"].nunique()
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(X_scaled)
    
    # PCA for visualization
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    
    return scaler, kmeans, pca, X_pca, features

def get_cluster_info(df, cluster_id):
    """Get comprehensive cluster information"""
    cluster_data = df[df["Cluster"] == cluster_id]
    
    if cluster_data.empty:
        return {
            "label": f"Cluster {cluster_id}",
            "description": "No description available",
            "icon": "",
            "color": "#6b7280"
        }
    
    return {
        "label": cluster_data["Cluster_Label"].iloc[0],
        "description": cluster_data["short_descriptions"].iloc[0],
        "icon": CLUSTER_ICONS.get(cluster_id, ""),
        "color": CLUSTER_COLORS.get(cluster_id, "#6b7280")
    }

def get_personalized_tips(user_values, cluster_info):
    """Generate personalized health tips based on user input"""
    tips = []
    exercise, diet, stress, sleep = user_values
    
    # Exercise tips
    if exercise < 3:
        tips.append("Consider increasing your exercise frequency. Aim for at least 150 minutes of moderate activity per week.")
    elif exercise > 5:
        tips.append("Great job on staying active! Make sure to include rest days for recovery.")
    
    # Diet tips
    if diet < 2:
        tips.append("Focus on improving your diet quality with more fruits, vegetables, and whole grains.")
    elif diet >= 2:
        tips.append("Excellent diet quality! Keep up the healthy eating habits.")
    
    # Stress tips
    if stress > 7:
        tips.append("Your stress levels are high. Consider meditation, yoga, or other stress-reduction techniques.")
    elif stress < 4:
        tips.append("You're managing stress well! Maintain your current coping strategies.")
    
    # Sleep tips
    if sleep < 7:
        tips.append("Try to get 7-9 hours of sleep per night for optimal health and recovery.")
    elif sleep > 9:
        tips.append("You're getting plenty of sleep! Make sure it's quality sleep with a consistent schedule.")
    
    return tips

def create_comparison_chart(user_values, df, cluster_id):
    """Create a comparison chart showing user vs cluster average"""
    features = ["Exercise_Freq", "Diet_Quality", "Stress_Level", "Sleep_Hours"]
    feature_labels = ["Exercise Frequency", "Diet Quality", "Stress Level", "Sleep Hours"]
    
    cluster_data = df[df["Cluster"] == cluster_id]
    cluster_means = [cluster_data[feature].mean() for feature in features]
    
    comparison_data = pd.DataFrame({
        'Metric': feature_labels,
        'Your Values': user_values,
        'Cluster Average': cluster_means
    })
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='Your Values',
        x=comparison_data['Metric'],
        y=comparison_data['Your Values'],
        marker_color='#667eea',
        opacity=0.8
    ))
    
    fig.add_trace(go.Bar(
        name='Cluster Average',
        x=comparison_data['Metric'],
        y=comparison_data['Cluster Average'],
        marker_color='#10b981',
        opacity=0.6
    ))
    
    fig.update_layout(
        title='Your Values vs Cluster Average',
        xaxis_title='Health Metrics',
        yaxis_title='Values',
        barmode='group',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Inter, sans-serif")
    )
    
    return fig

def create_cluster_visualization(df, X_pca, user_pca_point, predicted_cluster):
    """Create an interactive 2D scatter plot of clusters"""
    # Create DataFrame for plotting
    plot_df = df.copy()
    plot_df['PCA1'] = X_pca[:, 0]
    plot_df['PCA2'] = X_pca[:, 1]
    
    # Create the scatter plot
    fig = px.scatter(
        plot_df, 
        x='PCA1', 
        y='PCA2', 
        color='Cluster',
        hover_data=['Cluster_Label', 'Exercise_Freq', 'Diet_Quality', 'Stress_Level', 'Sleep_Hours'],
        color_discrete_map={cluster: CLUSTER_COLORS.get(cluster, '#6b7280') for cluster in plot_df['Cluster'].unique()},
        title='Health & Lifestyle Clusters (PCA Visualization)'
    )
    
    # Add user point
    fig.add_trace(go.Scatter(
        x=[user_pca_point[0]],
        y=[user_pca_point[1]],
        mode='markers',
        marker=dict(
            size=15,
            color='red',
            symbol='star',
            line=dict(width=2, color='white')
        ),
        name='Your Profile',
        hovertemplate='<b>Your Profile</b><br>Predicted Cluster: %{text}<extra></extra>',
        text=[f"Cluster {predicted_cluster}"]
    ))
    
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Inter, sans-serif"),
        height=500
    )
    
    return fig

def generate_pdf_report(user_values, cluster_info, tips, comparison_data):
    """Generate a PDF report of the user's health cluster analysis"""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    
    # Title
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        spaceAfter=30,
        textColor=colors.HexColor('#667eea'),
        alignment=1  # Center alignment
    )
    story.append(Paragraph("Health & Lifestyle Cluster Analysis Report", title_style))
    story.append(Spacer(1, 20))
    
    # Date
    date_style = ParagraphStyle(
        'DateStyle',
        parent=styles['Normal'],
        fontSize=12,
        alignment=1,
        textColor=colors.grey
    )
    story.append(Paragraph(f"Generated on: {datetime.now().strftime('%B %d, %Y')}", date_style))
    story.append(Spacer(1, 30))
    
    # User Input Section
    story.append(Paragraph("Your Health Profile", styles['Heading2']))
    
    user_data = [
        ['Metric', 'Your Value'],
        ['Exercise Frequency (days/week)', str(user_values[0])],
        ['Diet Quality (0-3)', str(user_values[1])],
        ['Stress Level (1-10)', str(user_values[2])],
        ['Sleep Hours', str(user_values[3])]
    ]
    
    user_table = Table(user_data)
    user_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#667eea')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    
    story.append(user_table)
    story.append(Spacer(1, 30))
    
    # Cluster Result Section
    story.append(Paragraph("Your Cluster Classification", styles['Heading2']))
    story.append(Paragraph(f"<b>Cluster:</b> {cluster_info['label']}", styles['Normal']))
    story.append(Paragraph(f"<b>Description:</b> {cluster_info['description']}", styles['Normal']))
    story.append(Spacer(1, 20))
    
    # Recommendations Section
    story.append(Paragraph("Personalized Recommendations", styles['Heading2']))
    for i, tip in enumerate(tips, 1):
        # Remove emoji from tip for PDF
        clean_tip = ''.join(char for char in tip if ord(char) < 127)
        story.append(Paragraph(f"{i}. {clean_tip}", styles['Normal']))
    
    doc.build(story)
    buffer.seek(0)
    return buffer

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>Health & Lifestyle Cluster Predictor</h1>
        <p>Discover your health profile and get personalized recommendations</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load data and prepare models
    df = load_data()
    scaler, kmeans, pca, X_pca, features = prepare_model_and_pca(df)
    
    # Sidebar
    with st.sidebar:
        st.markdown("""
        <div class="sidebar-header">
            Enter Your Health Profile
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### Exercise & Activity")
        exercise = st.slider(
            "Exercise Frequency (days/week)",
            min_value=0, max_value=7, value=3,
            help="How many days per week do you exercise for at least 30 minutes?"
        )
        
        st.markdown("### Nutrition")
        diet = st.slider(
            "Diet Quality",
            min_value=0, max_value=3, value=1,
            format="%d",
            help="0: Poor (mostly processed foods), 1: Fair (some healthy choices), 2: Good (balanced diet), 3: Excellent (very healthy)"
        )
        
        st.markdown("### Mental Health")
        stress = st.slider(
            "Stress Level",
            min_value=1, max_value=10, value=5,
            help="Rate your average stress level from 1 (very low) to 10 (very high)"
        )
        
        st.markdown("### Sleep")
        sleep = st.slider(
            "Sleep Hours",
            min_value=0.0, max_value=12.0, value=7.0, step=0.5,
            help="Average hours of sleep per night"
        )
        
        st.markdown("---")
        
        predict_button = st.button("Analyze My Health Profile", type="primary")
        
        if predict_button:
            st.session_state.prediction_made = True
            st.session_state.user_values = [exercise, diet, stress, sleep]
    
    # Main content
    if hasattr(st.session_state, 'prediction_made') and st.session_state.prediction_made:
        user_values = st.session_state.user_values
        
        # Make prediction
        # Make prediction
        input_df = pd.DataFrame([user_values], columns=features)  # Create DataFrame with feature names
        input_scaled = scaler.transform([user_values])
        predicted_cluster = kmeans.predict(input_scaled)[0]
        user_pca_point = pca.transform(input_scaled)[0]
        
        # Get cluster information
        cluster_info = get_cluster_info(df, predicted_cluster)
        
        # Display results
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown(f"""
            <div class="cluster-result">
                <h2>{cluster_info['icon']} {cluster_info['label']}</h2>
                <p style="font-size: 1.1rem; margin-bottom: 0;">Cluster {predicted_cluster}</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="cluster-description">
                <h3>Profile Description</h3>
                <p>{cluster_info['description']}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            # Quick stats
            st.markdown("""
            <div class="metric-card">
                <h3 style="color: #667eea; margin-bottom: 1rem;">Your Metrics</h3>
            </div>
            """, unsafe_allow_html=True)
            
            st.metric("Exercise Days", f"{user_values[0]}/week")
            st.metric("Diet Quality", f"{user_values[1]}/3")
            st.metric("Stress Level", f"{user_values[2]}/10")
            st.metric("Sleep Hours", f"{user_values[3]}h")
        
        # Visualization
        #st.markdown("---")
        #st.markdown("## 📊 Cluster Visualization")
        
        #col1, col2 = st.columns([3, 1])
        
        #ith col1:
            #fig = create_cluster_visualization(df, X_pca, user_pca_point, predicted_cluster)
            #st.plotly_chart(fig, use_container_width=True)
        
        #with col2:
            #st.markdown("### 🎯 Legend")
            #for cluster_id in sorted(df['Cluster'].unique()):
                #cluster_data = get_cluster_info(df, cluster_id)
                #color = cluster_data['color']
                #st.markdown(f"""
                #<div style="display: flex; align-items: center; margin-bottom: 0.5rem;">
                    #<div style="width: 15px; height: 15px; background-color: {color}; border-radius: 50%; margin-right: 10px;"></div>
                    #<span style="font-size: 0.9rem;">{cluster_data['icon']} {cluster_data['label']}</span>
                #</div>
                #""", unsafe_allow_html=True)
        
        # Comparison with cluster average
        #st.markdown("---")
        #st.markdown("## ⚖️ Compare with Your Cluster")
        
        #comparison_fig = create_comparison_chart(user_values, df, predicted_cluster)
        #st.plotly_chart(comparison_fig, use_container_width=True)
        
        # Personalized recommendations
        tips = get_personalized_tips(user_values, cluster_info)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="recommendation-box">
                <h3>Personalized Recommendations</h3>
            </div>
            """, unsafe_allow_html=True)
            
            for tip in tips:
                st.markdown(f"""
                <div class="tip-box">
                    {tip}
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            # Expandable sections
            with st.expander("Detailed Health Analysis", expanded=False):
                cluster_data = df[df["Cluster"] == predicted_cluster]
                
                st.write("**Cluster Statistics:**")
                st.write(f"- Average Exercise: {cluster_data['Exercise_Freq'].mean():.1f} days/week")
                st.write(f"- Average Diet Quality: {cluster_data['Diet_Quality'].mean():.1f}/3")
                st.write(f"- Average Stress Level: {cluster_data['Stress_Level'].mean():.1f}/10")
                st.write(f"- Average Sleep: {cluster_data['Sleep_Hours'].mean():.1f} hours")
                st.write(f"- Total people in this cluster: {len(cluster_data)}")
            
            with st.expander("Improvement Goals", expanded=False):
                st.write("**Suggested Goals:**")
                if user_values[0] < 5:
                    st.write("- Increase exercise to 5+ days/week")
                if user_values[1] < 2:
                    st.write("- Improve diet quality score to 2+")
                if user_values[2] > 6:
                    st.write("- Reduce stress level below 6")
                if user_values[3] < 7:
                    st.write("- Aim for 7-9 hours of sleep")
        
        # Export options
        st.markdown("---")
        st.markdown("## Export Your Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # CSV export
            results_df = pd.DataFrame({
                'Metric': ['Exercise_Freq', 'Diet_Quality', 'Stress_Level', 'Sleep_Hours', 'Predicted_Cluster', 'Cluster_Label'],
                'Value': [*user_values, predicted_cluster, cluster_info['label']]
            })
            
            csv = results_df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"health_cluster_analysis_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
        
        with col2:
            # PDF export
            try:
                pdf_buffer = generate_pdf_report(user_values, cluster_info, tips, None)
                st.download_button(
                    label="Download PDF Report",
                    data=pdf_buffer,
                    file_name=f"health_analysis_report_{datetime.now().strftime('%Y%m%d')}.pdf",
                    mime="application/pdf"
                )
            except Exception as e:
                st.warning("PDF generation requires additional setup. CSV export is available.")
        
        with col3:
            if st.button("🔄 Start New Analysis"):
                st.session_state.prediction_made = False
                st.rerun()
    
    else:
        # Welcome message
        st.markdown("""
        <div class="metric-card">
            <h3>Welcome to Your Health Journey!</h3>
            <p>This intelligent tool analyzes your lifestyle patterns and provides personalized health insights.</p>
            <h4>How it works:</h4>
            <ol>
                <li><strong>Input your data:</strong> Use the sidebar to enter your exercise, diet, stress, and sleep information</li>
                <li><strong>Get your cluster:</strong> Our AI will classify you into a health lifestyle cluster</li>
                <li><strong>Receive insights:</strong> Get personalized recommendations and visualizations</li>
                <li><strong>Track progress:</strong> Export your results and track improvements over time</li>
            </ol>
            <div style="background: #f0f9ff; padding: 1rem; border-radius: 8px; margin-top: 1rem; border-left: 4px solid #0ea5e9;">
                <strong>💡 Pro Tip:</strong> Be honest with your inputs for the most accurate recommendations!
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Sample visualization
        #st.markdown("## 📊 Sample Cluster Visualization")
        #sample_fig = create_cluster_visualization(df, X_pca, [0, 0], 0)
        #st.plotly_chart(sample_fig, use_container_width=True)
        
        st.info("**Get started by entering your health information in the sidebar!**")

if __name__ == "__main__":
    main()