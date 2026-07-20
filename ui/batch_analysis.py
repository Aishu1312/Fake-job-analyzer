import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from services.analysis_service import process_batch
from services.export_service import export_csv, export_excel, export_pdf

def render_batch_analysis():
    st.markdown("## 📂 Batch Job Analysis")
    st.markdown("Upload multiple job descriptions (PDF, DOCX, TXT, CSV, JSON, HTML, MD) or an entire folder.")
    
    uploaded_files = st.file_uploader(
        "Drag and drop files or folders here",
        accept_multiple_files=True,
        type=['pdf', 'docx', 'txt', 'csv', 'json', 'html', 'md']
    )
    
    if uploaded_files:
        if st.button("🚀 Start Batch Analysis", type="primary"):
            run_batch_analysis(uploaded_files)

def run_batch_analysis(uploaded_files):
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    def update_progress(current, total):
        progress = current / total
        progress_bar.progress(progress)
        status_text.markdown(f"**Analyzing {current}/{total}...**")
        
    # Run the heavy processing
    results, duplicates, processing_time = process_batch(uploaded_files, progress_callback=update_progress)
    
    progress_bar.empty()
    status_text.empty()
    
    # Render Dashboard
    render_dashboard(results, duplicates, processing_time)

def render_dashboard(results, duplicates, processing_time):
    st.success("Analysis Complete!")
    
    # Calculate Stats
    total = len(results)
    success_results = [r for r in results if r["Status"] == "Success"]
    fake = len([r for r in success_results if r["Verdict"] == "Fake"])
    legit = len([r for r in success_results if r["Verdict"] == "Legitimate"])
    suspicious = len([r for r in success_results if r["Verdict"] == "Suspicious"])
    avg_score = sum(r["Fraud Score"] for r in success_results) / max(len(success_results), 1)
    
    summary_stats = {
        "total": total,
        "legit": legit,
        "suspicious": suspicious,
        "fake": fake,
        "avg_score": avg_score
    }
    
    # Display Top Metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    def metric_card(col, label, value, color_class=""):
        col.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value {color_class}">{value}</div>
        </div>
        """, unsafe_allow_html=True)
        
    metric_card(col1, "Total Files", total)
    metric_card(col2, "Legitimate", legit)
    metric_card(col3, "Suspicious", suspicious)
    metric_card(col4, "Fake", fake)
    metric_card(col5, "Avg Risk", f"{avg_score:.1f}%")
    
    st.markdown(f"<p style='text-align: right; color: gray; font-size: 0.9em;'>Processing Time: {processing_time:.2f} seconds</p>", unsafe_allow_html=True)
    
    st.divider()
    
    # Charts Section
    st.markdown("### 📊 Analytics Dashboard")
    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        # Pie Chart
        labels = ['Legitimate', 'Suspicious', 'Fake']
        values = [legit, suspicious, fake]
        colors = ['#10b981', '#f59e0b', '#ef4444']
        
        fig_pie = go.Figure(data=[go.Pie(labels=labels, values=values, marker_colors=colors, hole=.4)])
        fig_pie.update_layout(title_text="Risk Distribution", margin=dict(t=40, b=0, l=0, r=0))
        st.plotly_chart(fig_pie, use_container_width=True)
        
    with chart_col2:
        # Histogram of Fraud Scores
        scores = [r["Fraud Score"] for r in success_results]
        fig_hist = px.histogram(x=scores, nbins=10, labels={'x':'Fraud Score', 'y':'Count'}, title="Fraud Score Distribution", color_discrete_sequence=['#8b5cf6'])
        st.plotly_chart(fig_hist, use_container_width=True)
        
    st.divider()
    
    # Duplicates Warning
    if duplicates:
        st.warning(f"⚠️ Detected {len(duplicates)} potential duplicate or near-duplicate job descriptions in this batch.")
        with st.expander("View Duplicates"):
            for dup in duplicates:
                file1 = success_results[dup['doc1_idx']]['Filename']
                file2 = success_results[dup['doc2_idx']]['Filename']
                st.write(f"- **{file1}** and **{file2}** ({dup['similarity']}% similar)")
                
    st.divider()
    
    # Batch Result Table
    st.markdown("### 📋 Detailed Results")
    
    # Prepare Dataframe for table
    df_data = []
    for r in results:
        if r["Status"] == "Success":
            df_data.append({
                "Filename": r["Filename"],
                "Verdict": r["Verdict"],
                "Risk Level": r["Risk Level"],
                "Fraud Score": r["Fraud Score"],
                "AI Confidence": r["AI Confidence"]
            })
        else:
            df_data.append({
                "Filename": r["Filename"],
                "Verdict": "Error",
                "Risk Level": "-",
                "Fraud Score": "-",
                "AI Confidence": "-"
            })
            
    df = pd.DataFrame(df_data)
    
    # Render table (st.dataframe has built-in sort and search starting 1.30+)
    st.dataframe(df, use_container_width=True)
    
    # Individual Reports
    st.markdown("### 📄 Individual Reports")
    for r in success_results:
        with st.expander(f"{r['Filename']} - {r['Verdict']} (Score: {r['Fraud Score']})"):
            st.markdown(f"**Verdict:** {r['Verdict']}")
            st.markdown(f"**Risk Level:** {r['Risk Level']}")
            st.markdown(f"**Fraud Score:** {r['Fraud Score']}/100")
            st.markdown(f"**AI Confidence:** {r['AI Confidence']}%")
            
            st.markdown("#### Detailed AI Findings")
            stats = r["Risk Stats"]
            st.write(f"- **Scam Probability:** {stats.get('Scam Probability', 0)}%")
            st.write(f"- **Urgency Manipulation:** {stats.get('Urgency Manipulation', 0)}/100")
            st.write(f"- **Salary Manipulation:** {stats.get('Salary Manipulation', 0)}/100")
            st.write(f"- **Grammar Quality:** {stats.get('Grammar Quality', 100)}/100")
            st.write(f"- **Contact Authenticity:** {stats.get('Contact Authenticity', 100)}/100")
            st.write(f"- **Missing Information:** {stats.get('Missing Information', 0)}/100")
            
            reasons = stats.get("Reasons", [])
            if reasons:
                st.markdown("#### Suspicious Indicators")
                for reason in reasons:
                    st.markdown(f"- 🚩 {reason}")
                    
    st.divider()
    
    # Export Options
    st.markdown("### 💾 Export Reports")
    export_col1, export_col2, export_col3 = st.columns(3)
    
    with export_col1:
        csv_bytes = export_csv(results)
        st.download_button(
            label="📄 Download CSV",
            data=csv_bytes,
            file_name="fake_job_batch_results.csv",
            mime="text/csv",
            use_container_width=True
        )
        
    with export_col2:
        excel_bytes = export_excel(results)
        st.download_button(
            label="📊 Download Excel",
            data=excel_bytes,
            file_name="fake_job_batch_results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )
        
    with export_col3:
        pdf_bytes = export_pdf(results, summary_stats)
        st.download_button(
            label="📕 Download PDF Report",
            data=pdf_bytes,
            file_name="fake_job_batch_report.pdf",
            mime="application/pdf",
            use_container_width=True
        )
