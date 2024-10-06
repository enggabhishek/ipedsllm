import streamlit as st
import pandas as pd
import altair as alt

# Load the updated DataFrame with the table names
file_path = '/Users/trishulchowdhury/Desktop/FINAL_folder/final_app.xlsx'
df = pd.read_excel(file_path)

# Streamlit app
st.title("Dashboard")

# Sidebar for question selection
selected_question = st.sidebar.selectbox("Select a Question", df['Question'].unique())

# Filter the DataFrame based on the selected question
filtered_df = df[df['Question'] == selected_question].iloc[0]

# Display the actual table info
st.subheader("Actual Table Info")
st.write(filtered_df['Actual Table Info'])

# Display the similarity score
st.write(f"Similarity Score: {filtered_df['Actual Table Info Score']}")

# Prepare data for visualization
scores_df = pd.DataFrame({
    'Table Info': ['Actual Table Info', 'Top 1 Similar Table Info', 'Top 2 Similar Table Info', 'Top 3 Similar Table Info'],
    'Similarity Score': [
        filtered_df['Actual Table Info Score'],
        filtered_df['Top 1 Score'],
        filtered_df['Top 2 Score'],
        filtered_df['Top 3 Score']
    ],
    'Table Names': [
        filtered_df['Actual Table Names'],
        filtered_df['Top 1 Table Names'],
        filtered_df['Top 2 Table Names'],
        filtered_df['Top 3 Table Names']
    ]
})

# Altair Bar Chart with hover tooltip for table groups
chart = alt.Chart(scores_df).mark_bar().encode(
    x=alt.X('Table Info', sort=None),
    y='Similarity Score',
    tooltip=['Table Info', 'Similarity Score', 'Table Names']
).properties(
    title='Similarity Scores with Table Groups'
).interactive()

st.altair_chart(chart, use_container_width=True)

# Additional sidebar information
st.sidebar.markdown("### Table Groups")
st.sidebar.write(f"**Actual Table Info Group:** {filtered_df['Actual Table Names']}")
st.sidebar.write(f"**Top 1 Similar Table Info Group:** {filtered_df['Top 1 Table Names']}")
st.sidebar.write(f"**Top 2 Similar Table Info Group:** {filtered_df['Top 2 Table Names']}")
st.sidebar.write(f"**Top 3 Similar Table Info Group:** {filtered_df['Top 3 Table Names']}")

# Add a button to view the entire table_info for all groups
if st.button('Show All Table Info'):
    st.subheader("Full Table Info for All Groups")
    
    st.markdown("**Actual Table Info:**")
    st.write(filtered_df['Actual Table Info'])
    
    st.markdown("**Top 1 Similar Table Info:**")
    st.write(filtered_df['Top 1 Similar Table Info'])
    
    st.markdown("**Top 2 Similar Table Info:**")
    st.write(filtered_df['Top 2 Similar Table Info'])
    
    st.markdown("**Top 3 Similar Table Info:**")
    st.write(filtered_df['Top 3 Similar Table Info'])
