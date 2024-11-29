import streamlit as st
import pandas as pd
import os
import pickle
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt

@st.cache
def load_data():
    filepath = "all_results_filtered.pkl"
    with open (filepath,"rb") as f:
        data = pickle.load(f)
    return data
data = load_data()


with open('models.pkl', 'rb') as f:
    models = pickle.load(f)
models_list = list(models.keys())
sorted_keys = sorted(models.keys(), key=lambda x: models[x]['macs_official'])
sorted_keys = [f for f in sorted_keys if 'intern' not in f]
sorted_keys = [f for f in sorted_keys if 'deit3_h_224_v2' not in f]

sorted_keys_index_dict = dict(zip(sorted_keys,range(len(sorted_keys))))
# Sidebar: Little Model Selection
st.sidebar.header("Select Models")
little_models = sorted_keys[:-1]#list(set([key[0] for key in data.keys()]))
big_models = sorted_keys
big_models = sorted_keys[1:]  # List of all big models
selected_big_model = st.sidebar.selectbox("Select Big Model", big_models)

# Sidebar: Little Model Selection
if selected_big_model:
    # Filter little models based on the selected big model
    selected_big_model_indx = sorted_keys_index_dict[selected_big_model]
    
    # Ensure that little models come before the selected big model in the sorted list
    little_models = sorted_keys[:selected_big_model_indx]  # Little models are the ones before the selected big model
    selected_little_model = st.sidebar.selectbox("Select Little Model", little_models)

# Main View: Display Data
if selected_little_model and selected_big_model:
    st.title("Interactive Little-Big Model Dashboard")
    st.write(f"Analyzing **{selected_big_model}** as the big model and **{selected_little_model}** as the little model")

    # Get the selected DataFrame
    selected_little_model='little_'+selected_little_model
    selected_big_model='_big_'+selected_big_model

    
    selected_df = data[selected_little_model+selected_big_model]
    
    little_model_name= str(selected_little_model.split('little_')[1].upper())

    big_model_name = str(selected_big_model.split('_big_')[1].upper())

   
    selected_df['Little GMACs']= selected_df['little']['macs_official']/1e9
    selected_df['Big GMACs']= selected_df['big']['macs_official']/1e9


    big_acc = selected_df['acc_big']
    target_acc = big_acc# -np.random.rand()*0.1 # 0.5% lower than the big accuracy
    close_idx = pd.Series((selected_df['acc_little_big'] - target_acc)).abs().idxmin()
    
    del selected_df['little'] 
    del selected_df['big']
    del selected_df['thresholds']
   
    

    selected_df = pd.DataFrame(selected_df)
    selected_df.rename(columns={'acc_little_big':'Little-Big Accuracy','GMACs_little_big':'Little-Big GMACs','acc_big':'Big Acc','acc_little':'Little ACC','delta_accuracy_big_pct':'Acc Change From Big Model (%)',
                                'delta_macs_big_pct':'Savings in GMACs from Big (%)', 'solvable_little':'(%) of Samples Solvable by Little Model'
                                  },inplace=True)
    selected_df['Big'] = big_model_name
    selected_df['Little'] = little_model_name
    new_column_order = [
    'Little', 'Big', 
    'Little ACC', 'Big Acc', 
    'Little-Big Accuracy', 
    'Little-Big GMACs', 
    'Savings in GMACs from Big (%)',
    'Acc Change From Big Model (%)',
    'Little GMACs',
    'Big GMACs',
    '(%) of Samples Solvable by Little Model'
]
    
    #selected_df = selected_df.apply(lambda col: col.round(3) if pd.api.types.is_numeric_dtype(col) else col)
    filtered_df = selected_df[selected_df['Little-Big Accuracy'] >= target_acc].iloc[[0]]
    filtered_df = filtered_df.apply(lambda col: col.round(3) if pd.api.types.is_numeric_dtype(col) else col)
    filtered_df= filtered_df[new_column_order]
    # filtered_df = selected_df.iloc[[close_idx]]
    # filtered_df = filtered_df[new_column_order]



# CSS to style the dataframe
    custom_css = """
        <style>
            .dataframe {
                width: 100% !important;
                height: auto !important;
                font-size: 16px !important;
            }
            .highlight_savings {
                background-color: yellow !important;
            }
            .highlight_accuracy {
                background-color: lightgreen !important;
            }
        </style>
    """

    # Apply the custom CSS styling
    st.markdown(custom_css, unsafe_allow_html=True)

    # Highlight the 'Savings in GMACs from Big (%)' and 'Acc Change From Big Model (%)' columns
    filtered_df_styled = filtered_df.copy()

    # Highlight "Savings in GMACs from Big (%)" column in yellow
    filtered_df_styled['Savings in GMACs from Big (%)'] = filtered_df_styled['Savings in GMACs from Big (%)'].apply(
        lambda x: f'<span class="highlight_savings">{x}</span>' if pd.notna(x) else x
    )

    # Highlight "Acc Change From Big Model (%)" column in light green
    filtered_df_styled['Acc Change From Big Model (%)'] = filtered_df_styled['Acc Change From Big Model (%)'].apply(
        lambda x: f'<span class="highlight_accuracy">{x}</span>' if pd.notna(x) else x
    )

    # Display the DataFrame as HTML, transposed, with the highlights
    st.markdown(filtered_df_styled.T.to_html(escape=False), unsafe_allow_html=True)



else:
    st.write("Please select a little model and a big model from the sidebar.")
