import streamlit as st
import pandas as pd


import numpy as np

import pickle


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
    target_acc = big_acc -np.random.rand()*0.1 # 0.5% lower than the big accuracy
    
    
    del selected_df['little'] 
    del selected_df['big']
    del selected_df['thresholds']
   
    close_idx = pd.Series((selected_df['acc_little_big'] - target_acc)).abs().idxmin()

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
    #filtered_df= selected_df.iloc[[close_idx]]
    filtered_df = filtered_df.apply(lambda col: col.round(3) if pd.api.types.is_numeric_dtype(col) else col)
    filtered_df= filtered_df[new_column_order]
  
    print(f"filtered_df['Savings in GMACs from Big (%)']",filtered_df['Savings in GMACs from Big (%)'])
    
    if filtered_df['Savings in GMACs from Big (%)'].values[0] >-1.:
        st.markdown("""
            <div style="background-color: #f8d7da; color: #721c24; padding: 10px; font-size: 18px; font-weight: bold; border-radius: 5px; border: 1px solid #f5c6cb;">
               Failed to speedup without a noticeable drop in accuracy. Please select another Little model.
            </div>
        """, unsafe_allow_html=True)
    # filtered_df = selected_df.iloc[[close_idx]]
    # filtered_df = filtered_df[new_column_order]
    else:
        


    # CSS to style the dataframe
        custom_css = """
        <style>
            /* General body styling */
            body {
                color: var(--text-color, #000); /* Default to black text for light mode */
                background-color: var(--background-color, #fff); /* Default to white background */
            }

            /* Styling for the dataframe (table) */
            .dataframe {
                width: 100% !important;
                height: auto !important;
                font-size: 16px !important;
                color: var(--text-color, #000); /* Text color depends on mode */
                background-color: var(--table-bg-color, #f8f8f8); /* Light background for tables */
                border-collapse: collapse;
            }

            /* Highlight Savings: more contrast for both modes */
            .highlight_savings {
                background-color: var(--highlight-savings-bg, #FFD700) !important; /* Gold for light/dark modes */
                color: var(--highlight-savings-text, #333); /* Dark text on gold for better contrast */
            }

            /* Highlight Accuracy: color contrast for both modes */
            .highlight_accuracy {
                background-color: var(--highlight-accuracy-bg, #32CD32) !important; /* Lime Green */
                color: var(--highlight-accuracy-text, #FFF); /* White text on lime green */
            }

            /* Table Headers */
            .dataframe thead th {
                background-color: var(--header-bg-color, #f1f1f1); /* Light gray for table headers */
                color: var(--header-text-color, #333); /* Dark text for headers */
            }

            /* Borders for table */
            .dataframe, .dataframe th, .dataframe td {
                border: 1px solid var(--border-color, #ddd); /* Light gray borders */
            }

            /* Dark mode detection */
            @media (prefers-color-scheme: dark) {
                body {
                    --text-color: #FFF; /* White text for dark mode */
                    --background-color: #222; /* Dark background */
                }

                .dataframe {
                    --table-bg-color: #333; /* Darker table background for dark mode */
                }

                .highlight_savings {
                    --highlight-savings-bg: #FFD700; /* Gold highlight for dark mode */
                    --highlight-savings-text: #333; /* Dark text */
                }

                .highlight_accuracy {
                    --highlight-accuracy-bg: #32CD32; /* LimeGreen for dark mode */
                    --highlight-accuracy-text: #FFF; /* White text on LimeGreen */
                }

                .dataframe thead th {
                    --header-bg-color: #444; /* Darker header background */
                    --header-text-color: #FFF; /* White text in headers */
                }

                .dataframe, .dataframe th, .dataframe td {
                    --border-color: #444; /* Dark borders for tables */
                }
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
