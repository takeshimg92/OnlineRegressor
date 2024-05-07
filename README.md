# OnlineRegressor
Streamlit Cloud app for no-code regression and classification tasks.
This code was developed to be used as a demonstration tool in lectures about data science for managers who do not have a background in analytics.

All files are already pre-built. Only re-run `preprocess_data.py` to generate files for students to use if you find any errors - train, test and test with ground truth are available at `data/processed`.

## Running the platform

Locally, one can run the Streamlit app from root: 
```bash
streamlit run Home.py
```

Otherwise, you can fork and deploy this repo to [Streamlit Community Cloud](https://streamlit.io/cloud); this is how I am using it to make the platform available for students.

### Comparing student results

After students have downloaded files using the platform, trained models, and gotten the predictions as a column `Prediction` on a .xlsx file, they must send it to the lecturer (I have been using email). 

Save the files wherever you want on your machine. Then:
* Edit the file `notebooks/compare_student_models.ipynb`; specifically, the variables `folder` and `files` on the first cell.
    * `folder` should be the directory containing the student model outputs. My default is `'../data/student_outputs/true_run/'`, ie. inside the `data` folder of this repo
    * `files` is a list of tuples. Each tuple contains a label for each group + the filename.xlsx of their predictions. The label is what will appear in the plots. For example,

    ```python
    files = [('Grupo Azul', 'azul_preds.xlsx'), 
             ('Grupo da Maria', 'maria.xlsx')]
    ```

    is a valid way.,
* If you want, you can also change some auxiliary variables. They are:
    * `target_delinq = 0.1`: the maximum allowed default of the portfolio (currently 10%)
    * `lucro_bom_pagador = 500`: value in R$ of profit from a good payer;
    * `preju_mau_pagador = 300`: loss in R$ due to a payer default.

* Run the notebook. The figure comparing the groups and profits should be generated below. You can then copy it (I usually just screenshot) to a slide to show to the groups.


### Necessary libraries (using Python 3.12.2)

You can check the output of the pip freeze in `requirements.txt`. We are currently using the following libraries in Python 3.12.2:

* scikit-learn=1.3.0
* lightgbm=4.3.0
* xgboost=2.0.3
* pandas=2.2.2
* matplotlib=3.8.4
* plotly=5.22.0
* streamlit=1.34.0
* openpyxl=3.1.2
* xlsxwriter=3.2.0
* category_encoders=2.6.3
* tqdm=4.66.2
* notebook=6.4.8