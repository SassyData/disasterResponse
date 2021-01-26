import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('messages_df.csv', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():

    # extract data needed for visuals
    gen_count = df.groupby('genre').count()['message']
    gen_per = round(100 * gen_count / gen_count.sum(), 2)
    gen = list(gen_count.index)

    def organise_data_for_graph(df) :
        cat_counts = df.iloc[:, 4 :].apply(pd.to_numeric)
        cat_counts = pd.concat([cat_counts, df.iloc[:, 3]], axis = 1)
        cat_counts = cat_counts.sum().reset_index(drop = False)
        cat_counts.columns = ['category', 'count']
        return cat_counts
    graph_data = organise_data_for_graph(df)

    # create visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=graph_data['category'],
                    y=graph_data['count']
                )
            ],

            'layout': {
                'title': 'Distribution of Message Categories',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Category"
                }
            }
        },
        {
            "data" : [
                {
                    "type" : "pie",
                    "uid" : "f4de1f",
                    "hole" : 0.4,
                    "name" : "Genre",
                    "pull" : 0,
                    "domain" : {
                        "x" : gen_per,
                        "y" : gen
                    },
                    "marker" : {
                        "colors" : [
                            "#7fc97f",
                            "#beaed4",
                            "#fdc086"
                        ]
                    },
                    "textinfo" : "label+value",
                    "hoverinfo" : "all",
                    "labels" : gen,
                    "values" : gen_count
                }
            ],
            "layout" : {
                "title" : "Count and Percent of Messages by Genre"
            }
        }
    ]

    # encode plotly graphs in JSON
    ids = [f"graph-{i}" for i, _ in enumerate(graphs)]
    graph_json = json.dumps(graphs, cls = plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graph_json)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '')

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()