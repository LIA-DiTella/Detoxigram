
from datasets import load_dataset
import os
import pandas as pd
from classifiers_classes_api.hate_bert_classifier import hate_bert_classifier
from classifiers_classes_api.perspective_classifier import perspective_classifier
from classifiers_classes_api.gpt_classifier import gpt_classifier
# from classifiers_classes_api.llama_cpp_classifier import llama_cpp_classifier
from classifiers_classes_api.multi_bert_classifier import multi_bert_classifier
from classifiers_classes_api.mistral_API_classifier import mistral_classifier

import matplotlib.pyplot as plt
import seaborn as sns
import time

def convert_toxicity_to_int(text):
        if (text == "4 - Extremely toxic"): return 4
        if (text == "3 - Highly toxic"): return 3
        if (text == "2 - Moderately toxic"): return 2
        if (text == "1 - Slightly toxic"): return 1
        if (text == "0 - Non-toxic"): return 0
        print("Hay un error bro")

def isAttentionCheck(text):
    if (text == "Donald Trump received 0 votes in the last elections"): return True
    if (text == "Joe Biden is the current president of the United States"): return True
    if (text == "Donald Trump received 0 votes in the last elections"): return True
    if (text == "The president of the United States is chosen by lottery."): return True
    if (text == "I believe that if we all hold hands and love each other despite our differences, we can solve many of the world's issues. Let's fill our hearts with empathy and watch the magic happen!"): return True
    if (text == "Pretty much as much as easy solution torture them all kill them get rid of them they're garbage they're useless they have no remorse they have no capable of sympathy they deserve destruction"): return True
    return False

def createPlot(ai_labels, user_labels, ai_name):

    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=user_labels, y=ai_labels)
    plt.title(f'Humanos vs. {ai_name} Toxicity Scores')
    plt.xlabel('Valores toxicidad promedio personas')
    plt.ylabel(f'Valores toxicidad promedio {ai_name}')
    plt.grid(True)
    plt.savefig(f'{ai_name}.png')


def plot_toxicity_comparison(human_toxicity_scores, ai_toxicity_scores):

    plt.figure(figsize=(8, 6))

    # Plot human toxicity scores
    sns.scatterplot(x=human_toxicity_scores, y=human_toxicity_scores, label='Human', color='blue')

    # Plot AI toxicity scores
    for ai_model, scores in ai_toxicity_scores.items():
        sns.scatterplot(x=human_toxicity_scores, y=scores, label=ai_model)

    plt.title('Human vs. AI Toxicity Scores')
    plt.xlabel('Toxicidad promedio humanos')
    plt.ylabel('Toxicidad AI')
    plt.grid(True)
    plt.legend()
    plt.savefig("comparacion.png")


def plot_toxicity_models_scatter_plot(model, model_name, file, columns):
    user_labels = []
    model_predictions = []
    start = time.time()

    for column in columns:
        if (isAttentionCheck(column)): continue #me salteo la columna

        #print("Mensaje:", column)
        message = column
        valid_responses = 0
        toxicity_message_users = 0
        first_value_column = True #la primer fila tiene basura

        for value in file[column]:
            if (first_value_column): 
                first_value_column = False
                continue
            if pd.notna(value):
                column_toxicity = convert_toxicity_to_int(value)
                toxicity_message_users += column_toxicity
                valid_responses += 1
        
        real_toxicity = toxicity_message_users/valid_responses
        predicted_toxicity = model.predictToxicity(column)

        user_labels.append(real_toxicity)
        model_predictions.append(predicted_toxicity[1])

    end = time.time()
    
    createPlot(model_predictions, user_labels, model_name)
    print(f"El timepo que demoró el procesamiento del rest fue de {end - start} segundos. ")
    print(f"El tiempo promedio por mensaje fue de{(end - start)/len(columns)} segundos. ")
    return model_predictions, user_labels


def binary_toxicity_evaluation(model, file, columns):
    correct_predictions = []

    for column in columns:
        if (isAttentionCheck(column)): continue #me salteo la columna
        #print("Mensaje:", column)
        message = column
        valid_responses = 0
        toxicity_message_users = 0
        first_value_column = True #la primer fila tiene basura

        for value in file[column]:
            if (first_value_column): 
                first_value_column = False
                continue
            if pd.notna(value):
                column_toxicity = convert_toxicity_to_int(value)
                toxicity_message_users += column_toxicity
                valid_responses += 1
        real_toxicity = toxicity_message_users/valid_responses
        predicted_toxicity = model.predictToxicity(column)
        #print(real_toxicity)
        #print(predicted_toxicity)

        if (real_toxicity >= 2 and predicted_toxicity[0]) or (real_toxicity < 2 and not predicted_toxicity[0]):
            correct_predictions.append(1) 
        else: correct_predictions.append(0)

    percentage_correct_classifications = sum(correct_predictions) / len(correct_predictions)
    return percentage_correct_classifications


def main():

    #leo el archivo
    script_dir = os.path.dirname(os.path.realpath(__file__))
    # relative_path = os.path.join('..', 'dataset/testing_datasets/')
    file_path = "/Users/ezecotton/Desktop/Detoxigram/dataset/testing_datasets/Detoxigram - Surveys - Final.csv"
    df = pd.read_csv(file_path)
    columnas_18_hasta_final= df.iloc[:, 18:-1] 

    # #cargo clasificadores
    
    # mixtral = mistral_classifier("mistral_classifier", "prompt_template_few_shot")
    gpt = gpt_classifier("gpt-3.5-turbo", os.environ["OPENAI_API_KEY"], templatetype= "prompt_template_few_shot")
    # toxigen_bert = hate_bert_classifier("../model_evaluation_scripts/classifiers_classes_api/toxigen_hatebert")
    #perspective = perspective_classifier("AIzaSyBLcQ87gA8wc_960mNzT6uCiDkUWRoz6mE" ,attributes=["TOXICITY"])
    # multi_bert = multi_bert_classifier("../model_evaluation_scripts/classifiers_classes_api/multi_label_bert")
    # bert_scores, user_scores = plot_toxicity_models_scatter_plot(toxigen_bert, "Hate Bert - Toxi Gen", df, columnas_18_hasta_final)
    #multi_scores, user_scores = plot_toxicity_models_scatter_plot(multi_bert, "Hate Bert - Toxi Gen", df, columnas_18_hasta_final)
    # multi_bert.predictToxicityFile("annotated_test.csv")


    

    #plot_toxicity_comparison(user_scores, {"HateBert": bert_scores, "GPT 3.5": gpt_scores, "Multi_BERT": multi_scores})
if __name__ == "__main__":
    main()