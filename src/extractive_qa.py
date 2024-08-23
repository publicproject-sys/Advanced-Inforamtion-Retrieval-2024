############ Extractive QA - Part 3

# Packages
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline 
from core_metrics import compute_f1, compute_exact
import pandas as pd
import pickle
import ast
from collections import defaultdict
from time import time
from typing import List
import torch
import warnings
import os
import numpy as np
import argparse


'''
In this script the code needed for part 3 of the assignment is implemented.
This script performs extractive question-answering (QA) on a dataset using a pre-trained model gathered from huggingface.
It includes functions for loading data, running the QA model, and calculating evaluation metrics.
We fist load the necessary data and extract top-ranked documents for each query of the best models reranking from part 2 of the assignment.
We then run the QA model on the data.
Finally, evaluation metrics are calculated. 

'''



############################################################################################
##### Functions
############################################################################################
# Define the function to get the top-ranked doc_id for each query_id
def get_top_ranked_docs(data, answers_df):
    
    """
    Identify the top-ranked documents for the reranked data (data)
    
    In order to get the overlap we merge the dataset with the validation set (used to obtain the reranking)
    as well as the answers_df
    
    Parameters:
    data (pd.DataFrame): DataFrame containing columns 'queryid' and 'rank', among others, representing the document rankings for various queries.
    answers_df (pd.DataFrame): DataFrame containing columns 'queryid' and 'documentid', along with additional information (e.g., answers).

    Returns:
    pd.DataFrame: DataFrame containing the merged results of the top-ranked documents, the validation set, and additional information from answers_df.
    """
    
    # Sort the data by query_id and rankf
    sorted_data = data.sort_values(by=['queryid', 'rank'])
    # Drop duplicates to keep only the top-ranked doc_id for each query_id
    top_ranked_docs = sorted_data.drop_duplicates(subset=['queryid'], keep='first')

    # load msmarco test set
    #msmarco_testset = pd.read_csv("data/Part-2/msmarco_tuples.test.tsv", sep="\t", header=None)
    #msmarco_testset.columns=["queryid", "documentid", "query-text", "document-text"]

    # validation set 
    msmarco_validation = pd.read_csv("data/Part-2/msmarco_tuples.validation.tsv", sep="\t", header=None)
    msmarco_validation.columns=["queryid", "documentid", "query-text", "document-text"]


    # merge datasets
    # Rename 'doc_id' to 'documentid' in top1_df
    top_ranked_docs.rename(columns={'doc_id': 'documentid'}, inplace=True)

    # Merge the DataFrames on 'queryid' and 'documentid'
    #merged_df = pd.merge(top_ranked_docs, msmarco_testset, on=['queryid', 'documentid'], how='inner')
    merged_df = pd.merge(top_ranked_docs, msmarco_validation, on=['queryid', 'documentid'], how='inner')

    result = merged_df.merge(answers_df, on=["queryid", "documentid"])

    return result

def run_model(data, model, tokenizer):
    
    """
    Run a question-answering model on the provided data.

    Parameters:
    data (pd.DataFrame): DataFrame containing columns 'query-text' and 'document-text'.
    model: The pre-trained model to be used for question-answering.
    tokenizer: The tokenizer corresponding to the model.

    Returns:
    pd.DataFrame: DataFrame with an additional column 'answer' containing the model's answers to the questions.
    """

    # check whether there is a GPU
    device = 0 if torch.cuda.is_available() else -1
    if(device == 0):
        print("GPU available")
    else:
        print("No GPU availabe, CPU will be used")

    
    n = len(data)
    answers = []
    pipe = pipeline('question-answering', model=model, tokenizer=tokenizer, device = device)
    for i in range(n):
        
        pipe_in = {"question": data.loc[i, "query-text"], "context": data.loc[i, "document-text"]}
        
        answer = pipe(pipe_in)["answer"]
        answers.append(answer)
        print(f"\rProcessing step {i} of {n}", end="", flush=True)

    print()
    data["answer"] = answers
    return data



def calc_metrics(df, func, ground_truth, answer):
    
    """
    Calculate metrics by comparing ground truth answers to model-generated answers using a specified function.

    Parameters:
    df (pd.DataFrame): DataFrame containing the data.
    func (function): Function to calculate the score between ground truth and generated answer.
    ground_truth (str): Column name containing the ground truth answers.
    answer (str): Column name containing the model-generated answers.

    Returns:
    float: The mean score calculated across all rows in the DataFrame.
    """

    # calc scores
    scores =[max(
        [func(df.loc[i, ground_truth][j], df.loc[i, answer])
            for j in range(len(df.loc[i, ground_truth]))])
        for i in range(len(df))]
    
    # average out
    mean_score = np.mean(scores)
    return mean_score


def data_loader(path, cols, sep = "\t"):
    
    """
    Load data from a file into a pandas DataFrame.
    This used for cases in which we have multiple entries for text-selection

    Parameters:
    path (str): Path to the input file.
    cols (list): List of column names.
    sep (str): Separator used in the file (default is tab).

    Returns:
    pd.DataFrame: DataFrame containing the loaded data.
    """

    dict_df = defaultdict(list)
    # read line by line
    with open(path, "r", encoding="utf8") as infile:
        for line in infile:

            line = line.strip("\n")
            if line == "":
                continue

            fields = line.split(sep)

            for i, name in enumerate(cols[:-1]):
                if not name:
                    continue
                dict_df[name].append(fields[i])

            # for text-selection line we need to save the multiple entries as a list
            dict_df[cols[-1]].append(fields[len(cols)-1:])

    df = pd.DataFrame.from_dict(dict_df)
    return df

############################################################################################
##### Run Extractive QA
############################################################################################
if __name__ == "__main__":

    
    parser = argparse.ArgumentParser(
        description="Extractive QA Script"
    )

    parser.add_argument(
        "-t",
        "--tuples-file",
        #required=True,
        default="data/Part-3/msmarco-fira-21.qrels.qa-tuples.tsv",
        help="Path to the msmarco-fira-21.qrels.qa-tuples.tsv file",
        metavar="TUPLES_FILE_PATH",
        dest="tuples_file_path",
    )

    parser.add_argument(
        "-a",
        "--answers-file",
        #required=True,
        default="data/Part-3/msmarco-fira-21.qrels.qa-answers.tsv",
        help="Path to the msmarco-fira-21.qrels.qa-answers.tsv file",
        metavar="ANSWERS_FILE_PATH",
        dest="answers_file_path",
    )

    parser.add_argument(
        "-r",
        "--reranked-file",
        #required=True,
        default="data/results_part2/tk_msmarco_ranking_final.tsv",
        help="Path to the reranking results file from part 2",
        metavar="reranked_FILE_PATH",
        dest="reranked_file_path",
    )

    parser.add_argument(
        "--run_model", 
        type=bool, 
        default=True, 
        help="Set to True to run models."
    )

    parser.add_argument(
        "-o",
        default="data/results_part3",
        help="Path to output folder",
        metavar="PATH_TO_OUTPUT",
        dest="path_to_output",
    )



    args = parser.parse_args()

    ############################################################################################
    ##### Data Loading 
    ############################################################################################
    print("Data Loading...")
    # tuples
    tuples_path = args.tuples_file_path
    tuple_cols = ['queryid', 'documentid', 'relevance-grade', 'query-text', 'document-text', None, 'text-selection']
    #tuples_df = pd.read_csv(tuples_path, sep='\t', header=None, names=tuple_cols, index_col=False)
    tuples_df = data_loader(tuples_path, tuple_cols)
    # answers 
    # This ds needs to be loaded differently due to several answers being separated by tabs -> save them as a list
    answers_path = args.answers_file_path
    answer_cols = ['queryid', 'documentid', 'relevance-grade', None,'text-selection']
    answers_df = data_loader(answers_path, answer_cols)
    

    answers_df.queryid = answers_df.queryid.astype("int64")
    answers_df.documentid = answers_df.documentid.astype("int64")

    # results from part 2
    results_p2_path = args.reranked_file_path
    resP2_cols = ['queryid', 'doc_id', 'score', 'rank']
    resP2_df  = pd.read_csv(results_p2_path, sep='\t', names=resP2_cols, skiprows=1)





    print("loading data successful")

    ############################################################################################
    ##### Extract Top-1 neural re-ranking results 
    ############################################################################################

    print("exracting top-1 reranked result")
    top1_df = get_top_ranked_docs(resP2_df, answers_df)

    boolCheck1 = (top1_df['queryid'].isin(resP2_df['queryid'])).all()
    boolCheck2 = (top1_df["rank"] == 1).all()

    if(boolCheck1 and boolCheck2):
        print("extraction successful")
    else:
           raise ValueError("There is an issue with the extrction please refer to chapter: Extract Top-1 neural re-ranking results")



    
    ############################################################################################
    ##### Download Model
    ############################################################################################
    
    # Create output folder if it does not exist
    os.makedirs(args.path_to_output, exist_ok=True)

    # set output file paths
    output_file_path_fira = os.path.join(args.path_to_output, "fira_tuples_qa_results.tsv")
    output_file_path_top1 = os.path.join(args.path_to_output, "msmarco_top1_qa_results.tsv")

    if(args.run_model):
        print("Downloading Model: roberta-base-squad2")
        # Load model directly
        tokenizer = AutoTokenizer.from_pretrained("deepset/roberta-base-squad2")
        model = AutoModelForQuestionAnswering.from_pretrained("deepset/roberta-base-squad2")

        # run model for FIRA tuples
        print("Run Model for Fira Tuples dataset")
        tuples_answers_df = run_model(tuples_df, model, tokenizer)
        print("Save Results")
        # Save the results for the Fira tuples_df
        
        with open(output_file_path_fira, "w", encoding="utf8") as outfile:
            outfile.write(tuples_answers_df.to_csv(sep="\t", index=False))
        print(f"msmarco-fira-21.qrels.qa-tuples results saved to {output_file_path_fira}")

        # run model on top1 reranked
        print("Run Model for Top-1 Reranked data")
        msmarco_answers_df = run_model(top1_df, model, tokenizer)
        print("Save Results")
        # Save the results for the Fira tuples_df
        
        with open(output_file_path_top1, "w", encoding="utf8") as outfile:
            outfile.write(msmarco_answers_df.to_csv(sep="\t", index=False))

        print(f"msmarco reranked results saved to {output_file_path_top1}")
    else:
        
        print("loading saved results")
        tuples_answers_df = pd.read_csv(output_file_path_fira, sep="\t")
        msmarco_answers_df = pd.read_csv(output_file_path_top1, sep="\t")

        # we need to convert the sel column to a list of strings, otherwise the metric functions don't work
        tuples_answers_df["text-selection"] = tuples_answers_df["text-selection"].apply(ast.literal_eval)
        msmarco_answers_df["text-selection"] = msmarco_answers_df["text-selection"].apply(ast.literal_eval)

    ############################################################################################
    ##### Evaluate Metrics
    ############################################################################################

    print("evaluate metrics using core_metrics.py")
    fira_f1 = calc_metrics(tuples_answers_df, compute_f1, "text-selection", "answer")
    fira_exact = calc_metrics(tuples_answers_df, compute_exact, "text-selection", "answer")

    print("evaluating...")
    msmarco_f1 = calc_metrics(msmarco_answers_df, compute_f1, "text-selection", "answer")
    msmarco_exact = calc_metrics(msmarco_answers_df, compute_exact, "text-selection", "answer")

    print("Evaliaton:")
    print("Fira-21-tuples")
    print(f"#Rows: {len(tuples_answers_df)}, F1 (Overlap): {fira_f1}, Exact Match: {fira_exact}")
    print("Top-1-Reranked")
    print(f"#Rows: {len(msmarco_answers_df)}, F1 (Overlap): {msmarco_f1}, Exact Match: {msmarco_exact}")    