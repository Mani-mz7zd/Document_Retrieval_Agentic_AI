import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from nltk.translate.bleu_score import sentence_bleu
from sentence_transformers import SentenceTransformer
import evaluate
from rouge_score import rouge_scorer
import json

df = pd.read_csv("Question_Answer_Evaluation_LLM.csv")
df.columns = df.columns.str.strip()

print(df.columns)

rouge_hf = evaluate.load("rouge")
rouge_scorer_model = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')


def evaluate_responses(df, model_col):
    results = {'scores': [], 'average_scores': {}}
    model_scores = {
        'ROUGE-1 (precision)': [], 'ROUGE-2 (precision)': [], 'ROUGE-L (precision)': [],
        'ROUGE-1 (recall)': [], 'ROUGE-2 (recall)': [], 'ROUGE-L (recall)': [],
        'ROUGE-1 (fscore)': [], 'ROUGE-2 (fscore)': [], 'ROUGE-L (fscore)': [],
        'BLEU': [], 'Cosine Similarity': []
    }
    
    for index, row in df.iterrows():
        question = row['user_query']
        ground_truth = row['ground_truth_response']
        response = row[model_col]

        rouge_hf_scores = rouge_hf.compute(predictions=[response], references=[ground_truth])
        # rouge1_evaluate = float(rouge_hf_scores['rouge1'])
        # rouge2_evaluate = float(rouge_hf_scores['rouge2'])
        # rougel_evaluate = float(rouge_hf_scores['rougeL'])


        rouge_scorer_scores = rouge_scorer_model.score(ground_truth, response)
        rouge1_fscore = float(rouge_scorer_scores['rouge1'].fmeasure)
        rouge2_fscore = float(rouge_scorer_scores['rouge2'].fmeasure)
        rougel_fscore = float(rouge_scorer_scores['rougeL'].fmeasure)

        rouge1_precision = float(rouge_scorer_scores['rouge1'].precision)
        rouge2_precision  = float(rouge_scorer_scores['rouge2'].precision)
        rougel_precision  = float(rouge_scorer_scores['rougeL'].precision)

        rouge1_recall = float(rouge_scorer_scores['rouge1'].recall)
        rouge2_recall = float(rouge_scorer_scores['rouge2'].recall)
        rougel_recall = float(rouge_scorer_scores['rougeL'].recall)

        
        ground_truth_tokens = ground_truth.split()
        response_tokens = response.split()
        bleu_score = float(sentence_bleu([ground_truth_tokens], response_tokens))

        
        ground_truth_embedding = embedding_model.encode([ground_truth])[0].reshape(1, -1)
        response_embedding = embedding_model.encode([response])[0].reshape(1, -1)
        cosine_sim = float(cosine_similarity(ground_truth_embedding, response_embedding)[0][0])
        
        model_scores['ROUGE-1 (precision)'].append(rouge1_precision)
        model_scores['ROUGE-2 (precision)'].append(rouge2_precision)
        model_scores['ROUGE-L (precision)'].append(rougel_precision)

        model_scores['ROUGE-1 (recall)'].append(rouge1_recall)
        model_scores['ROUGE-2 (recall)'].append(rouge2_recall)
        model_scores['ROUGE-L (recall)'].append(rougel_recall)

        model_scores['ROUGE-1 (fscore)'].append(rouge1_fscore)
        model_scores['ROUGE-2 (fscore)'].append(rouge2_fscore)
        model_scores['ROUGE-L (fscore)'].append(rougel_fscore)
        
        model_scores['BLEU'].append(bleu_score)
        model_scores['Cosine Similarity'].append(cosine_sim)

    
        results['scores'].append({
            'question': question,
            'model': model_col,
            'ROUGE-1 (precision)': rougel_precision,
            'ROUGE-2 (precision)': rouge2_precision,
            'ROUGE-L (precision)': rougel_precision,
            'ROUGE-1 (recall)': rouge1_recall,
            'ROUGE-2 (recall)': rouge2_recall,
            'ROUGE-L (recall)': rougel_recall,
            'ROUGE-1 (fscore)': rouge1_fscore,
            'ROUGE-2 (fscore)': rouge2_fscore,
            'ROUGE-L (fscore)': rougel_fscore,
            'BLEU': bleu_score,
            'Cosine Similarity': cosine_sim
        })

    results['average_scores'] = {
        'ROUGE-1 (precision)': sum(model_scores['ROUGE-1 (precision)']) / len(model_scores['ROUGE-1 (precision)']),
        'ROUGE-1 (recall)': sum(model_scores['ROUGE-1 (recall)']) / len(model_scores['ROUGE-1 (recall)']),
        'ROUGE-1 (fscore)': sum(model_scores['ROUGE-1 (fscore)']) / len(model_scores['ROUGE-1 (fscore)']),

        'ROUGE-2 (precision)': sum(model_scores['ROUGE-2 (precision)']) / len(model_scores['ROUGE-2 (precision)']),
        'ROUGE-2 (recall)': sum(model_scores['ROUGE-2 (recall)']) / len(model_scores['ROUGE-2 (recall)']),
        'ROUGE-2 (fscore)': sum(model_scores['ROUGE-2 (fscore)']) / len(model_scores['ROUGE-2 (fscore)']),


        'ROUGE-L (precision)': sum(model_scores['ROUGE-L (precision)']) / len(model_scores['ROUGE-L (precision)']),        
        'ROUGE-L (recall)': sum(model_scores['ROUGE-L (recall)']) / len(model_scores['ROUGE-L (recall)']),
        'ROUGE-L (fscore)': sum(model_scores['ROUGE-L (fscore)']) / len(model_scores['ROUGE-L (fscore)']),

        'BLEU': sum(model_scores['BLEU']) / len(model_scores['BLEU']),
        'Cosine Similarity': sum(model_scores['Cosine Similarity']) / len(model_scores['Cosine Similarity'])
    }
    
    return results


all_results = {}
average_results = {}
models = ['llama_3_2_vision_response', 'GPT_4_turbo_response', 'claude_3_5_sonnet_response', 'paligemma_response']

for model_col in models:
    print(f"Evaluating {model_col}")
    model_results = evaluate_responses(df, model_col)
    all_results[model_col] = model_results
    average_results[model_col] = model_results['average_scores']


json_results = json.dumps(all_results, indent=4)
with open("evaluation_results_v4.json", "w") as f:
    f.write(json_results)


average_json_results = json.dumps(average_results, indent=4)
with open("average_evaluation_results_v4.json", "w") as f:
    f.write(average_json_results)