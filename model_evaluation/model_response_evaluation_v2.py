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

def calculate_average(scores):
    return {metric: sum(values) / len(values) for metric, values in scores.items()}

def evaluate_responses(df, model_col):
    results = {'overall': {}, 'by_category': {}}
    
    # Metrics structure
    metrics = {
        'ROUGE-1 (precision)': [], 'ROUGE-2 (precision)': [], 'ROUGE-L (precision)': [],
        'ROUGE-1 (recall)': [], 'ROUGE-2 (recall)': [], 'ROUGE-L (recall)': [],
        'ROUGE-1 (fscore)': [], 'ROUGE-2 (fscore)': [], 'ROUGE-L (fscore)': [],
        'BLEU': [], 'Cosine Similarity': []
    }
    
    # Iterate over rows
    for index, row in df.iterrows():
        question = row['user_query']
        ground_truth = row['ground_truth_response']
        response = row[model_col]
        category = row['question_category']
        
        # ROUGE and Cosine Similarity calculations
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
        
        # Append scores to metrics
        for key, value in zip(
            ['ROUGE-1 (precision)', 'ROUGE-2 (precision)', 'ROUGE-L (precision)',
             'ROUGE-1 (recall)', 'ROUGE-2 (recall)', 'ROUGE-L (recall)',
             'ROUGE-1 (fscore)', 'ROUGE-2 (fscore)', 'ROUGE-L (fscore)',
             'BLEU', 'Cosine Similarity'],
            [rouge1_precision, rouge2_precision, rougel_precision,
             rouge1_recall, rouge2_recall, rougel_recall,
             rouge1_fscore, rouge2_fscore, rougel_fscore,
             bleu_score, cosine_sim]
        ):
            metrics[key].append(value)
        
        # Category-specific metrics
        if category not in results['by_category']:
            results['by_category'][category] = {k: [] for k in metrics.keys()}
        
        for key, value in zip(metrics.keys(),
                              [rouge1_precision, rouge2_precision, rougel_precision,
                               rouge1_recall, rouge2_recall, rougel_recall,
                               rouge1_fscore, rouge2_fscore, rougel_fscore,
                               bleu_score, cosine_sim]):
            results['by_category'][category][key].append(value)
    
    # Calculate averages for overall and by category
    results['overall'] = calculate_average(metrics)
    results['by_category'] = {category: calculate_average(scores) for category, scores in results['by_category'].items()}
    
    return results

# Evaluate models
all_results = {}
models = ['llama_3_2_vision_response', 'GPT_4_turbo_response', 'claude_3_5_sonnet_response', 'paligemma_response']

for model_col in models:
    print(f"Evaluating {model_col}")
    model_results = evaluate_responses(df, model_col)
    all_results[model_col] = model_results

# Save results to JSON
json_results = json.dumps(all_results, indent=4)
with open("evaluation_results_by_category.json", "w") as f:
    f.write(json_results)
