import pandas as pd
import json
import anthropic

def create_evaluation_prompt(question, ground_truth, model_response):
    prompt = f"""You are an expert evaluator. Compare the model's response with the ground truth response for the given question.
    
Question: {question}
Ground Truth Response: {ground_truth}
Model Response: {model_response}

Evaluate if the model response is correct or incorrect based on the ground truth. The response should be considered correct if it conveys the same meaning or information as the ground truth, even if the wording is different. Consider the model response as incorrect if the information is not complete with respect to ground truth.

Respond with ONLY one word - either 'CORRECT' or 'INCORRECT'."""
    return prompt

def evaluate_responses(df):
    client = anthropic.Client(api_key='<API KEY HERE>')
    
    
    results = {
        "llama_3_2_vision_response": {"evaluations": []},
        "GPT_4_turbo_response": {"evaluations": []},
        "claude_3_5_sonnet_response": {"evaluations": []},
        "paligemma_response": {"evaluations": []},
        "summary": {}
    }
    
    models = ['llama_3_2_vision_response', 'GPT_4_turbo_response', 'claude_3_5_sonnet_response', 'paligemma_response']
    

    model_stats = {model: {'correct': 0, 'incorrect': 0} for model in models}
    num_rows = df.shape[0]
    print("Number of rows:", num_rows)


    for index, row in df.iterrows():
        question = row['user_query']
        ground_truth = row['ground_truth_response']
        
        for model in models:
            model_response = row[model]
            
            prompt = create_evaluation_prompt(question, ground_truth, model_response)
            

            response = client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=10,
                temperature=0,
                messages=[{"role": "user", "content": prompt}]
            )
            print(response)
            evaluation = response.content[0].text.strip()
            

            question_eval = {
                "question": question,
                "evaluation": evaluation
            }
            results[model]["evaluations"].append(question_eval)
            
            
            if evaluation == "CORRECT":
                model_stats[model]['correct'] += 1
            else:
                model_stats[model]['incorrect'] += 1
    

    total_questions = len(df)
    summary = {}
    
    for model in models:
        correct = model_stats[model]['correct']
        incorrect = model_stats[model]['incorrect']
        accuracy = correct / total_questions
        
        summary[model] = {
            "correct_responses": correct,
            "incorrect_responses": incorrect,
            "accuracy": round(accuracy, 2)
        }
    
    results["summary"] = summary
    
    return results


def main():
    
    df = pd.read_csv('Question_Answer_Evaluation_LLM.csv')
    df.columns = df.columns.str.strip()
    
  
    results = evaluate_responses(df)
    
    
    with open('evaluation_results_llm.json', 'w') as f:
        json.dump(results, f, indent=4)
 
    print("\nEvaluation Summary:")
    for model, metrics in results["summary"].items():
        print(f"\n{model}:")
        print(f"Correct Responses: {metrics['correct_responses']}")
        print(f"Incorrect Responses: {metrics['incorrect_responses']}")
        print(f"Accuracy: {metrics['accuracy']:.2%}")

if __name__ == "__main__":
    main()
