from typing import Any
from lm_eval.base import RequestFactory, MultipleChoiceTask, PerplexityTask, Task
from lm_eval.utils import get_s3_csv_dataset
import re
import random

########### Helper Functions ############
def get_correct_index(correst_choice): # Some csvs have the correct answer as a letter, others as an index
    if isinstance(correst_choice, int):
        return correst_choice
    elif isinstance(correst_choice, str):
        return ord(correst_choice.replace(" ", "").lower()) - ord('a') # Some have a trailing space

def ensure_leading_space(text):
    if text == '':
        return ' asdfkjsdovinsavklwneviowevhwoirnwlkegnwroigeovinrlbkefnbdoifb@#$@#!$(!)#R!F@UEF(V#R)'
    elif text[0] != ' ':
        return ' ' + text
    else:
        return text
    
def ensure_prompt_doesnt_end_with_space(text):
    if text.endswith(' '):
        return text[:-1]
    else:
        return text
    
def remove_leading_answer_choices(answer_choice):
    answer_choice = answer_choice.lstrip() # Remove leading spaces
    answer_choice = re.sub(r'^[a-zA-Z]\.', '', answer_choice) # Remove leading answer choices (e.g. 'A. <answer>' -> '<answer>')
    answer_choice = re.sub(r'^[a-zA-Z]\)', '', answer_choice) # Remove leading answer choices (e.g. 'A) <answer>' -> '<answer>')
    answer_choice = re.sub(r'^\([a-zA-Z]\)', '', answer_choice) # Remove leading answer choices (e.g. '(A) <answer>' -> '<answer>')
    return answer_choice

def has_underscore(text):
    pattern = r"_{3,}"
    return bool(re.search(pattern, text))

def index_to_capital_letter(index):
    if 0 <= index < 26:
        return chr(ord('A') + index)
    else:
        raise ValueError("Index out of range for capital letters (0-25)")


############## Base Medical Task Class that all other tasks inherit from ##############
class MedicalTask(Task):
    VERSION = 1
    DATASET_PATH = None
    DATASET_NAME = None
    
    def has_training_docs(self):
        return False
    def has_validation_docs(self):
        return False
    def has_test_docs(self):
        return True
    
    def test_docs(self):
        if self.has_test_docs():
            return map(self._process_doc, self.dataset)
        
    def doc_to_text(self, doc):
        return doc['query']

    
############### Medical Task Classes ###############
class MedicalMCPerplexityFormatter(MultipleChoiceTask, MedicalTask):
    def construct_requests(self, doc, ctx):
        # Helper function to determine if context has underscore
        def has_underscore(text):
            return '___' in text

        # Replace ___ with $$$ to be handled properly
        if has_underscore(ctx):
            ctx = re.sub(r'_+.*', '$$$', ctx)

        # Handle $$$
        if ctx.startswith("$$$"):
            lls = [rf.loglikelihood(ensure_prompt_doesnt_end_with_space(choice), ensure_leading_space(ctx.replace('$$$', '')))[0] for choice in doc["choices"]]
        elif "$$$" in ctx:
            split_on_dollar_sign = ctx.split("$$$")
            lls = [rf.loglikelihood(ensure_prompt_doesnt_end_with_space(split_on_dollar_sign[0]), ensure_leading_space(choice + split_on_dollar_sign[1]))[0] for choice in doc["choices"]]
        
        # Handle $$
        elif ctx.startswith("$$"):
            lls = [rf.loglikelihood(ensure_prompt_doesnt_end_with_space(choice), ensure_leading_space(ctx.replace('$$', '')))[0] for choice in doc["choices"]]
        elif "$$" in ctx:
            split_on_dollar_sign = ctx.split("$$")
            lls = [rf.loglikelihood(ensure_prompt_doesnt_end_with_space(split_on_dollar_sign[0]), ensure_leading_space(choice + split_on_dollar_sign[1]))[0] for choice in doc["choices"]]
        
        else:
            lls = [rf.loglikelihood(ensure_prompt_doesnt_end_with_space(ctx), ensure_leading_space(choice))[0] for choice in doc["choices"]]

        return lls
                
class CutoffMedicalMCPerplexityFormatter(MultipleChoiceTask, MedicalTask):
    def construct_requests(self, doc, ctx):
        # Helper function to determine if context has underscore
        def has_underscore(text):
            return '___' in text

        # Replace ___ with $$$ to be handled properly
        if has_underscore(ctx):
            ctx = re.sub(r'_+.*', '$$$', ctx)

        # Handle $$$
        if ctx.startswith("$$$"):
            lls = [rf.loglikelihood(ensure_prompt_doesnt_end_with_space(choice), ensure_leading_space(ctx.replace('$$$', '')))[0] for choice in doc["choices"]]
        elif "$$$" in ctx:
            split_on_dollar_sign = ctx.split("$$$")
            # Chop out the part after the $$$
            lls = [rf.loglikelihood(ensure_prompt_doesnt_end_with_space(split_on_dollar_sign[0]), ensure_leading_space(choice))[0] for choice in doc["choices"]]
        
        # Handle $$
        elif ctx.startswith("$$"):
            lls = [rf.loglikelihood(ensure_prompt_doesnt_end_with_space(choice), ensure_leading_space(ctx.replace('$$', '')))[0] for choice in doc["choices"]]
        elif "$$" in ctx:
            split_on_dollar_sign = ctx.split("$$")
            # Chop out the part after the $$
            lls = [rf.loglikelihood(ensure_prompt_doesnt_end_with_space(split_on_dollar_sign[0]), ensure_leading_space(choice))[0] for choice in doc["choices"]]
        
        else:
            lls = [rf.loglikelihood(ensure_prompt_doesnt_end_with_space(ctx), ensure_leading_space(choice))[0] for choice in doc["choices"]]

        return lls
    
    
class MedicalMCPerplexityTask(MedicalMCPerplexityFormatter):
    def _process_doc(self, doc):
        answer_choices = []
        for letter in ['a', 'b', 'c' ,'d']:
            try:
                answer_choices.append(doc[letter])
            except:
                answer_choices.append('asdfkjsdovinsavklwneviowevhwoirnwlkegnwroigeovinrlbkefnbdoifb@#$@#!$(!)#R!F@UEF(V#R)')
            if answer_choices[-1] is None:
                answer_choices[-1] = 'asdfkjsdovinsavklwneviowevhwoirnwlkegnwroigeovinrlbkefnbdoifb@#$@#!$(!)#R!F@UEF(V#R)'
        answer_choices = [remove_leading_answer_choices(answer_choice).lstrip() for answer_choice in answer_choices]
        
        out_doc = {
            "query": doc["question"], # The query prompt
            "choices": answer_choices, # The list of choices
            "gold": 0 if "correct" not in doc else get_correct_index(doc["correct"]), # The integer used to index into the correct element of "choices"
        }
        return out_doc

class MedicalMCPerplexityTaskV1(MedicalMCPerplexityTask):
    ADDITIONAL_CONTEXT = "The following statement is related to medical and healthcare topics."

    def _process_doc(self, doc):
        out_doc = super()._process_doc(doc)
        out_doc["query"] = self.ADDITIONAL_CONTEXT + " " + out_doc["query"]
        return out_doc

class MedicalMCPerplexityTaskV2(MedicalMCPerplexityTask):
    ADDITIONAL_CONTEXT = "The doctor said:"

    def _process_doc(self, doc):
        out_doc = super()._process_doc(doc)
        out_doc["query"] = self.ADDITIONAL_CONTEXT + " " + out_doc["query"]
        return out_doc
    
class CutoffMedicalMCPerplexityTask(CutoffMedicalMCPerplexityFormatter):
    def _process_doc(self, doc):
        answer_choices = []
        for letter in ['a', 'b', 'c' ,'d']:
            try:
                answer_choices.append(doc[letter])
            except:
                answer_choices.append('asdfkjsdovinsavklwneviowevhwoirnwlkegnwroigeovinrlbkefnbdoifb@#$@#!$(!)#R!F@UEF(V#R)')
            if answer_choices[-1] is None:
                answer_choices[-1] = 'asdfkjsdovinsavklwneviowevhwoirnwlkegnwroigeovinrlbkefnbdoifb@#$@#!$(!)#R!F@UEF(V#R)'
        answer_choices = [remove_leading_answer_choices(answer_choice).lstrip() for answer_choice in answer_choices]
        
        out_doc = {
            "query": doc["question"], # The query prompt
            "choices": answer_choices, # The list of choices
            "gold": 0 if "correct" not in doc else get_correct_index(doc["correct"]), # The integer used to index into the correct element of "choices"
        }
        return out_doc
    
class CutoffMedicalPerplexityTask(CutoffMedicalMCPerplexityFormatter):
    def _process_doc(self, doc):
        answer_choices = []
        for letter in ['a']:
            try:
                answer_choices.append(doc[letter])
            except:
                answer_choices.append('asdfkjsdovinsavklwneviowevhwoirnwlkegnwroigeovinrlbkefnbdoifb@#$@#!$(!)#R!F@UEF(V#R)')
            if answer_choices[-1] is None:
                answer_choices[-1] = 'asdfkjsdovinsavklwneviowevhwoirnwlkegnwroigeovinrlbkefnbdoifb@#$@#!$(!)#R!F@UEF(V#R)'
        answer_choices = [remove_leading_answer_choices(answer_choice).lstrip() for answer_choice in answer_choices]
        self.max_len_answer_choice = max([len(answer_choice) for answer_choice in answer_choices])
        
        out_doc = {
            "query": doc["question"], # The query prompt
            "choices": answer_choices, # The list of choices
            "gold": 0 if "correct" not in doc else get_correct_index(doc["correct"]), # The integer used to index into the correct element of "choices"
        }
        return out_doc
    
class MedicalCertificationTask(MedicalMCPerplexityFormatter):
    def _process_doc(self, doc):
        
        answer_choices = []
        for key in sorted(doc.keys()):
            if ('answer' in key.strip().lower() and 'correct' not in key.strip().lower()) and doc[key] is not None:
                answer_choices.append(remove_leading_answer_choices(doc[key]))
        answer_choices = [answer_choice.lstrip() for answer_choice in answer_choices] # Remove leading spaces
        
        for question_stem_key in ["Question Stem", "question_stem"]:
            if question_stem_key in doc:
                query = doc[question_stem_key]
                break
        
        for correct_answer_key in ["Correct Answer", "correct_answer"]:
            if correct_answer_key in doc:
                correct_answer = doc[correct_answer_key]
                break
            
        new_query = ensure_prompt_doesnt_end_with_space(query)

        out_doc = {
            "query": new_query, # The query prompt.
            "choices": answer_choices, # The list of choices.
            "gold": get_correct_index(correct_answer), # The integer used to index into the correct element of `"choices"`.
        }
        return out_doc

class MedicalMCTask(MultipleChoiceTask, MedicalTask):
    def __init__(self):
        self.random_seed_counter = 0
        
    def _process_doc(self, doc):
        def format_example(question, answer_choices, starting_gold):
            """
            <prompt>
            A. <choice1>
            B. <choice2>
            C. <choice3>
            D. <choice4>
            Answer:
            """
            
            question = re.sub(r'\$\$+', '__________', question)
            
            num_choices = len(answer_choices)
            # Permute the answer choices
            random.seed(self.random_seed_counter)
            self.random_seed_counter += 1
            
            answer_choice_indices = random.sample(range(num_choices), num_choices)
            gold = answer_choice_indices.index(starting_gold)
            
            keys = [index_to_capital_letter(index) for index in range(num_choices)]
            choices = "".join(
                [f"{key}. {answer_choices[choice]}\n" for key, choice in zip(keys, answer_choice_indices)]
            )
            prompt = f"The following are fill in the blank multiple choice questions (with answers) related to healthcare and medicine.\n\n{question}\n{choices}Answer:"
            return prompt, keys, gold
        
        question = doc["question"].strip()
        answer_choices = [doc["a"], doc["b"], doc["c"], doc["d"]]
        answer_choices = [remove_leading_answer_choices(answer_choice).lstrip() for answer_choice in answer_choices]
        starting_gold = 0 if "correct" not in doc else get_correct_index(doc["correct"])
        
        query, keys, gold = format_example(question, answer_choices, starting_gold)
        
        out_doc = {
            "query": query, # The query prompt
            "choices": keys, # The list of choices
            "gold": gold, # The integer used to index into the correct element of "choices"
        }
        return out_doc
    
class MedicalMCPerplexityTask_8_16_Forward(MedicalMCPerplexityFormatter):
    def _process_doc(self, doc):
        answer_choices = []
        for letter in ['a', 'b', 'c' ,'d']:
            try:
                answer_choices.append(doc[letter])
            except:
                answer_choices.append('asdfkjsdovinsavklwneviowevhwoirnwlkegnwroigeovinrlbkefnbdoifb@#$@#!$(!)#R!F@UEF(V#R)')
            if answer_choices[-1] is None:
                answer_choices[-1] = 'asdfkjsdovinsavklwneviowevhwoirnwlkegnwroigeovinrlbkefnbdoifb@#$@#!$(!)#R!F@UEF(V#R)'
        answer_choices = [remove_leading_answer_choices(answer_choice).lstrip() for answer_choice in answer_choices]
        
        out_doc = {
            "query": doc["stem"], # The query prompt
            "choices": answer_choices, # The list of choices
            "gold": 0 if "correct" not in doc else get_correct_index(doc["correct"]), # The integer used to index into the correct element of "choices"
        }
        return out_doc

class MedicalTFPerplexityTask(MedicalMCPerplexityFormatter):
    def _process_doc(self, doc):
        answer_choices = []
        for letter in ['a', 'b']:
            try:
                answer_choices.append(doc[letter])
            except:
                answer_choices.append('asdfkjsdovinsavklwneviowevhwoirnwlkegnwroigeovinrlbkefnbdoifb@#$@#!$(!)#R!F@UEF(V#R)')
            if answer_choices[-1] is None:
                answer_choices[-1] = 'asdfkjsdovinsavklwneviowevhwoirnwlkegnwroigeovinrlbkefnbdoifb@#$@#!$(!)#R!F@UEF(V#R)'
        answer_choices = [remove_leading_answer_choices(answer_choice).lstrip() for answer_choice in answer_choices]
        
        out_doc = {
            "query": doc["question"], # The query prompt
            "choices": answer_choices, # The list of choices
            "gold": 0 if "correct" not in doc else get_correct_index(doc["correct"]), # The integer used to index into the correct element of "choices"
        }
        return out_doc
    
# class MedicalRawPerplexityTaskTxt(PerplexityTask, MedicalTask):
#     def test_docs(self):
#         with smart_open.open(self.dataset_path, 'rt', encoding='utf-8') as f:
#             for line in f:
#                 yield line

# falcon_tokenizer = AutoTokenizer.from_pretrained("tiiuae/falcon-40b")
# class MedicalContextPerplexityTask(MedicalRawPerplexityTaskTxt):
#     def construct_requests(self, doc, ctx):
#         assert not ctx
#         req = rf.loglikelihood_rolling(self.doc_to_target(doc))
#         return req
#         tokens = doc
        
#         lls = [rf.loglikelihood(ensure_prompt_doesnt_end_with_space(ctx), ensure_leading_space(choice))[0] for choice in doc["choices"]]

#         return lls
    
# falcon_tokenizer = AutoTokenizer.from_pretrained("tiiuae/falcon-40b")
# class SlidingWndowTask(Task):
#     VERSION = 1
#     DATASET_PATH = None
#     DATASET_NAME = None
    
#     def has_training_docs(self):
#         return False
#     def has_validation_docs(self):
#         return False
#     def has_test_docs(self):
#         return True
    
#     def data_iter(self, context_length=2048, min_length=128):
#         for doc in self.dataset:
#             tokens = self.tokenizer.encode(doc["content"])
#             for i in range(0, len(tokens), context_length):
#                 context = tokens[i:i+context_length]
#                 if len(context) < min_length:
#                     continue
#                 yield context
    
#     def test_docs(self):
#         if self.has_test_docs():
#             return map(self._preprocess_doc, self.data_iter())
    
# class ContextPerplexityFormatter(SlidingWindowTask):
#     def construct_requests(self, doc, ctx):
#         tokens = doc
        
#         lls = [rf.loglikelihood(ensure_prompt_doesnt_end_with_space(ctx), ensure_leading_space(choice))[0] for choice in doc["choices"]]

#         return lls
    
# class ContextPerplexityTask(ContextPerplexityFormatter):
#     context_lengths = [512, 1024, 2048]
#     text_length = 512
#     def _preprocess_doc(self, doc):
#         tokens = doc
#         text = doc[:-self.text_length]
#         contexts = []
        
#         # lls = [rf.loglikelihood(ensure_prompt_doesnt_end_with_space(ctx), ensure_leading_space(choice))[0] for choice in doc["choices"]]

#         return {
#             "text": text,
#             "contexts": contexts,
#         }

    
### Gold set tasks
class esrd_combined(MedicalMCPerplexityTask):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://lm-eval/gold_set_9_12/esrd_combined.csv")
        
class knowledge_evals_combined(MedicalMCPerplexityTask):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://lm-eval/gold_set_9_12/knowledge_evals_combined.csv")
        
class knowledge_evals_combined_v1(MedicalMCPerplexityTaskV1):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://lm-eval/gold_set_9_12/knowledge_evals_combined.csv")
        
class knowledge_evals_combined_v2(MedicalMCPerplexityTaskV2):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://lm-eval/gold_set_9_12/knowledge_evals_combined.csv")
        
class SL_conditions_completion(MedicalMCPerplexityTask):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://lm-eval/gold_set_9_12/SL_conditions_completion.csv")
        
class SL_drug_completion(MedicalMCPerplexityTask):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://lm-eval/gold_set_9_12/SL_drug_completion.csv")
        
class SL_symptom_completion(MedicalMCPerplexityTask):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://lm-eval/gold_set_9_12/SL_symptom_completion.csv")

class SL_sentence_selection(MedicalMCPerplexityTask):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://lm-eval/gold_set_9_12/SL_sentence_selection_eval.csv")
        
class SL_handpicked_evals(MedicalMCPerplexityTask):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://lm-eval/gold_set_9_12/SL_handpicked_evals.csv")
        
class tkr_combined(MedicalMCPerplexityTask):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://lm-eval/gold_set_9_12/tkr_combined.csv")
        
class high_quality_v1(MedicalMCPerplexityTask):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://lm-eval/9_11_batch/high_quality.csv")
        
class high_quality_v2(MedicalMCTask):
    def __init__(self):
        super().__init__()
        self.dataset = get_s3_csv_dataset("s3://lm-eval/9_11_batch/high_quality.csv")
        
class high_quality_v3(CutoffMedicalMCPerplexityTask):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://lm-eval/9_11_batch/high_quality.csv")
        
class hqm_v2(MedicalMCPerplexityTask):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://lm-eval/gold_set_9_12/hqm_v2.csv")
        
class brochure_ctx0_len512(CutoffMedicalPerplexityTask):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://lm-eval/9_15_batch/brochure_ctx0_len512.csv")

class brochure_ctx512_len512(CutoffMedicalPerplexityTask):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://lm-eval/9_15_batch/brochure_ctx512_len512.csv")
        
class brochure_ctx1024_len512(CutoffMedicalPerplexityTask):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://lm-eval/9_15_batch/brochure_ctx1024_len512.csv")

class brochure_ctx1536_len512(CutoffMedicalPerplexityTask):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://lm-eval/9_15_batch/brochure_ctx1536_len512.csv")

class clinical_guidelines_ctx0_len512(CutoffMedicalPerplexityTask):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://lm-eval/9_15_batch/clinical_guidelines_ctx0_len512.csv")

class clinical_guidelines_ctx512_len512(CutoffMedicalPerplexityTask):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://lm-eval/9_15_batch/clinical_guidelines_ctx512_len512.csv")
        
class clinical_guidelines_ctx1024_len512(CutoffMedicalPerplexityTask):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://lm-eval/9_15_batch/clinical_guidelines_ctx1024_len512.csv")

class clinical_guidelines_ctx1536_len512(CutoffMedicalPerplexityTask):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://lm-eval/9_15_batch/clinical_guidelines_ctx1536_len512.csv")
        
class foia_ctx0_len512(CutoffMedicalPerplexityTask):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://lm-eval/9_15_batch/foia_ctx0_len512.csv")

class foia_ctx512_len512(CutoffMedicalPerplexityTask):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://lm-eval/9_15_batch/foia_ctx512_len512.csv")
        
class foia_ctx1024_len512(CutoffMedicalPerplexityTask):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://lm-eval/9_15_batch/foia_ctx1024_len512.csv")

class foia_ctx1536_len512(CutoffMedicalPerplexityTask):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://lm-eval/9_15_batch/foia_ctx1536_len512.csv")
        
class tjc_ctx0_len512(CutoffMedicalPerplexityTask):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://lm-eval/9_15_batch/tjc_ctx0_len512.csv")

class tjc_ctx512_len512(CutoffMedicalPerplexityTask):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://lm-eval/9_15_batch/tjc_ctx512_len512.csv")
        
class tjc_ctx1024_len512(CutoffMedicalPerplexityTask):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://lm-eval/9_15_batch/tjc_ctx1024_len512.csv")

class tjc_ctx1536_len512(CutoffMedicalPerplexityTask):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://lm-eval/9_15_batch/tjc_ctx1536_len512.csv")
        
class usptf_ctx0_len512(CutoffMedicalPerplexityTask):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://lm-eval/9_15_batch/usptf_ctx0_len512.csv")

class usptf_ctx512_len512(CutoffMedicalPerplexityTask):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://lm-eval/9_15_batch/usptf_ctx512_len512.csv")
        
class usptf_ctx1024_len512(CutoffMedicalPerplexityTask):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://lm-eval/9_15_batch/usptf_ctx1024_len512.csv")

class usptf_ctx1536_len512(CutoffMedicalPerplexityTask):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://lm-eval/9_15_batch/usptf_ctx1536_len512.csv")
        
### LM perplexity eval
class high_quality_v3(CutoffMedicalMCPerplexityTask):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://lm-eval/9_11_batch/high_quality.csv")

### 8_23 tasks

class cancer_tf(MedicalTFPerplexityTask):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://lm-eval/medical_knowledge_evals/8_23/true_false/cancer_tf.csv")

class cardiology_tf(MedicalTFPerplexityTask):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://lm-eval/medical_knowledge_evals/8_23/true_false/cardiology_tf.csv")

class chf_tf(MedicalTFPerplexityTask):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://lm-eval/medical_knowledge_evals/8_23/true_false/chf_tf.csv")

class ckd_tf(MedicalTFPerplexityTask):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://lm-eval/medical_knowledge_evals/8_23/true_false/ckd_tf.csv")

class cms_tf(MedicalTFPerplexityTask):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://lm-eval/medical_knowledge_evals/8_23/true_false/cms_tf.csv")

class copd_tf(MedicalTFPerplexityTask):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://lm-eval/medical_knowledge_evals/8_23/true_false/copd_tf.csv")

class depression_tf(MedicalTFPerplexityTask):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://lm-eval/medical_knowledge_evals/8_23/true_false/depression_tf.csv")

class diabetes_tf(MedicalTFPerplexityTask):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://lm-eval/medical_knowledge_evals/8_23/true_false/diabetes_tf.csv")

class diet_tf(MedicalTFPerplexityTask):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://lm-eval/medical_knowledge_evals/8_23/true_false/diet_tf.csv")

class drug_tiers_tf(MedicalTFPerplexityTask):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://lm-eval/medical_knowledge_evals/8_23/true_false/drug_tiers_tf.csv")

class emergency_med_tf(MedicalTFPerplexityTask):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://lm-eval/medical_knowledge_evals/8_23/true_false/emergency_med_tf.csv")

class exercise_tf(MedicalTFPerplexityTask):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://lm-eval/medical_knowledge_evals/8_23/true_false/exercise_tf.csv")

class fertility_tf(MedicalTFPerplexityTask):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://lm-eval/medical_knowledge_evals/8_23/true_false/fertility_tf.csv")

class gi_tf(MedicalTFPerplexityTask):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://lm-eval/medical_knowledge_evals/8_23/true_false/gi_tf.csv")

class hematology_tf(MedicalTFPerplexityTask):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://lm-eval/medical_knowledge_evals/8_23/true_false/hematology_tf.csv")

class hepatitis_tf(MedicalTFPerplexityTask):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://lm-eval/medical_knowledge_evals/8_23/true_false/hepatitis_tf.csv")

class herbal_medicine_tf(MedicalTFPerplexityTask):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://lm-eval/medical_knowledge_evals/8_23/true_false/herbal_medicine_tf.csv")

class hiv_aids_tf(MedicalTFPerplexityTask):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://lm-eval/medical_knowledge_evals/8_23/true_false/hiv_aids_tf.csv")

class home_remedies_tf(MedicalTFPerplexityTask):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://lm-eval/medical_knowledge_evals/8_23/true_false/home_remedies_tf.csv")

class homeopathy_tf(MedicalTFPerplexityTask):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://lm-eval/medical_knowledge_evals/8_23/true_false/homeopathy_tf.csv")

class hyperlipidemia_tf(MedicalTFPerplexityTask):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://lm-eval/medical_knowledge_evals/8_23/true_false/hyperlipidemia_tf.csv")

class ischemic_hd_tf(MedicalTFPerplexityTask):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://lm-eval/medical_knowledge_evals/8_23/true_false/ischemic_hd_tf.csv")

class joint_commission_tf(MedicalTFPerplexityTask):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://lm-eval/medical_knowledge_evals/8_23/true_false/joint_commission_tf.csv")

class medicaid_tf(MedicalTFPerplexityTask):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://lm-eval/medical_knowledge_evals/8_23/true_false/medicaid_tf.csv")

class medicare_tf(MedicalTFPerplexityTask):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://lm-eval/medical_knowledge_evals/8_23/true_false/medicare_tf.csv")

class nutrition_2_tf(MedicalTFPerplexityTask):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://lm-eval/medical_knowledge_evals/8_23/true_false/nutrition_2_tf.csv")

class nutrition_tf(MedicalTFPerplexityTask):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://lm-eval/medical_knowledge_evals/8_23/true_false/nutrition_tf.csv")

class obstetrics_tf(MedicalTFPerplexityTask):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://lm-eval/medical_knowledge_evals/8_23/true_false/obstetrics_tf.csv")

class osteoporosis_tf(MedicalTFPerplexityTask):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://lm-eval/medical_knowledge_evals/8_23/true_false/osteoporosis_tf.csv")

class schizophrenia_tf(MedicalTFPerplexityTask):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://lm-eval/medical_knowledge_evals/8_23/true_false/schizophrenia_tf.csv")

class stroke_tf(MedicalTFPerplexityTask):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://lm-eval/medical_knowledge_evals/8_23/true_false/stroke_tf.csv")

class substance_abuse_tf(MedicalTFPerplexityTask):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://lm-eval/medical_knowledge_evals/8_23/true_false/substance_abuse_tf.csv")

class uspstf_tf(MedicalTFPerplexityTask):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://lm-eval/medical_knowledge_evals/8_23/true_false/uspstf_tf.csv")

class vitamin_tf(MedicalTFPerplexityTask):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://lm-eval/medical_knowledge_evals/8_23/true_false/vitamin_tf.csv")

class alex_conversation_test_task(MedicalMCPerplexityTask):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://lm-eval/medical_knowledge_evals/8_23/alex_conversation_test_task.csv")

class drug_indications(MedicalMCPerplexityTask):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://lm-eval/medical_knowledge_evals/8_23/drug_indications.csv")
        
class drug_side_effects(MedicalMCPerplexityTask):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://lm-eval/medical_knowledge_evals/8_23/drug_side_effects.csv")

### fitb tasks

class all_fill_in_the_blank(MedicalMCTask):
    def __init__(self):
        super().__init__()
        self.dataset = get_s3_csv_dataset("s3://lm-eval/9_11_batch/all_fill_in_the_blank.csv")

class fitb_asd(MedicalMCPerplexityTask):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://lm-eval/medical_knowledge_evals/8_23/fitb/fitb_asd.csv")

class fitb_cancer(MedicalMCPerplexityTask):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://lm-eval/medical_knowledge_evals/8_23/fitb/fitb_cancer.csv")

class fitb_cardiology(MedicalMCPerplexityTask):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://lm-eval/medical_knowledge_evals/8_23/fitb/fitb_cardiology.csv")

class fitb_ckd(MedicalMCPerplexityTask):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://lm-eval/medical_knowledge_evals/8_23/fitb/fitb_ckd.csv")

class fitb_copd(MedicalMCPerplexityTask):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://lm-eval/medical_knowledge_evals/8_23/fitb/fitb_copd.csv")

class fitb_depression(MedicalMCPerplexityTask):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://lm-eval/medical_knowledge_evals/8_23/fitb/fitb_depression.csv")

class fitb_diabetes(MedicalMCPerplexityTask):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://lm-eval/medical_knowledge_evals/8_23/fitb/fitb_diabetes.csv")

class fitb_diet(MedicalMCPerplexityTask):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://lm-eval/medical_knowledge_evals/8_23/fitb/fitb_diet.csv")

class fitb_emergency_med(MedicalMCPerplexityTask):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://lm-eval/medical_knowledge_evals/8_23/fitb/fitb_emergency_med.csv")

class fitb_exercise_recommendations(MedicalMCPerplexityTask):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://lm-eval/medical_knowledge_evals/8_23/fitb/fitb_exercise_recommendations.csv")

class fitb_hepatitis(MedicalMCPerplexityTask):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://lm-eval/medical_knowledge_evals/8_23/fitb/fitb_hepatitis.csv")

class fitb_hiv_aids(MedicalMCPerplexityTask):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://lm-eval/medical_knowledge_evals/8_23/fitb/fitb_hiv_aids.csv")

class fitb_home_remedies(MedicalMCPerplexityTask):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://lm-eval/medical_knowledge_evals/8_23/fitb/fitb_home_remedies.csv")

class fitb_hyperlipidemia(MedicalMCPerplexityTask):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://lm-eval/medical_knowledge_evals/8_23/fitb/fitb_hyperlipidemia.csv")

class fitb_joint_commission(MedicalMCPerplexityTask):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://lm-eval/medical_knowledge_evals/8_23/fitb/fitb_joint_commission.csv")

class fitb_medicaid(MedicalMCPerplexityTask):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://lm-eval/medical_knowledge_evals/8_23/fitb/fitb_medicaid.csv")

class fitb_medicare(MedicalMCPerplexityTask):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://lm-eval/medical_knowledge_evals/8_23/fitb/fitb_medicare.csv")

class fitb_nutrition(MedicalMCPerplexityTask):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://lm-eval/medical_knowledge_evals/8_23/fitb/fitb_nutrition.csv")

class fitb_obstetrics(MedicalMCPerplexityTask):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://lm-eval/medical_knowledge_evals/8_23/fitb/fitb_obstetrics.csv")

class fitb_substance_abuse(MedicalMCPerplexityTask):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://lm-eval/medical_knowledge_evals/8_23/fitb/fitb_substance_abuse.csv")

class fitb_uspstf(MedicalMCPerplexityTask):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://lm-eval/medical_knowledge_evals/8_23/fitb/fitb_uspstf.csv")

class fitb_vitamins(MedicalMCPerplexityTask):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://lm-eval/medical_knowledge_evals/8_23/fitb/fitb_vitamins.csv")

class fitb_santa_1(MedicalMCPerplexityTask):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://lm-eval/9_11_batch/santa_1_fitb_questions.csv")

class fitb_santa_2(MedicalMCPerplexityTask):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://lm-eval/9_11_batch/santa_2_fitb_questions.csv")

class fitb_santa_3(MedicalMCPerplexityTask):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://lm-eval/9_11_batch/santa_3_fitb_questions.csv")

class fitb_santa_4(MedicalMCPerplexityTask):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://lm-eval/9_11_batch/santa_4_fitb_questions.csv")

class fitb_santa_5(MedicalMCPerplexityTask):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://lm-eval/9_11_batch/santa_5_fitb_questions.csv")

class fitb_santa_6(MedicalMCPerplexityTask):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://lm-eval/9_11_batch/santa_6_fitb_questions.csv")

class fitb_santa_7(MedicalMCPerplexityTask):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://lm-eval/9_11_batch/santa_7_fitb_questions.csv")

class fitb_santa_8(MedicalMCPerplexityTask):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://lm-eval/9_11_batch/santa_8_fitb_questions.csv")

class fitb_USPTF_1(MedicalMCPerplexityTask):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://lm-eval/9_11_batch/USPTF_1_fitb_questions.csv")

class fitb_USPTF_2(MedicalMCPerplexityTask):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://lm-eval/9_11_batch/USPTF_2_fitb_questions.csv")

class fitb_USPTF_3(MedicalMCPerplexityTask):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://lm-eval/9_11_batch/USPTF_3_fitb_questions.csv")

class fitb_JC(MedicalMCPerplexityTask):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://lm-eval/9_11_batch/JC_fitb_questions.csv")



### MedicalMCPerplexityTasksCSV
    
class chf_eval(MedicalMCPerplexityTask):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://lm-eval/medical_knowledge_evals/chf_eval.csv")
        
class dme_eval_1(MedicalMCPerplexityTask):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://lm-eval/medical_knowledge_evals/dme_eval_1.csv")
    
class drug_names_eval(MedicalMCPerplexityTask):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://lm-eval/medical_knowledge_evals/drug_names_eval.csv")
        
class regulatory_eval(MedicalMCPerplexityTask):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://lm-eval/medical_knowledge_evals/regulatory_eval.csv")
    
class UMLS_definitions_eval_A(MedicalMCPerplexityTask):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://lm-eval/medical_knowledge_evals/UMLS_definitions_eval_A.csv")
    
class UMLS_definitions_eval_B(MedicalMCPerplexityTask):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://lm-eval/medical_knowledge_evals/UMLS_definitions_eval_B.csv")
        
### MedicalCertificationTasks

class ABFM(MedicalCertificationTask):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://exam-benchmarks/parsed_mcqs/ABFM.csv")

class ABFM_sports(MedicalCertificationTask):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://exam-benchmarks/parsed_mcqs/ABFM_sports.csv")

class ABIM_CV(MedicalCertificationTask):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://exam-benchmarks/parsed_mcqs/ABIM_CV.csv")

class ABIM_ID(MedicalCertificationTask):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://exam-benchmarks/parsed_mcqs/ABIM_ID.csv")

class AB_PULMCC(MedicalCertificationTask):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://exam-benchmarks/parsed_mcqs/AB_PULMCC.csv")

class AB_obgyn(MedicalCertificationTask):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://exam-benchmarks/parsed_mcqs/AB_obgyn.csv")

class ACHPN(MedicalCertificationTask):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://exam-benchmarks/parsed_mcqs/ACHPN.csv")

class ACLS(MedicalCertificationTask):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://exam-benchmarks/parsed_mcqs/ACLS.csv")

class ACLS_2(MedicalCertificationTask):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://exam-benchmarks/parsed_mcqs/ACLS_2.csv")

class ADEX_hygienist(MedicalCertificationTask):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://exam-benchmarks/parsed_mcqs/ADEX_hygienist.csv")

class AHIMA_CCA(MedicalCertificationTask):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://exam-benchmarks/parsed_mcqs/AHIMA_CCA.csv")

class AMT_1(MedicalCertificationTask):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://exam-benchmarks/parsed_mcqs/AMT_1.csv")

class AMT_2(MedicalCertificationTask):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://exam-benchmarks/parsed_mcqs/AMT_2.csv")

class ASCP_HT(MedicalCertificationTask):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://exam-benchmarks/parsed_mcqs/ASCP_HT.csv")

class CASCG_multi(MedicalCertificationTask):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://exam-benchmarks/parsed_mcqs/CASCG_multi.csv")

class CCHT(MedicalCertificationTask):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://exam-benchmarks/parsed_mcqs/CCHT.csv")

class CCMA_parsed(MedicalCertificationTask):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://exam-benchmarks/parsed_mcqs/CCMA_parsed.csv")

class CDM(MedicalCertificationTask):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://exam-benchmarks/parsed_mcqs/CDM.csv")

class CEHRS(MedicalCertificationTask):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://exam-benchmarks/parsed_mcqs/CEHRS.csv")

class CEN(MedicalCertificationTask):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://exam-benchmarks/parsed_mcqs/CEN.csv")

class CHES(MedicalCertificationTask):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://exam-benchmarks/parsed_mcqs/CHES.csv")

class CMAA(MedicalCertificationTask):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://exam-benchmarks/parsed_mcqs/CMAA.csv")

class CMSRN_med_surg_nursing(MedicalCertificationTask):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://exam-benchmarks/parsed_mcqs/CMSRN_med_surg_nursing.csv")

class CNA(MedicalCertificationTask):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://exam-benchmarks/parsed_mcqs/CNA.csv")

class COBCG_2(MedicalCertificationTask):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://exam-benchmarks/parsed_mcqs/COBCG_2.csv")

class COBCG_multi(MedicalCertificationTask):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://exam-benchmarks/parsed_mcqs/COBCG_multi.csv")

class COMLEX(MedicalCertificationTask):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://exam-benchmarks/parsed_mcqs/COMLEX.csv")

class COMLEX_2(MedicalCertificationTask):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://exam-benchmarks/parsed_mcqs/COMLEX_2.csv")

class COMT(MedicalCertificationTask):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://exam-benchmarks/parsed_mcqs/COMT.csv")

class CPCT(MedicalCertificationTask):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://exam-benchmarks/parsed_mcqs/CPCT.csv")

class CPHIMS(MedicalCertificationTask):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://exam-benchmarks/parsed_mcqs/CPHIMS.csv")

class CPHQ(MedicalCertificationTask):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://exam-benchmarks/parsed_mcqs/CPHQ.csv")

class CPJE(MedicalCertificationTask):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://exam-benchmarks/parsed_mcqs/CPJE.csv")

class CPR_quiz(MedicalCertificationTask):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://exam-benchmarks/parsed_mcqs/CPR_quiz.csv")

class CPT(MedicalCertificationTask):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://exam-benchmarks/parsed_mcqs/CPT.csv")

class CPhT(MedicalCertificationTask):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://exam-benchmarks/parsed_mcqs/CPhT.csv")

class CRRN(MedicalCertificationTask):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://exam-benchmarks/parsed_mcqs/CRRN.csv")

class CSCS(MedicalCertificationTask):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://exam-benchmarks/parsed_mcqs/CSCS.csv")

class CV_NP(MedicalCertificationTask):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://exam-benchmarks/parsed_mcqs/CV_NP.csv")

class Certified_Gastro_RN(MedicalCertificationTask):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://exam-benchmarks/parsed_mcqs/Certified_Gastro_RN.csv")

class Chemical_Dependency_Counselor(MedicalCertificationTask):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://exam-benchmarks/parsed_mcqs/Chemical Dependency Counselor.csv")

class EMT(MedicalCertificationTask):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://exam-benchmarks/parsed_mcqs/EMT.csv")

class EM_Module(MedicalCertificationTask):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://exam-benchmarks/parsed_mcqs/EM_Module.csv")

class Family_Nurse_Practitioner(MedicalCertificationTask):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://exam-benchmarks/parsed_mcqs/Family Nurse Practitioner.csv")

class HCISPP(MedicalCertificationTask):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://exam-benchmarks/parsed_mcqs/HCISPP.csv")

class HIPAA_MA_Security(MedicalCertificationTask):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://exam-benchmarks/parsed_mcqs/HIPAA_MA_Security.csv")

class HIPAA_Privacy(MedicalCertificationTask):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://exam-benchmarks/parsed_mcqs/HIPAA_Privacy.csv")

class MACE(MedicalCertificationTask):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://exam-benchmarks/parsed_mcqs/MACE.csv")

class MEDMCQA_TEST(MedicalCertificationTask):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://exam-benchmarks/parsed_mcqs/MEDMCQA_TEST.csv")

class MEDMCQA_TRAIN(MedicalCertificationTask):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://exam-benchmarks/parsed_mcqs/MEDMCQA_TRAIN.csv")

class MEDMCQA_VALIDATION(MedicalCertificationTask):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://exam-benchmarks/parsed_mcqs/MEDMCQA_VALIDATION.csv")

# class MEDMCQA_quick_100(MedicalCertificationTask):
#     def __init__(self):
#         self.dataset = get_s3_csv_dataset("s3://exam-benchmarks/parsed_mcqs/MEDMCQA_quick_100.csv")

# class MEDMCQA_sample1_1000(MedicalCertificationTask):
#     def __init__(self):
#         self.dataset = get_s3_csv_dataset("s3://exam-benchmarks/parsed_mcqs/MEDMCQA_sample1_1000.csv")

# class MEDMCQA_sample1_2000(MedicalCertificationTask):
#     def __init__(self):
#         self.dataset = get_s3_csv_dataset("s3://exam-benchmarks/parsed_mcqs/MEDMCQA_sample1_2000.csv")

# class MEDMCQA_sample1_3000(MedicalCertificationTask):
#     def __init__(self):
#         self.dataset = get_s3_csv_dataset("s3://exam-benchmarks/parsed_mcqs/MEDMCQA_sample1_3000.csv")

class MEDQA_TEST(MedicalCertificationTask):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://exam-benchmarks/parsed_mcqs/MEDQA_TEST.csv")

class MEDQA_TRAIN(MedicalCertificationTask):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://exam-benchmarks/parsed_mcqs/MEDQA_TRAIN.csv")

# class MEDQA_quick_100(MedicalCertificationTask):
#     def __init__(self):
#         self.dataset = get_s3_csv_dataset("s3://exam-benchmarks/parsed_mcqs/MEDQA_quick_100.csv")

# class MEDQA_sample1_1000(MedicalCertificationTask):
#     def __init__(self):
#         self.dataset = get_s3_csv_dataset("s3://exam-benchmarks/parsed_mcqs/MEDQA_sample1_1000.csv")

# class MEDQA_sample1_2000(MedicalCertificationTask):
#     def __init__(self):
#         self.dataset = get_s3_csv_dataset("s3://exam-benchmarks/parsed_mcqs/MEDQA_sample1_2000.csv")

# class MEDQA_sample1_3000(MedicalCertificationTask):
#     def __init__(self):
#         self.dataset = get_s3_csv_dataset("s3://exam-benchmarks/parsed_mcqs/MEDQA_sample1_3000.csv")

class MPJE(MedicalCertificationTask):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://exam-benchmarks/parsed_mcqs/MPJE.csv")

class NAHQ_CPHQ(MedicalCertificationTask):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://exam-benchmarks/parsed_mcqs/NAHQ_CPHQ.csv")

class NCLEX_PN(MedicalCertificationTask):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://exam-benchmarks/parsed_mcqs/NCLEX_PN.csv")

class NCMHCE(MedicalCertificationTask):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://exam-benchmarks/parsed_mcqs/NCMHCE.csv")

class NLN_pax(MedicalCertificationTask):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://exam-benchmarks/parsed_mcqs/NLN_pax.csv")

class NNAAP(MedicalCertificationTask):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://exam-benchmarks/parsed_mcqs/NNAAP.csv")

class PCCN(MedicalCertificationTask):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://exam-benchmarks/parsed_mcqs/PCCN.csv")

class PNCB_Acute_Care(MedicalCertificationTask):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://exam-benchmarks/parsed_mcqs/PNCB Acute Care.csv")

class PNCB_CPB(MedicalCertificationTask):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://exam-benchmarks/parsed_mcqs/PNCB CPB.csv")

class PNCB_PMHS(MedicalCertificationTask):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://exam-benchmarks/parsed_mcqs/PNCB PMHS.csv")

class PNCB_Primary_Care(MedicalCertificationTask):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://exam-benchmarks/parsed_mcqs/PNCB Primary Care.csv")

class PTCB(MedicalCertificationTask):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://exam-benchmarks/parsed_mcqs/PTCB.csv")

# class PUBMEDQA_LABELED(MedicalCertificationTask):
#     def __init__(self):
#         self.dataset = get_s3_csv_dataset("s3://exam-benchmarks/parsed_mcqs/PUBMEDQA_LABELED.csv")

# class PUBMEDQA_LABELED_1000(MedicalCertificationTask):
#     def __init__(self):
#         self.dataset = get_s3_csv_dataset("s3://exam-benchmarks/parsed_mcqs/PUBMEDQA_LABELED_1000.csv")

# class PUBMEDQA_W_CONTEXT_1000(MedicalCertificationTask):
#     def __init__(self):
#         self.dataset = get_s3_csv_dataset("s3://exam-benchmarks/parsed_mcqs/PUBMEDQA_W_CONTEXT_1000.csv")

class RD_A(MedicalCertificationTask):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://exam-benchmarks/parsed_mcqs/RD_A.csv")

class RMA_AMT(MedicalCertificationTask):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://exam-benchmarks/parsed_mcqs/RMA_AMT.csv")

class RNAS_C(MedicalCertificationTask):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://exam-benchmarks/parsed_mcqs/RNAS_C.csv")

class RPSGT(MedicalCertificationTask):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://exam-benchmarks/parsed_mcqs/RPSGT.csv")

class SPT(MedicalCertificationTask):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://exam-benchmarks/parsed_mcqs/SPT.csv")

class WH_NP(MedicalCertificationTask):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://exam-benchmarks/parsed_mcqs/WH_NP.csv")

class acep_PEER_exam_275qa_parsed(MedicalCertificationTask):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://exam-benchmarks/parsed_mcqs/acep_PEER_exam_275qa_parsed.csv")

class ada_dental_board(MedicalCertificationTask):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://exam-benchmarks/parsed_mcqs/ada_dental_board.csv")

class bls_quiz(MedicalCertificationTask):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://exam-benchmarks/parsed_mcqs/bls_quiz.csv")

class cdeo_A(MedicalCertificationTask):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://exam-benchmarks/parsed_mcqs/cdeo_A.csv")

class certified_audiologist(MedicalCertificationTask):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://exam-benchmarks/parsed_mcqs/certified_audiologist.csv")

class certified_wound_specialist(MedicalCertificationTask):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://exam-benchmarks/parsed_mcqs/certified_wound_specialist.csv")

class coc_A(MedicalCertificationTask):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://exam-benchmarks/parsed_mcqs/coc_A.csv")

class compliance_quiz(MedicalCertificationTask):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://exam-benchmarks/parsed_mcqs/compliance_quiz.csv")

class cpb_A(MedicalCertificationTask):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://exam-benchmarks/parsed_mcqs/cpb_A.csv")

class cpc_A(MedicalCertificationTask):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://exam-benchmarks/parsed_mcqs/cpc_A.csv")

class cpco_A(MedicalCertificationTask):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://exam-benchmarks/parsed_mcqs/cpco_A.csv")

class cpma_A(MedicalCertificationTask):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://exam-benchmarks/parsed_mcqs/cpma_A.csv")

class cppm_A(MedicalCertificationTask):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://exam-benchmarks/parsed_mcqs/cppm_A.csv")

class crc_A(MedicalCertificationTask):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://exam-benchmarks/parsed_mcqs/crc_A.csv")

class cultural_competency(MedicalCertificationTask):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://exam-benchmarks/parsed_mcqs/cultural_competency.csv")

class dental_assistant(MedicalCertificationTask):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://exam-benchmarks/parsed_mcqs/dental_assistant.csv")

class dental_board(MedicalCertificationTask):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://exam-benchmarks/parsed_mcqs/dental_board.csv")

class diabetes_A(MedicalCertificationTask):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://exam-benchmarks/parsed_mcqs/diabetes_A.csv")

class dietician_B(MedicalCertificationTask):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://exam-benchmarks/parsed_mcqs/dietician_B.csv")

class disability(MedicalCertificationTask):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://exam-benchmarks/parsed_mcqs/disability.csv")

class employer_discrimination(MedicalCertificationTask):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://exam-benchmarks/parsed_mcqs/employer_discrimination.csv")

class ent_md_board(MedicalCertificationTask):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://exam-benchmarks/parsed_mcqs/ent_md_board.csv")

class fitness_nutrition(MedicalCertificationTask):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://exam-benchmarks/parsed_mcqs/fitness_nutrition.csv")

class fraud_waste_abuse_quiz(MedicalCertificationTask):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://exam-benchmarks/parsed_mcqs/fraud_waste_abuse_quiz.csv")

class gender(MedicalCertificationTask):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://exam-benchmarks/parsed_mcqs/gender.csv")

class heme_onc(MedicalCertificationTask):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://exam-benchmarks/parsed_mcqs/heme_onc.csv")

class hipaa_revised(MedicalCertificationTask):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://exam-benchmarks/parsed_mcqs/hipaa_revised.csv")

class hospice_md_board(MedicalCertificationTask):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://exam-benchmarks/parsed_mcqs/hospice_md_board.csv")

class hospital_safety_quiz(MedicalCertificationTask):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://exam-benchmarks/parsed_mcqs/hospital_safety_quiz.csv")

class lactation_consultant(MedicalCertificationTask):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://exam-benchmarks/parsed_mcqs/lactation_consultant.csv")

class naplex(MedicalCertificationTask):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://exam-benchmarks/parsed_mcqs/naplex.csv")

class nclex_350qa_parsed_test(MedicalCertificationTask):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://exam-benchmarks/parsed_mcqs/nclex_350qa_parsed_test.csv")

class nephrology_md_board(MedicalCertificationTask):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://exam-benchmarks/parsed_mcqs/nephrology_md_board.csv")

class neurology(MedicalCertificationTask):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://exam-benchmarks/parsed_mcqs/neurology.csv")

class nse(MedicalCertificationTask):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://exam-benchmarks/parsed_mcqs/nse.csv")

class nutritionist(MedicalCertificationTask):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://exam-benchmarks/parsed_mcqs/nutritionist.csv")

class optho_md_board(MedicalCertificationTask):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://exam-benchmarks/parsed_mcqs/optho_md_board.csv")

class psychiatry(MedicalCertificationTask):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://exam-benchmarks/parsed_mcqs/psychiatry.csv")

# class pubmedqa_100(MedicalCertificationTask):
#     def __init__(self):
#         self.dataset = get_s3_csv_dataset("s3://exam-benchmarks/parsed_mcqs/pubmedqa_100.csv")

# class pubmedqa_900(MedicalCertificationTask):
#     def __init__(self):
#         self.dataset = get_s3_csv_dataset("s3://exam-benchmarks/parsed_mcqs/pubmedqa_900.csv")

# class pubmedqa_middle_800(MedicalCertificationTask):
#     def __init__(self):
#         self.dataset = get_s3_csv_dataset("s3://exam-benchmarks/parsed_mcqs/pubmedqa_middle_800.csv")

class race_AA(MedicalCertificationTask):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://exam-benchmarks/parsed_mcqs/race_AA.csv")

class resp_therapy(MedicalCertificationTask):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://exam-benchmarks/parsed_mcqs/resp_therapy.csv")

class rheumatology_md_boards(MedicalCertificationTask):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://exam-benchmarks/parsed_mcqs/rheumatology_md_boards.csv")

class senior_dementia(MedicalCertificationTask):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://exam-benchmarks/parsed_mcqs/senior_dementia.csv")

class sleep_medicine(MedicalCertificationTask):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://exam-benchmarks/parsed_mcqs/sleep_medicine.csv")

class social_worker(MedicalCertificationTask):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://exam-benchmarks/parsed_mcqs/social_worker.csv")

class step1_2018(MedicalCertificationTask):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://exam-benchmarks/parsed_mcqs/step1_2018.csv")

class step1_2019(MedicalCertificationTask):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://exam-benchmarks/parsed_mcqs/step1_2019.csv")

class step1_2020(MedicalCertificationTask):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://exam-benchmarks/parsed_mcqs/step1_2020.csv")

class step1_2021(MedicalCertificationTask):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://exam-benchmarks/parsed_mcqs/step1_2021.csv")

class step1_qa_parsed(MedicalCertificationTask):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://exam-benchmarks/parsed_mcqs/step1_qa_parsed.csv")

class step2_qa_parsed(MedicalCertificationTask):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://exam-benchmarks/parsed_mcqs/step2_qa_parsed.csv")

class step3_qa_parsed(MedicalCertificationTask):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://exam-benchmarks/parsed_mcqs/step3_qa_parsed.csv")

class surgery_md_board(MedicalCertificationTask):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://exam-benchmarks/parsed_mcqs/surgery_md_board.csv")

class urology_md_board(MedicalCertificationTask):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://exam-benchmarks/parsed_mcqs/urology_md_board.csv")

class val_AB_PULMCC(MedicalCertificationTask):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://exam-benchmarks/parsed_mcqs/val_AB_PULMCC.csv")

class val_CASCG(MedicalCertificationTask):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://exam-benchmarks/parsed_mcqs/val_CASCG.csv")

class val_COBCG(MedicalCertificationTask):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://exam-benchmarks/parsed_mcqs/val_COBCG.csv")

class val_heme_onc(MedicalCertificationTask):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://exam-benchmarks/parsed_mcqs/val_heme_onc.csv")

class val_hipaa(MedicalCertificationTask):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://exam-benchmarks/parsed_mcqs/val_hipaa.csv")

class val_neurology(MedicalCertificationTask):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://exam-benchmarks/parsed_mcqs/val_neurology.csv")

class val_obgyn(MedicalCertificationTask):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://exam-benchmarks/parsed_mcqs/val_obgyn.csv")

class val_psychiatry(MedicalCertificationTask):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://exam-benchmarks/parsed_mcqs/val_psychiatry.csv")

class val_sleep_medicine(MedicalCertificationTask):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://exam-benchmarks/parsed_mcqs/val_sleep_medicine.csv")

class val_sports_medicine(MedicalCertificationTask):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://exam-benchmarks/parsed_mcqs/val_sports_medicine.csv")

class val_step_1(MedicalCertificationTask):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://exam-benchmarks/parsed_mcqs/val_step_1.csv")

class val_step_2(MedicalCertificationTask):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://exam-benchmarks/parsed_mcqs/val_step_2.csv")

class vascular_access_tech(MedicalCertificationTask):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://exam-benchmarks/parsed_mcqs/vascular_access_tech.csv")

class wh_np_2(MedicalCertificationTask):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://exam-benchmarks/parsed_mcqs/wh_np_2.csv")

class workplace_harassment(MedicalCertificationTask):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://exam-benchmarks/parsed_mcqs/workplace_harassment.csv")
        
### 8_16_batch

class alternative_med(MedicalMCPerplexityTask_8_16_Forward):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://lm-eval/medical_knowledge_evals/8_16_batch/alternative_med.csv")
        
class alzheimers(MedicalMCPerplexityTask_8_16_Forward):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://lm-eval/medical_knowledge_evals/8_16_batch/alzheimers.csv")
        
class arthritis(MedicalMCPerplexityTask_8_16_Forward):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://lm-eval/medical_knowledge_evals/8_16_batch/arthritis.csv")
        
class asd(MedicalMCPerplexityTask_8_16_Forward):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://lm-eval/medical_knowledge_evals/8_16_batch/asd.csv")
        
class cancer_general(MedicalMCPerplexityTask_8_16_Forward):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://lm-eval/medical_knowledge_evals/8_16_batch/cancer_general.csv")
        
class ckd(MedicalMCPerplexityTask_8_16_Forward):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://lm-eval/medical_knowledge_evals/8_16_batch/ckd.csv")
        
class cms(MedicalMCPerplexityTask_8_16_Forward):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://lm-eval/medical_knowledge_evals/8_16_batch/cms.csv")
        
class community_resources(MedicalMCPerplexityTask_8_16_Forward):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://lm-eval/medical_knowledge_evals/8_16_batch/community_resources.csv")
        
class copd(MedicalMCPerplexityTask_8_16_Forward):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://lm-eval/medical_knowledge_evals/8_16_batch/copd.csv")
        
class depression(MedicalMCPerplexityTask_8_16_Forward):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://lm-eval/medical_knowledge_evals/8_16_batch/depression.csv")
        
class diabetes(MedicalMCPerplexityTask_8_16_Forward):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://lm-eval/medical_knowledge_evals/8_16_batch/diabetes.csv")
        
class diabetes_2(MedicalMCPerplexityTask_8_16_Forward):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://lm-eval/medical_knowledge_evals/8_16_batch/diabetes_2.csv")
        
class drug_discount(MedicalMCPerplexityTask_8_16_Forward):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://lm-eval/medical_knowledge_evals/8_16_batch/drug_discount.csv")
        
class fertility(MedicalMCPerplexityTask_8_16_Forward):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://lm-eval/medical_knowledge_evals/8_16_batch/fertility.csv")
        
class heart_disease(MedicalMCPerplexityTask_8_16_Forward):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://lm-eval/medical_knowledge_evals/8_16_batch/heart_disease.csv")
        
class heart_failure(MedicalMCPerplexityTask_8_16_Forward):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://lm-eval/medical_knowledge_evals/8_16_batch/heart_failure.csv")
        
class hld(MedicalMCPerplexityTask_8_16_Forward):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://lm-eval/medical_knowledge_evals/8_16_batch/hld.csv")
        
class htn(MedicalMCPerplexityTask_8_16_Forward):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://lm-eval/medical_knowledge_evals/8_16_batch/htn.csv")
        
class joint_commission(MedicalMCPerplexityTask_8_16_Forward):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://lm-eval/medical_knowledge_evals/8_16_batch/joint_commission.csv")
        
class medicaid(MedicalMCPerplexityTask_8_16_Forward):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://lm-eval/medical_knowledge_evals/8_16_batch/medicaid.csv")
        
class medicare(MedicalMCPerplexityTask_8_16_Forward):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://lm-eval/medical_knowledge_evals/8_16_batch/medicare.csv")
        
class obstetrics(MedicalMCPerplexityTask_8_16_Forward):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://lm-eval/medical_knowledge_evals/8_16_batch/obstetrics.csv")
        
class osteoporosis(MedicalMCPerplexityTask_8_16_Forward):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://lm-eval/medical_knowledge_evals/8_16_batch/osteoporosis.csv")
        
class schizophrenia(MedicalMCPerplexityTask_8_16_Forward):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://lm-eval/medical_knowledge_evals/8_16_batch/schizophrenia.csv")
        
class stroke(MedicalMCPerplexityTask_8_16_Forward):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://lm-eval/medical_knowledge_evals/8_16_batch/stroke.csv")
        
class substance_abuse(MedicalMCPerplexityTask_8_16_Forward):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://lm-eval/medical_knowledge_evals/8_16_batch/substance_abuse.csv")
        
class vitamins(MedicalMCPerplexityTask_8_16_Forward):
    def __init__(self):
        self.dataset = get_s3_csv_dataset("s3://lm-eval/medical_knowledge_evals/8_16_batch/vitamins.csv")

GOLD_SET = {
    "esrd_combined": esrd_combined,
    "knowledge_evals_combined": knowledge_evals_combined,
    "knowledge_evals_combined_v1": knowledge_evals_combined_v1,
    "knowledge_evals_combined_v2": knowledge_evals_combined_v2,
    "SL_conditions_completion": SL_conditions_completion,
    "SL_drug_completion": SL_drug_completion,
    "SL_symptom_completion": SL_symptom_completion,
    "tkr_combined": tkr_combined,
    "high_quality_v1": high_quality_v1,
    "high_quality_v2": high_quality_v2,
    "high_quality_v3": high_quality_v3,
    "hqm_v2": hqm_v2,
    "SL_handpicked_evals": SL_handpicked_evals,
    "SL_sentence_selection": SL_sentence_selection,
}

TASKS_8_16_BATCH = {
    "alternative_med": alternative_med,
    "alzheimers": alzheimers,
    "arthritis": arthritis,
    "asd": asd,
    "cancer_general": cancer_general,
    "ckd": ckd,
    "cms": cms,
    "community_resources": community_resources,
    "copd": copd,
    "depression": depression,
    "diabetes": diabetes,
    "diabetes_2": diabetes_2,
    "drug_discount": drug_discount,
    "fertility": fertility,
    "heart_disease": heart_disease,
    "heart_failure": heart_failure,
    "hld": hld,
    "htn": htn,
    "joint_commission": joint_commission,
    "medicaid": medicaid,
    "medicare": medicare,
    "obstetrics": obstetrics,
    "osteoporosis": osteoporosis,
    "schizophrenia": schizophrenia,
    "stroke": stroke,
    "substance_abuse": substance_abuse,
    "vitamins": vitamins,
}

MEDICAL_MCQ_TASKS = {
    "ABFM": ABFM,
    "ABFM_sports": ABFM_sports,
    "ABIM_CV": ABIM_CV,
    "ABIM_ID": ABIM_ID,
    "AB_PULMCC": AB_PULMCC,
    "AB_obgyn": AB_obgyn,
    "ACHPN": ACHPN,
    "ACLS": ACLS,
    "ACLS_2": ACLS_2,
    "ADEX_hygienist": ADEX_hygienist,
    "AHIMA_CCA": AHIMA_CCA,
    "AMT_1": AMT_1,
    "AMT_2": AMT_2,
    "ASCP_HT": ASCP_HT,
    "CASCG_multi": CASCG_multi,
    "CCHT": CCHT,
    "CCMA_parsed": CCMA_parsed,
    "CDM": CDM,
    "CEHRS": CEHRS,
    "CEN": CEN,
    "CHES": CHES,
    "CMAA": CMAA,
    "CMSRN_med_surg_nursing": CMSRN_med_surg_nursing,
    "CNA": CNA,
    "COBCG_2": COBCG_2,
    "COBCG_multi": COBCG_multi,
    "COMLEX": COMLEX,
    "COMLEX_2": COMLEX_2,
    "COMT": COMT,
    "CPCT": CPCT,
    "CPHIMS": CPHIMS,
    "CPHQ": CPHQ,
    "CPJE": CPJE,
    "CPR_quiz": CPR_quiz,
    "CPT": CPT,
    "CPhT": CPhT,
    "CRRN": CRRN,
    "CSCS": CSCS,
    "CV_NP": CV_NP,
    "Certified_Gastro_RN": Certified_Gastro_RN,
    "Chemical_Dependency_Counselor": Chemical_Dependency_Counselor,
    "EMT": EMT,
    "EM_Module": EM_Module,
    "Family_Nurse_Practitioner": Family_Nurse_Practitioner,
    "HCISPP": HCISPP,
    "HIPAA_MA_Security": HIPAA_MA_Security,
    "HIPAA_Privacy": HIPAA_Privacy,
    "MACE": MACE,
    "MEDMCQA_TEST": MEDMCQA_TEST,
    "MEDMCQA_TRAIN": MEDMCQA_TRAIN,
    "MEDMCQA_VALIDATION": MEDMCQA_VALIDATION,
    # "MEDMCQA_quick_100": MEDMCQA_quick_100,
    # "MEDMCQA_sample1_1000": MEDMCQA_sample1_1000,
    # "MEDMCQA_sample1_2000": MEDMCQA_sample1_2000,
    # "MEDMCQA_sample1_3000": MEDMCQA_sample1_3000,
    "MEDQA_TEST": MEDQA_TEST,
    "MEDQA_TRAIN": MEDQA_TRAIN,
    # "MEDQA_quick_100": MEDQA_quick_100,
    # "MEDQA_sample1_1000": MEDQA_sample1_1000,
    # "MEDQA_sample1_2000": MEDQA_sample1_2000,
    # "MEDQA_sample1_3000": MEDQA_sample1_3000,
    "MPJE": MPJE,
    "NAHQ_CPHQ": NAHQ_CPHQ,
    "NCLEX_PN": NCLEX_PN,
    "NCMHCE": NCMHCE,
    "NLN_pax": NLN_pax,
    "NNAAP": NNAAP,
    "PCCN": PCCN,
    "PNCB_Acute_Care": PNCB_Acute_Care,
    "PNCB_CPB": PNCB_CPB,
    "PNCB_PMHS": PNCB_PMHS,
    "PNCB_Primary_Care": PNCB_Primary_Care,
    "PTCB": PTCB,
    # "PUBMEDQA_LABELED": PUBMEDQA_LABELED,
    # "PUBMEDQA_LABELED_1000": PUBMEDQA_LABELED_1000,
    # "PUBMEDQA_W_CONTEXT_1000": PUBMEDQA_W_CONTEXT_1000,
    "RD_A": RD_A,
    "RMA_AMT": RMA_AMT,
    "RNAS_C": RNAS_C,
    "RPSGT": RPSGT,
    "SPT": SPT,
    "WH_NP": WH_NP,
    "acep_PEER_exam_275qa_parsed": acep_PEER_exam_275qa_parsed,
    "ada_dental_board": ada_dental_board,
    "bls_quiz": bls_quiz,
    "cdeo_A": cdeo_A,
    "certified_audiologist": certified_audiologist,
    "certified_wound_specialist": certified_wound_specialist,
    "coc_A": coc_A,
    "compliance_quiz": compliance_quiz,
    "cpb_A": cpb_A,
    "cpc_A": cpc_A,
    "cpco_A": cpco_A,
    "cpma_A": cpma_A,
    "cppm_A": cppm_A,
    "crc_A": crc_A,
    "cultural_competency": cultural_competency,
    "dental_assistant": dental_assistant,
    "dental_board": dental_board,
    "diabetes_A": diabetes_A,
    "dietician_B": dietician_B,
    "disability": disability,
    "employer_discrimination": employer_discrimination,
    "ent_md_board": ent_md_board,
    "fitness_nutrition": fitness_nutrition,
    "fraud_waste_abuse_quiz": fraud_waste_abuse_quiz,
    "gender": gender,
    "heme_onc": heme_onc,
    "hipaa_revised": hipaa_revised,
    "hospice_md_board": hospice_md_board,
    "hospital_safety_quiz": hospital_safety_quiz,
    "lactation_consultant": lactation_consultant,
    "naplex": naplex,
    "nclex_350qa_parsed_test": nclex_350qa_parsed_test,
    "nephrology_md_board": nephrology_md_board,
    "neurology": neurology,
    "nse": nse,
    "nutritionist": nutritionist,
    "optho_md_board": optho_md_board,
    "psychiatry": psychiatry,
    # "pubmedqa_100": pubmedqa_100,
    # "pubmedqa_900": pubmedqa_900,
    # "pubmedqa_middle_800": pubmedqa_middle_800,
    "race_AA": race_AA,
    "resp_therapy": resp_therapy,
    "rheumatology_md_boards": rheumatology_md_boards,
    "senior_dementia": senior_dementia,
    "sleep_medicine": sleep_medicine,
    "social_worker": social_worker,
    "step1_2018": step1_2018,
    "step1_2019": step1_2019,
    "step1_2020": step1_2020,
    "step1_2021": step1_2021,
    "step1_qa_parsed": step1_qa_parsed,
    "step2_qa_parsed": step2_qa_parsed,
    "step3_qa_parsed": step3_qa_parsed,
    "surgery_md_board": surgery_md_board,
    "urology_md_board": urology_md_board,
    # "val_AB_PULMCC": val_AB_PULMCC,
    # "val_CASCG": val_CASCG,
    # "val_COBCG": val_COBCG,
    # "val_heme_onc": val_heme_onc,
    # "val_hipaa": val_hipaa,
    # "val_neurology": val_neurology,
    # "val_obgyn": val_obgyn,
    # "val_psychiatry": val_psychiatry,
    # "val_sleep_medicine": val_sleep_medicine,
    # "val_sports_medicine": val_sports_medicine,
    # "val_step_1": val_step_1,
    # "val_step_2": val_step_2,
    "vascular_access_tech": vascular_access_tech,
    "wh_np_2": wh_np_2,
    "workplace_harassment": workplace_harassment,
}

TASKS_8_23_BATCH = {
    "alex_conversation_test_task": alex_conversation_test_task,
    "drug_indications": drug_indications,
    "drug_side_effects": drug_side_effects,
    "fitb_asd": fitb_asd,
    "fitb_cancer": fitb_cancer,
    "fitb_cardiology": fitb_cardiology,
    "fitb_ckd": fitb_ckd,
    "fitb_copd": fitb_copd,
    "fitb_depression": fitb_depression,
    "fitb_diabetes": fitb_diabetes,
    "fitb_diet": fitb_diet,
    "fitb_emergency_med": fitb_emergency_med,
    "fitb_exercise_recommendations": fitb_exercise_recommendations,
    "fitb_hepatitis": fitb_hepatitis,
    "fitb_hiv_aids": fitb_hiv_aids,
    "fitb_home_remedies": fitb_home_remedies,
    "fitb_hyperlipidemia": fitb_hyperlipidemia,
    "fitb_joint_commission": fitb_joint_commission,
    "fitb_medicaid": fitb_medicaid,
    "fitb_medicare": fitb_medicare,
    "fitb_nutrition": fitb_nutrition,
    "fitb_obstetrics": fitb_obstetrics,
    "fitb_substance_abuse": fitb_substance_abuse,
    "fitb_uspstf": fitb_uspstf,
    "fitb_vitamins": fitb_vitamins,
}

TF_8_23_TASKS = {
    "cancer_tf": cancer_tf,
    "cardiology_tf": cardiology_tf,
    "chf_tf": chf_tf,
    "ckd_tf": ckd_tf,
    "cms_tf": cms_tf,
    "copd_tf": copd_tf,
    "depression_tf": depression_tf,
    "diabetes_tf": diabetes_tf,
    "diet_tf": diet_tf,
    "drug_tiers_tf": drug_tiers_tf,
    "emergency_med_tf": emergency_med_tf,
    "exercise_tf": exercise_tf,
    "fertility_tf": fertility_tf,
    "gi_tf": gi_tf,
    "hematology_tf": hematology_tf,
    "hepatitis_tf": hepatitis_tf,
    "herbal_medicine_tf": herbal_medicine_tf,
    "hiv_aids_tf": hiv_aids_tf,
    "home_remedies_tf": home_remedies_tf,
    "homeopathy_tf": homeopathy_tf,
    "hyperlipidemia_tf": hyperlipidemia_tf,
    "ischemic_hd_tf": ischemic_hd_tf,
    "joint_commission_tf": joint_commission_tf,
    "medicaid_tf": medicaid_tf,
    "medicare_tf": medicare_tf,
    "nutrition_2_tf": nutrition_2_tf,
    "nutrition_tf": nutrition_tf,
    "obstetrics_tf": obstetrics_tf,
    "osteoporosis_tf": osteoporosis_tf,
    "schizophrenia_tf": schizophrenia_tf,
    "stroke_tf": stroke_tf,
    "substance_abuse_tf": substance_abuse_tf,
    "uspstf_tf": uspstf_tf,
    "vitamin_tf": vitamin_tf,
}

FITB_9_11 ={
    "fitb_santa_1" : fitb_santa_1,
    "fitb_santa_2" : fitb_santa_2,
    "fitb_santa_3" : fitb_santa_3,
    "fitb_santa_4" : fitb_santa_4,
    "fitb_santa_5" : fitb_santa_5,
    "fitb_santa_6" : fitb_santa_6,
    "fitb_santa_7" : fitb_santa_7,
    "fitb_santa_8" : fitb_santa_8,
    "fitb_USPTF_1" : fitb_USPTF_1,
    "fitb_USPTF_2" : fitb_USPTF_2,
    "fitb_USPTF_3" : fitb_USPTF_3,
    "fitb_JC" : fitb_JC,
}

MED_PPL_TASKS = {
    "brochure_ctx0_len512": brochure_ctx0_len512,
    "brochure_ctx512_len512": brochure_ctx512_len512,
    "brochure_ctx1024_len512": brochure_ctx1024_len512,
    "brochure_ctx1536_len512": brochure_ctx1536_len512,
    "clinical_guidelines_ctx0_len512": clinical_guidelines_ctx0_len512,
    "clinical_guidelines_ctx512_len512": clinical_guidelines_ctx512_len512,
    "clinical_guidelines_ctx1024_len512": clinical_guidelines_ctx1024_len512,
    "clinical_guidelines_ctx1536_len512": clinical_guidelines_ctx1536_len512,
    "foia_ctx0_len512": foia_ctx0_len512,
    "foia_ctx512_len512": foia_ctx512_len512,
    "foia_ctx1024_len512": foia_ctx1024_len512,
    "foia_ctx1536_len512": foia_ctx1536_len512,
    "tjc_ctx0_len512": tjc_ctx0_len512,
    "tjc_ctx512_len512": tjc_ctx512_len512,
    "tjc_ctx1024_len512": tjc_ctx1024_len512,
    "tjc_ctx1536_len512": tjc_ctx1536_len512,
    "usptf_ctx0_len512": usptf_ctx0_len512,
    "usptf_ctx512_len512": usptf_ctx512_len512,
    "usptf_ctx1024_len512": usptf_ctx1024_len512,
    "usptf_ctx1536_len512": usptf_ctx1536_len512,
}

HIPPOCRATIC_TASKS = {
    
    "chf_eval": chf_eval,
    "dme_eval_1": dme_eval_1,
    "drug_names_eval": drug_names_eval,
    "regulatory_eval": regulatory_eval,
    
    "UMLS_definitions_eval_A": UMLS_definitions_eval_A,
    "UMLS_definitions_eval_B": UMLS_definitions_eval_B,
    
    "all_fill_in_the_blank": all_fill_in_the_blank,
    
    # "drug_names_perplexity": drug_names_perplexity,
    # "drug_questions_perplexity": drug_questions_perplexity,
    
    **TF_8_23_TASKS,
    
    **TASKS_8_23_BATCH,
    
    **TASKS_8_16_BATCH,
    
    **MEDICAL_MCQ_TASKS,
    **FITB_9_11,
    
    **GOLD_SET,
    
    **MED_PPL_TASKS,
}

def create_all_tasks():
    hippo_tasks = {("hippo_" + old_key): value for old_key, value in HIPPOCRATIC_TASKS.items()}
    # print(f"There are {len(hippo_tasks)} HippocraticAI tasks in total")
    return hippo_tasks

rf = RequestFactory()