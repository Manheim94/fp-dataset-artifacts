
import textattack

from transformers import AutoModel, AutoTokenizer
from textattack.models.wrappers import HuggingFaceModelWrapper
from textattack.datasets import Dataset
from transformers import ElectraForSequenceClassification
from textattack.constraints.pre_transformation import RepeatModification, StopwordModification
from textattack.constraints.semantics.sentence_encoders import UniversalSentenceEncoder
from textattack.constraints.grammaticality import PartOfSpeech
from textattack.transformations import WordSwapEmbedding
from textattack.search_methods import GreedySearch
from textattack.goal_functions import UntargetedClassification
from textattack import Attack
import pandas as pd

def apply_attack_to_premise(dataset, attack):
    transformed_premise_data = []
    for original_premise, original_hypothesis, label in dataset:
        attack_results = attack.attack(original_premise, label)

        if hasattr(attack_results, '__iter__'):
            for result in attack_results:
                if isinstance(result, textattack.attack_results.SuccessfulAttackResult):
                    transformed_premise_data.append((original_premise, result.perturbed_text(), original_hypothesis, label))
        else:
            if isinstance(attack_results, textattack.attack_results.SuccessfulAttackResult):
                transformed_premise_data.append((original_premise, attack_results.perturbed_text(), original_hypothesis, label))
    return transformed_premise_data

def apply_attack_to_hypothesis(dataset, attack):
    transformed_hypothesis_data = []
    for original_premise, original_hypothesis, label in dataset:
        attack_results = attack.attack(original_hypothesis, label)

        if hasattr(attack_results, '__iter__'):
            for result in attack_results:
                if isinstance(result, textattack.attack_results.SuccessfulAttackResult):
                    transformed_hypothesis_data.append((original_premise, original_hypothesis, result.perturbed_text(), label))
        else:
            if isinstance(attack_results, textattack.attack_results.SuccessfulAttackResult):
                transformed_hypothesis_data.append((original_premise, original_hypothesis, attack_results.perturbed_text(), label))
    return transformed_hypothesis_data


class CustomRecipe(Attack):
    def __init__(self, model):
        transformation = WordSwapEmbedding(max_candidates=10)
        constraints = [
            RepeatModification(),
            StopwordModification(),
            UniversalSentenceEncoder(threshold=0.8),
            PartOfSpeech()
        ]

        search_method = GreedySearch()
        goal_function = UntargetedClassification(model)
        super().__init__(goal_function, constraints, transformation, search_method)

# Define a class for your custom dataset
class CustomTextAttackDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def main():
    print('start')
    file_path = 'snlitraindata.csv'
    df_ad = pd.read_csv(file_path)


    model_path = '/home/menghao_yang/workspace/fp-dataset-artifacts/trained_model/checkpoint-pretrain'
    model = ElectraForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)


    model.eval()

    import torch

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(device)
    model.to(device)

    # Wrap the model for TextAttack
    model_wrapper = HuggingFaceModelWrapper(model, tokenizer)

    # Initialize your TextAttack custom recipe with the model wrapper
    attack = CustomRecipe(model_wrapper)

    num_examples = 10

    combined_data = [(row['premise'], row['hypothesis'], row['label']) for _, row in df_ad.iterrows()]

    for ite in range(10000):
        print(ite)
        limited_combined_data = combined_data[ite*num_examples:(ite+1)*num_examples]
        custom_dataset = CustomTextAttackDataset(limited_combined_data)

        # Apply the attack to the premise
        transformed_premise_data = apply_attack_to_premise(custom_dataset, attack)
        import json

        # Structure transformed_premise_data for JSON
        structured_premise_data = [
            {
                'original_premise': orig_premise,
                'adversarial_premise': adv_premise,
                'original_hypothesis': orig_hypothesis,
                'label': label
            }
            for orig_premise, adv_premise, orig_hypothesis, label in transformed_premise_data
        ]

        # Save structured_premise_data to a JSON file
        with open('output.jsonl', 'a') as outfile:
            for entry in structured_premise_data:
                json.dump(entry, outfile)
                outfile.write('\n')


        

if __name__ == "__main__":
    main()