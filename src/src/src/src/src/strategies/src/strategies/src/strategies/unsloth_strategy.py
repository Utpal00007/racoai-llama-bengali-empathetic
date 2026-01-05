from unsloth import FastLanguageModel
from .base_strategy import FineTuneStrategy

class UnslothStrategy(FineTuneStrategy):
    def prepare_model(self, model):
        model = FastLanguageModel.get_peft_model(
            model,
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            target_modules=["q_proj", "v_proj"],
            use_gradient_checkpointing=True
        )
        return model

