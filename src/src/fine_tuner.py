from transformers import Trainer, TrainingArguments

class LLAMAFineTuner:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def train(self, train_ds, val_ds):
        args = TrainingArguments(
            output_dir="./checkpoints",
            per_device_train_batch_size=1,
            gradient_accumulation_steps=8,
            num_train_epochs=1,
            fp16=True,
            logging_steps=10,
            save_steps=100,
            evaluation_strategy="steps",
            eval_steps=100,
            report_to="none"
        )

        trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=train_ds,
            eval_dataset=val_ds
        )

        trainer.train()
        return trainer
