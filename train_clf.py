# 两种模型都适合你的科室分类任务
SUITABLE_MODELS = {
    "structbert": {
        "name": "damo/nlp_structbert_backbone_base_std",
        "type": "编码器模型",
        "适合任务": "分类、理解任务",
        "优点": "双向注意力，理解上下文能力强",
        "在你的任务中": "能更好理解医学描述的上下文关系"
    },
    "roberta": {
        "name": "damo/nlp_roberta_backbone_base_std", 
        "type": "编码器模型",
        "适合任务": "分类、理解任务",
        "优点": "训练稳定，社区支持好",
        "在你的任务中": "成熟的解决方案，可靠性高"
    }
}

import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    Trainer, TrainingArguments, DataCollatorWithPadding
)
import evaluate
import numpy as np
import os
from modelscope import snapshot_download
import swanlab

os.environ["SWANLAB_PROJECT"] = f"medical-classification-ft1"

class MedicalDepartmentExperiment:
    def __init__(self, model_choices):
        self.model_choices = model_choices
        self.results = {}

    def load_and_prepare_data(self, jsonl_path, sample_frac=0.01):
        """加载和准备数据"""
        full_df = pd.read_json(jsonl_path, lines=True)
        df = full_df.sample(frac=sample_frac, random_state=42)

        # 数据清洗
        df = self.clean_data(df)

        # 标签编码
        self.labels = sorted(df["dept"].unique())
        self.label2id = {label: i for i, label in enumerate(self.labels)}
        self.id2label = {i: label for label, i in self.label2id.items()}

        df["label"] = df["dept"].map(self.label2id)
        df["question"] = df["input_text"]

        # 数据集分割
        dataset = Dataset.from_pandas(df[["question", "label"]])
        dataset_split = dataset.train_test_split(test_size=0.2, seed=42)

        return dataset_split['train'], dataset_split['test']

    def clean_data(self, df):
        """数据清洗"""
        # 去除重复
        df = df.drop_duplicates(subset=['input_text', 'dept'])

        # 过滤过短文本
        df = df[df['input_text'].str.len() >= 10]

        # 检查类别平衡
        print("科室分布:")
        print(df['dept'].value_counts())

        return df

    def run_experiments(self, train_dataset, eval_dataset):
        """运行所有模型的实验"""
        for model_id, model_info in self.model_choices.items():
            print(f"\n{'=' * 50}")
            print(f"开始实验: {model_id}")
            print(f"{'=' * 50}")

            result = self.train_single_model(
                model_info['name'],
                model_id,
                train_dataset,
                eval_dataset
            )
            self.results[model_id] = result

        return self.results

    def train_single_model(self, model_name, model_id, train_dataset, eval_dataset):
        """训练单个模型"""
        # 下载模型
        cache_path = f"./models/{model_id}"
        os.makedirs(cache_path, exist_ok=True)
        model_dir = snapshot_download(model_name, cache_dir=cache_path)

        # 加载tokenizer和模型
        tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token else tokenizer.unk_token

        # 数据分词
        def tokenize_function(batch):
            return tokenizer(
                batch["question"],
                padding=False,  # 让collator处理
                truncation=True,
                max_length=128
            )

        tokenized_train = train_dataset.map(tokenize_function, batched=True)
        tokenized_eval = eval_dataset.map(tokenize_function, batched=True)

        # 加载模型
        model = AutoModelForSequenceClassification.from_pretrained(
            model_dir,
            num_labels=len(self.labels),
            id2label=self.id2label,
            label2id=self.label2id,
            ignore_mismatched_sizes=True
        )

        # 训练配置
        training_args = TrainingArguments(
            output_dir=f"./output/{model_id}",
            per_device_train_batch_size=8,
            per_device_eval_batch_size=16,
            num_train_epochs=3,
            learning_rate=2e-5,
            eval_strategy="steps",
            eval_steps=100,
            logging_steps=50,
            save_steps=800,
            warmup_steps=100,
            weight_decay=0.01,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to="swanlab",
            run_name=f"{model_id}-ft1",
        )

        # 评估指标
        accuracy = evaluate.load("accuracy")
        f1 = evaluate.load("f1")

        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            predictions = np.argmax(logits, axis=-1)
            acc = accuracy.compute(predictions=predictions, references=labels)
            f1_macro = f1.compute(predictions=predictions, references=labels, average="macro")
            return {"accuracy": acc["accuracy"], "f1_macro": f1_macro["f1"]}

        # 创建Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_eval,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
        )

        # 训练
        print(f"开始训练 {model_id}...")
        train_result = trainer.train()

        # 评估
        eval_result = trainer.evaluate()

        return {
            'trainer': trainer,
            'model': model,
            'tokenizer': tokenizer,
            'train_result': train_result,
            'eval_result': eval_result
        }


def compare_results(results):
    """比较不同模型的结果"""
    print("\n" + "=" * 60)
    print("模型比较结果")
    print("=" * 60)

    comparison_data = []

    for model_id, result in results.items():
        eval_result = result['eval_result']
        comparison_data.append({
            '模型': model_id,
            '验证损失': eval_result['eval_loss'],
            '准确率': eval_result['eval_accuracy'],
            'F1宏平均': eval_result['eval_f1_macro']
        })

    # 创建比较表格
    comparison_df = pd.DataFrame(comparison_data)
    print(comparison_df.to_string(index=False))

    # 找出最佳模型
    best_by_accuracy = comparison_df.loc[comparison_df['准确率'].idxmax()]
    best_by_f1 = comparison_df.loc[comparison_df['F1宏平均'].idxmax()]

    print(f"\n最佳准确率模型: {best_by_accuracy['模型']} ({best_by_accuracy['准确率']:.4f})")
    print(f"最佳F1分数模型: {best_by_f1['模型']} ({best_by_f1['F1宏平均']:.4f})")

    return comparison_df


# 可视化比较结果
def plot_comparison(comparison_df):
    """绘制模型比较图"""
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # 准确率比较
    models = comparison_df['模型']
    accuracy = comparison_df['准确率']
    f1_scores = comparison_df['F1宏平均']

    ax1.bar(models, accuracy, color=['skyblue', 'lightcoral'])
    ax1.set_title('模型准确率比较')
    ax1.set_ylabel('准确率')
    ax1.set_ylim(0, 1)

    ax2.bar(models, f1_scores, color=['skyblue', 'lightcoral'])
    ax2.set_title('模型F1分数比较')
    ax2.set_ylabel('F1宏平均')
    ax2.set_ylim(0, 1)

    plt.tight_layout()
    plt.show()


# 使用示例
def main():
    # 定义要比较的模型
    model_choices = {
        "structbert": {
            "name": "damo/nlp_structbert_backbone_base_std",
            "description": "达摩院StructBERT，中文理解强"
        },
        "roberta": {
            "name": "damo/nlp_roberta_backbone_base_std",
            "description": "达摩院RoBERTa，稳定可靠"
        }
    }

    # 创建实验
    experiment = MedicalDepartmentExperiment(model_choices)

    # 加载数据
    jsonl_path = "data/full_data_ft1.jsonl"
    train_dataset, eval_dataset = experiment.load_and_prepare_data(jsonl_path, sample_frac=0.01)

    # 运行实验
    results = experiment.run_experiments(train_dataset, eval_dataset)

    # 比较结果
    compare_results(results)


if __name__ == "__main__":
    main()