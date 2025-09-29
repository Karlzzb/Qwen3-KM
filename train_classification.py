import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments,
    DataCollatorWithPadding, EarlyStoppingCallback, AutoConfig
)
import evaluate
import numpy as np
import os
import time
import swanlab
import torch
from modelscope import snapshot_download


class MedicalDepartmentClassifier:
    """医学科室分类器训练框架"""

    # 可用的模型选择
    MODEL_CHOICES = {
        "structbert": {
            "name": "damo/nlp_structbert_backbone_base_std",
            "description": "达摩院StructBERT，中文理解强"
        },
        "roberta": {
            "name": "damo/nlp_roberta_backbone_base_std",
            "description": "达摩院RoBERTa，稳定可靠"
        }
    }

    def __init__(self,
                 model_id="structbert",
                 project_name="medical-dept-classifier",
                 data_path="data/full_data_ft1.jsonl",
                 sample_frac=0.04,
                 test_size=0.2,
                 max_length=128,
                 prompt="你是一个医学专家，你需要根据用户的问题，指定科室"):

        # 初始化配置
        self.model_name = self.MODEL_CHOICES[model_id]["name"]
        self.display_name = model_id
        self.project_name = project_name
        self.data_path = data_path
        self.sample_frac = sample_frac
        self.test_size = test_size
        self.max_length = max_length
        self.prompt = prompt

        # 初始化变量
        self.device = None
        self.tokenizer = None
        self.model = None
        self.trainer = None
        self.train_dataset = None
        self.eval_dataset = None
        self.labels = None
        self.label2id = None
        self.id2label = None

        # 设置环境
        self._setup_environment()

    def _setup_environment(self):
        """设置训练环境"""
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
        os.environ["SWANLAB_PROJECT"] = self.project_name

        # SwanLab配置
        swanlab.config.update({
            "model": f"{self.model_name}",
            "prompt": self.prompt,
            "data_max_length": self.max_length,
            "sample_frac": self.sample_frac,
            "test_size": self.test_size,
        })

        # 选择设备
        self.device, _ = self._select_device_and_dtype()
        print(f"使用设备: {self.device}")

    def _select_device_and_dtype(self):
        """选择设备和数据类型"""
        if torch.cuda.is_available():
            try:
                _ = torch.zeros(1, device="cuda")
                print("INFO: Using CUDA (GPU) for training.")
                return "cuda", torch.float32
            except Exception as e:
                print(f"WARN: CUDA available but failed to use, falling back to CPU. Reason: {e}")

        print("INFO: Using CPU for training.")
        return "cpu", torch.float32

    def clean_and_augment_data(self, df):
        """数据清洗和增强"""
        # 1. 去除重复数据
        initial_size = len(df)
        df = df.drop_duplicates(subset=['input_text', 'dept'])
        print(f"去重: {initial_size} -> {len(df)}")

        # 2. 过滤过短文本
        df = df[df['input_text'].str.len() >= 10]

        # 3. 检查类别平衡
        dept_counts = df['dept'].value_counts()
        print("科室分布:")
        for dept, count in dept_counts.items():
            print(f"  {dept}: {count} 样本")

        # 4. 处理类别不平衡
        min_samples = dept_counts.min()
        balanced_df = pd.DataFrame()

        for dept in df['dept'].unique():
            dept_data = df[df['dept'] == dept]
            if len(dept_data) > min_samples * 2:  # 如果某个科室数据太多
                dept_data = dept_data.sample(n=min_samples * 2, random_state=42)
            balanced_df = pd.concat([balanced_df, dept_data])

        return balanced_df

    def load_data(self):
        """加载和预处理数据"""
        script_path = os.path.dirname(os.path.abspath(__file__))
        jsonl_path = os.path.join(script_path, self.data_path)

        if not os.path.exists(jsonl_path):
            raise ValueError(f"数据集地址：{jsonl_path} 不存在")

        # 1. 加载数据
        full_df = pd.read_json(jsonl_path, lines=True)
        print(f"原始数据大小: {len(full_df)}")

        # 2. 采样数据
        df = full_df.sample(frac=self.sample_frac, random_state=42)
        print(f"采样后数据大小: {len(df)}")

        # 3. 数据清洗（可选）
        # df = self.clean_and_augment_data(df)

        # 4. 检查数据列
        print(f"数据列: {df.columns.tolist()}")

        # 5. 标签映射
        self.labels = sorted(df["dept"].unique())
        self.label2id = {label: i for i, label in enumerate(self.labels)}
        self.id2label = {i: label for label, i in self.label2id.items()}
        df["label"] = df["dept"].map(self.label2id)
        df["question"] = df["input_text"]

        print(f"科室数量: {len(self.labels)}")
        print(f"科室列表: {self.labels}")

        # 6. 转换为Dataset并分割
        dataset = Dataset.from_pandas(df[["question", "label"]])
        dataset_split = dataset.train_test_split(test_size=self.test_size, seed=42)

        self.train_dataset = dataset_split['train']
        self.eval_dataset = dataset_split['test']

        print(f"训练集大小: {len(self.train_dataset)}")
        print(f"验证集大小: {len(self.eval_dataset)}")

        return self.train_dataset, self.eval_dataset

    def setup_tokenizer_and_model(self):
        """设置tokenizer和模型"""
        script_path = os.path.dirname(os.path.abspath(__file__))
        cache_path = os.path.join(script_path, "models")
        os.makedirs(cache_path, exist_ok=True)

        # 1. 下载模型
        print(f"下载模型: {self.model_name}")
        model_dir = snapshot_download(self.model_name, cache_dir=cache_path)

        # 2. 加载tokenizer
        print("加载tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)

        # 设置pad_token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token if self.tokenizer.eos_token else self.tokenizer.unk_token

        # 3. 数据分词
        def tokenize_function(batch):
            return self.tokenizer(
                batch["question"],
                padding="max_length",
                truncation=True,
                max_length=self.max_length
            )

        print("对数据集进行分词...")
        self.tokenized_train = self.train_dataset.map(tokenize_function, batched=True)
        self.tokenized_eval = self.eval_dataset.map(tokenize_function, batched=True)

        # 4. 数据整理器
        self.collator = DataCollatorWithPadding(
            tokenizer=self.tokenizer,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        # 5. 加载模型
        print("加载模型...")
        config = AutoConfig.from_pretrained(
            model_dir,
            num_labels=len(self.labels),
            id2label=self.id2label,
            label2id=self.label2id,
            classifier_dropout=0.2,
            hidden_dropout_prob=0.2,
            attention_probs_dropout_prob=0.1,
        )

        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_dir,
            config=config,
            ignore_mismatched_sizes=True
        )

        self.model.to(self.device)
        print(f"模型已加载到设备: {self.device}")

    def compute_metrics(self, eval_pred):
        """计算评估指标"""
        accuracy = evaluate.load("accuracy")
        f1 = evaluate.load("f1")

        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)

        acc = accuracy.compute(predictions=predictions, references=labels)
        f1_macro = f1.compute(predictions=predictions, references=labels, average="macro")

        return {
            "accuracy": acc["accuracy"],
            "f1_macro": f1_macro["f1"]
        }

    def setup_training(self, training_args=None):
        """设置训练参数和Trainer"""
        if training_args is None:
            training_args = TrainingArguments(
                output_dir=f"output/{self.display_name}-ft1",
                per_device_train_batch_size=16,
                per_device_eval_batch_size=32,
                gradient_accumulation_steps=1,
                eval_strategy="steps",
                eval_steps=200,
                logging_steps=10,
                num_train_epochs=6,
                save_steps=2000,
                learning_rate=5e-5,
                save_on_each_node=True,
                gradient_checkpointing=False,
                report_to="swanlab",
                run_name=f"{self.display_name}-ft1",
                warmup_steps=100,
                weight_decay=0.01,
                max_grad_norm=1.0,
                lr_scheduler_type="cosine",
                load_best_model_at_end=True,
                metric_for_best_model="eval_loss",
                greater_is_better=False,
            )

        # 创建Trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.tokenized_train,
            eval_dataset=self.tokenized_eval,
            tokenizer=self.tokenizer,
            compute_metrics=self.compute_metrics,
            data_collator=self.collator,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=6)],
        )

        return self.trainer

    def train(self):
        """开始训练"""
        if self.trainer is None:
            raise ValueError("请先调用 setup_training() 方法设置训练参数")

        print("开始训练...")
        train_result = self.trainer.train()

        # 保存最终模型
        self.trainer.save_model(f"output/{self.display_name}-final")

        return train_result

    def evaluate(self):
        """评估模型"""
        if self.trainer is None:
            raise ValueError("请先训练模型")

        eval_result = self.trainer.evaluate()
        return eval_result

    def predict(self, texts):
        """预测科室"""
        if isinstance(texts, str):
            texts = [texts]

        results = []
        for text in texts:
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=self.max_length
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            self.model.eval()
            with torch.no_grad():
                outputs = self.model(**inputs)
                pred = outputs.logits.argmax(dim=-1).item()

            results.append({
                "text": text,
                "predicted_department": self.id2label[pred],
                "confidence": torch.softmax(outputs.logits, dim=1).max().item()
            })

        return results

    def run_pipeline(self, custom_training_args=None):
        """运行完整的训练流水线"""
        print(f"开始医学科室分类器训练流水线")
        print(f"模型: {self.display_name} ({self.model_name})")
        print(f"项目: {self.project_name}")
        print("=" * 60)

        # 1. 加载数据
        self.load_data()

        # 2. 设置模型和tokenizer
        self.setup_tokenizer_and_model()

        # 3. 设置训练
        self.setup_training(custom_training_args)

        # 4. 训练模型
        train_result = self.train()

        # 5. 评估模型
        eval_result = self.evaluate()

        # 6. 测试预测
        test_texts = [
            "最近咳嗽厉害，晚上睡不着",
            "头痛恶心应该看什么科",
            "皮肤长红疹很痒",
            "胸口闷痛呼吸困难"
        ]

        print("\n测试预测结果:")
        predictions = self.predict(test_texts)
        for pred in predictions:
            print(f"问题: {pred['text']}")
            print(f"预测科室: {pred['predicted_department']} (置信度: {pred['confidence']:.3f})")
            print("-" * 50)

        return {
            "train_result": train_result,
            "eval_result": eval_result,
            "predictions": predictions
        }


# 使用示例
# def main():
    # 方法1: 快速使用默认配置
    # classifier = MedicalDepartmentClassifier(
    #     model_name="structbert",  # 或 "roberta"
    #     project_name="medical-dept-classifier",
    #     sample_frac=0.04,
    #     test_size=0.2
    # )

    # 运行完整流水线
    # results = classifier.run_pipeline()

    # 方法2: 自定义训练参数
    # custom_args = TrainingArguments(
    #     output_dir="output/custom-training",
    #     per_device_train_batch_size=8,
    #     num_train_epochs=10,
    #     learning_rate=3e-5,
    #     # ... 其他参数
    # )
    # results = classifier.run_pipeline(custom_args)


# 多模型比较示例
def compare_models():
    """比较不同模型的表现"""
    models_to_compare = ["structbert", "roberta"]
    results = {}

    for model_id in models_to_compare:
        print(f"\n{'=' * 60}")
        print(f"训练模型: {model_id}")
        print(f"{'=' * 60}")

        run = swanlab.init(
            project=f"medical-dept-classifier",
            experiment_name=f"{model_id}-ft1",
        )

        classifier = MedicalDepartmentClassifier(
            model_id=model_id,
            project_name=f"medical-dept-classifier",
            sample_frac=0.1,  # 使用小样本快速比较
            test_size=0.2
        )

        result = classifier.run_pipeline()
        results[model_id] = result["eval_result"]
        # 等待一下，确保SwanLab记录完成
        run.finish()
        time.sleep(4)

    # 比较结果
    print("\n模型比较结果:")
    print("=" * 60)
    for model_name, result in results.items():
        print(
            f"{model_name}: 准确率={result['eval_accuracy']:.4f}, F1={result['eval_f1_macro']:.4f}, 损失={result['eval_loss']:.4f}")

    return results


if __name__ == "__main__":
    # 运行单个模型训练
    # main()

    # 或者运行模型比较
    compare_models()