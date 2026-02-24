# 该文件实现了语音重音检测任务的数据加载器，支持本地和HuggingFace数据集的加载、预处理和分割
from whistress.training.processor import DSProcessor
from datasets import load_from_disk
import os

# 通用数据加载类，支持本地和HuggingFace数据集
class PreprocessedDataLoader():
    """
    Generic data loading class for speech emphasis detection datasets.
    
    Handles dataset preprocessing, loading from disk or HuggingFace,
    adding necessary column indices, and preparing datasets for training.
    
    Attributes:
        preprocessed_dataset_path: Root directory for preprocessed dataset local storage
        model_with_emphasis_head: Model with emphasis detection capability
        hf_token: HuggingFace API token for accessing datasets
        ds_hf_train: HuggingFace dataset name for training (if applicable and the data is also used for training the emphasis detection head)
        ds_hf_eval: HuggingFace dataset name for evaluation (if applicable and the data is also used for evaluation of the emphasis detection head)
        emphasis_indices_column_name: Column name for emphasis labels
        columns_to_remove: Columns to exclude from the dataset
        split_train_val_percentage: Percentage of data to use for validation
    """
    
    def __init__(self, 
                preprocessed_dataset_path, 
                columns_to_remove, 
                model_with_emphasis_head, 
                hf_token=None,
                ds_hf_train=None, 
                ds_hf_eval=None,
                emphasis_indices_column_name="emphasis_indices", 
                transcription_column_name='transcription', 
                split_train_val_percentage=0.02
            ):
        # 初始化各项参数
        self.preprocessed_dataset_path = preprocessed_dataset_path
        self.model_with_emphasis_head = model_with_emphasis_head
        self.hf_token = hf_token
        self.ds_hf_train = ds_hf_train
        self.ds_hf_eval = ds_hf_eval
        self.emphasis_indices_column_name = emphasis_indices_column_name
        self.columns_to_remove = columns_to_remove
        self.transcription_column_name = transcription_column_name
        self.split_train_val_percentage = split_train_val_percentage
        # 加载预处理数据集
        self.dataset = self.load_preproc_datasets(model_with_emphasis_head, 
                                                preprocessed_dataset_path, 
                                                columns_to_remove,
                                                emphasis_indices_column_name, 
                                                transcription_column_name, 
                                                ds_hf_train, 
                                                hf_token)

    def load_preproc_datasets(self, 
                            model_with_emphasis_head, 
                            preprocessed_dataset_path, 
                            columns_to_remove,
                            emphasis_indices_column_name, 
                            transcription_column_name,
                            ds_name_hf, 
                            hf_token):
        """
        Load and preprocess datasets from disk or HuggingFace.
        
        If the dataset exists on disk, loads it directly. Otherwise, downloads and
        processes it using the DSProcessor, then saves it to disk for future use.
        Also adds sentence indices and performs necessary column transformations.
        
        Args:
            model_with_emphasis_head: Model with emphasis detection capability
            columns_to_remove: Columns to exclude from the dataset
            emphasis_indices_column_name: Column name for emphasis labels
            transcription_column_name: Column name for transcription text
            ds_name_hf: HuggingFace dataset name
            hf_token: HuggingFace API token
            
        Returns:
            Processed dataset with unnecessary columns removed
        """
        # 内部函数：调整input_features格式
        def change_input_features(example):
            example['input_features'] = example['input_features'][0]
            return example
        # 内部函数：为每个样本添加句子索引
        def add_sentence_index(row, index_container):
            curr_index = index_container['sentence_index']
            row['sentence_index'] = curr_index
            index_container["sentence_index"] += 1
            return row
        
        # 优先加载本地预处理数据集，否则用DSProcessor处理后保存
        if os.path.exists(preprocessed_dataset_path):
            train_set = load_from_disk(preprocessed_dataset_path)
            return train_set.remove_columns(columns_to_remove)
        else:
            ds_preprocessor = DSProcessor(
                ds_name=ds_name_hf,
                processor=model_with_emphasis_head.processor,
                hyperparameters={"split_train_val_percentage": self.split_train_val_percentage},
                hf_token=hf_token
            )
            train_set = ds_preprocessor.get_train_dataset(emphasis_indices_column_name=emphasis_indices_column_name, 
                                                            transcription_column_name=transcription_column_name,
                                                            model=model_with_emphasis_head,
                                                            columns_to_remove=[])
            index_container = {"sentence_index": 0}
            train_set = train_set.map(add_sentence_index, num_proc=1, load_from_cache_file=False, fn_kwargs={'index_container': index_container})
            train_set = train_set.map(change_input_features, load_from_cache_file=False, num_proc=1)
            if "labels" in train_set['train'].column_names:
                train_set = train_set.rename_column("labels", f"labels_{transcription_column_name}")
            train_set.save_to_disk(os.path.join(preprocessed_dataset_path))            
            return train_set.remove_columns(columns_to_remove)
    
    def split_train_val(self):
        """
        Split dataset into training and validation sets.
        
        Args:
            rows_to_remove: Optional list of row indices to exclude
            
        Returns:
            Tuple of (train_dataset, eval_dataset)
        """
        # 按split_train_val_percentage划分训练和验证集
        if self.split_train_val_percentage == 0.0:
            return self.dataset, None
        dataset_split = self.dataset["train"].train_test_split(
            test_size=self.split_train_val_percentage,
            shuffle=True,
            seed=42,
        )
        return dataset_split["train"], dataset_split["test"]
 

# 针对TinyStress-15K合成数据集的专用加载器
class PreprocessedTinyStress15KLoader(PreprocessedDataLoader):
    """
    Data loader for synthetic GPT-generated data.
    
    This dataset contains TTS-generated speech from GPT-written stories with
    specifically marked emphasis. This synthetic data helps supplement real
    datasets for training emphasis detection models.
    """
    def __init__(self, model_with_emphasis_head, transcription_column_name, save_path):
        import os
        # 指定数据集名称和要移除的列
        # 如果本地预处理数据集存在，使用本地路径；否则使用HuggingFace数据集名称
        # 处理相对路径，确保从正确的目录查找
        if save_path.startswith('./'):
            # 如果是相对路径，需要考虑当前工作目录
            current_dir = os.getcwd()
            # 如果当前目录不包含whistress-en，则需要添加whistress-en路径
            if 'whistress-en' not in current_dir:
                save_path = os.path.join('whistress-en', save_path)
        
        if os.path.exists(save_path):
            ds_hf_train = save_path  # 使用本地预处理数据集路径
        else:
            ds_hf_train = "slprl/TinyStress-15K"  # 使用HuggingFace数据集名称
        columns_to_remove = ['id', 'original_sample_index', 'ssml', 'emphasis_indices', 'metadata', 'word_start_timestamps', 'audio']
        super().__init__(preprocessed_dataset_path=save_path, 
                        columns_to_remove=columns_to_remove,
                        model_with_emphasis_head=model_with_emphasis_head, 
                        emphasis_indices_column_name='emphasis_indices',
                        transcription_column_name=transcription_column_name, 
                        ds_hf_train=ds_hf_train)

    def split_train_val_test(self):
        """
        Split synthetic GPT dataset into train, validation, and test sets.
        
        Uses predefined splits from the HuggingFace dataset, with an optional
        further split of the training set for validation.
        
        Returns:
            train_set, eval_set, test_set
        """
        # 先用父类方法划分train/val，再取test
        if self.split_train_val_percentage == 0.0:
            return self.dataset["train"], None, self.dataset["test"]
        train_set, eval_set = super().split_train_val()
        return train_set, eval_set, self.dataset["test"]
    
    
# 中文数据集加载器
class PreprocessedChineseDatasetLoader(PreprocessedDataLoader):
    """
    中文语音重音检测数据集加载器
    适配Aishell_final_with_char_mapping数据集格式
    """
    
    def __init__(self, model_with_emphasis_head, transcription_column_name, save_path=None):
        # 中文数据集的列配置
        columns_to_remove = ['audio', 'sentence_index', 'map_dict', 'ssml', 'voice_name', 'gender', 'timestamps_mfa']
        
        super().__init__(
            preprocessed_dataset_path=save_path, 
            columns_to_remove=columns_to_remove,
            model_with_emphasis_head=model_with_emphasis_head, 
            emphasis_indices_column_name='emphasis_indices',  # 中文数据集使用emphasis_indices
            transcription_column_name=transcription_column_name, 
            ds_hf_train=save_path,  # 直接使用本地路径
            split_train_val_percentage=0.02
        )

    def split_train_val_test(self):
        """
        分割中文数据集为训练、验证和测试集
        
        Returns:
            train_set, eval_set, test_set
        """
        if self.split_train_val_percentage == 0.0:
            return self.dataset["train"], None, self.dataset["test"]
        
        # 从训练集中分出验证集
        train_set, eval_set = super().split_train_val()
        return train_set, eval_set, self.dataset["test"]


# 工厂函数：根据数据集名称返回对应的数据加载器
# 支持扩展新数据集

def load_data(dataset_path=None, dataset_train=None, dataset_eval=None, 
              transcription_column_name="transcription", emphasis_indices_column_name="emphasis_indices",
              columns_to_remove=None, split_train_val_percentage=0.02, hf_token=None):
    """
    简化的数据加载函数，直接返回训练和验证数据集
    
    Args:
        dataset_path: 数据集路径
        dataset_train: 训练数据集名称
        dataset_eval: 评估数据集名称  
        transcription_column_name: 转录文本列名
        emphasis_indices_column_name: 重音标签列名
        columns_to_remove: 要移除的列
        split_train_val_percentage: 验证集比例
        hf_token: HuggingFace token
        
    Returns:
        train_dataset, eval_dataset
    """
    from datasets import load_from_disk
    
    # 加载数据集
    if dataset_path and os.path.exists(dataset_path):
        print(f"从本地加载数据集: {dataset_path}")
        dataset = load_from_disk(dataset_path)
    else:
        raise ValueError(f"数据集路径不存在: {dataset_path}")
    
    # 获取训练和测试集
    if "train" in dataset:
        train_dataset = dataset["train"]
    else:
        print("⚠️ 数据集中未找到 'train' 划分，设置为空列表")
        train_dataset = []
        
    test_dataset = dataset["test"] if "test" in dataset else None
    
    # 如果是SinoReal_TestOnly，它只有test集，没有train集
    if len(train_dataset) == 0 and test_dataset is not None and len(test_dataset) > 0:
        print("⚠️ 检测到训练集为空，但测试集不为空。这可能是纯测试数据集。")
        print("⚠️ 将使用测试集作为评估集。")
        eval_dataset = test_dataset
        # 为了避免后续代码报错，创建一个空的train_dataset
        train_dataset = test_dataset.select(range(0)) 
    # 从训练集中分出验证集
    elif split_train_val_percentage > 0 and len(train_dataset) > 0:
        split_size = int(len(train_dataset) * split_train_val_percentage)
        if split_size > 0:
            train_val_split = train_dataset.train_test_split(test_size=split_size, seed=42)
            train_dataset = train_val_split["train"]
            eval_dataset = train_val_split["test"]
        else:
            print("⚠️ 训练集太小，无法分割验证集。使用全部数据作为训练集，验证集为空。")
            eval_dataset = train_dataset.select(range(0))
    else:
        eval_dataset = test_dataset if test_dataset is not None else train_dataset.select(range(0))
    
    # 移除不需要的列
    if columns_to_remove:
        available_columns = train_dataset.column_names
        columns_to_remove = [col for col in columns_to_remove if col in available_columns]
        if columns_to_remove:
            train_dataset = train_dataset.remove_columns(columns_to_remove)
            eval_dataset = eval_dataset.remove_columns(columns_to_remove)
    
    print(f"训练集样本数: {len(train_dataset)}")
    print(f"验证集样本数: {len(eval_dataset)}")
    print(f"训练集列名: {train_dataset.column_names}")
    
    return train_dataset, eval_dataset


# 保留原有的工厂函数以兼容旧代码
def load_data_legacy(model_with_emphasis_head, transcription_column_name, dataset_name, save_path=None):
    """
    Factory function to create the appropriate dataset loader.
    
    *Add here any new datasets you want to support.*
    
    Args:
        model_with_emphasis_head: Model with emphasis detection capability
        transcription_column_name: Column name for transcription text
        dataset_name: Name of the dataset to load (e.g., "tinyStress-15K")
        save_path: Path to save or load the preprocessed dataset
        
    Returns:
        Instantiated dataset loader for the specified dataset
        
    Raises:
        ValueError: If the requested dataset is not supported
    """
    dataset = None
    if dataset_name == "tinyStress-15K":
        dataset = PreprocessedTinyStress15KLoader(model_with_emphasis_head, transcription_column_name, save_path=save_path)
    elif dataset_name == "Aishell_final_with_char_mapping":
        dataset = PreprocessedChineseDatasetLoader(model_with_emphasis_head, transcription_column_name, save_path=save_path)
    else:
        raise ValueError(f"Dataset {dataset_name} is not defined in data_loader.py")

    return dataset
