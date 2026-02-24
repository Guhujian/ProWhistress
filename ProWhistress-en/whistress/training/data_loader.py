# data_loader.py 该文件实现了语音重音检测任务的数据加载器，支持本地和HuggingFace数据集的加载、预处理和分割
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
            def safe_remove(ds_any, cols):
                # 支持 DatasetDict 或 单个 Dataset
                try:
                    is_dict = hasattr(ds_any, 'keys') and callable(ds_any.keys)
                except Exception:
                    is_dict = False
                if is_dict:
                    # 从任意一个split拿列名（优先train，其次第一个split）
                    split_names = list(ds_any.keys())
                    ref_split = 'train' if 'train' in ds_any else split_names[0]
                    existing = set(ds_any[ref_split].column_names)
                else:
                    existing = set(ds_any.column_names)
                cols_keep = [c for c in cols if c in existing]
                return ds_any.remove_columns(column_names=cols_keep)
            return safe_remove(train_set, columns_to_remove)
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
            # 仅移除存在的列，支持 DatasetDict 或 单个 Dataset
            def safe_remove(ds_any, cols):
                try:
                    is_dict = hasattr(ds_any, 'keys') and callable(ds_any.keys)
                except Exception:
                    is_dict = False
                if is_dict:
                    split_names = list(ds_any.keys())
                    ref_split = 'train' if 'train' in ds_any else split_names[0]
                    existing = set(ds_any[ref_split].column_names)
                else:
                    existing = set(ds_any.column_names)
                cols_keep = [c for c in cols if c in existing]
                return ds_any.remove_columns(column_names=cols_keep)
            return safe_remove(train_set, columns_to_remove)
    
    def split_train_val(self):
        ds = self.dataset
        try:
            is_dict = hasattr(ds, 'keys') and callable(ds.keys)
        except Exception:
            is_dict = False
        if is_dict:
            keys = list(ds.keys())
            base_key = 'train' if 'train' in ds else (keys[0] if keys else None)
            if base_key is None:
                return ds, None
            base = ds[base_key]
        else:
            base = ds
        if self.split_train_val_percentage == 0.0:
            return base, None
        split = base.train_test_split(
            test_size=self.split_train_val_percentage,
            shuffle=True,
            seed=42,
        )
        return split["train"], split["test"]
 

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
            # 如果当前目录不包含ProWhistress，则需要添加ProWhistress路径
            if 'ProWhistress' not in current_dir:
                save_path = os.path.join('ProWhistress', save_path)
        
        if os.path.exists(save_path):
            ds_hf_train = save_path  # 使用本地预处理数据集路径
        else:
            ds_hf_train = "slprl/TinyStress-15K"  # 使用HuggingFace数据集名称
        columns_to_remove = ['id', 'original_sample_index', 'ssml', 'emphasis_indices', 'metadata', 'audio']
        super().__init__(preprocessed_dataset_path=save_path, 
                        columns_to_remove=columns_to_remove,
                        model_with_emphasis_head=model_with_emphasis_head, 
                        emphasis_indices_column_name='emphasis_indices',
                        transcription_column_name=transcription_column_name, 
                        ds_hf_train=ds_hf_train)

    def split_train_val_test(self):
        ds = self.dataset
        try:
            is_dict = hasattr(ds, 'keys') and callable(ds.keys)
        except Exception:
            is_dict = False
        train_set, eval_set = super().split_train_val()
        if is_dict:
            if 'test' in ds:
                test_set = ds['test']
            elif 'validation' in ds:
                test_set = ds['validation']
            else:
                test_set = eval_set if eval_set is not None else train_set
        else:
            test_set = eval_set if eval_set is not None else train_set
        return train_set, eval_set, test_set
    
    
# 工厂函数：根据数据集名称返回对应的数据加载器
# 支持扩展新数据集

def load_data(model_with_emphasis_head, transcription_column_name, dataset_name, save_path=None):
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
    else:
        raise ValueError(f"Dataset {dataset_name} is not defined in data_loader.py")

    return dataset
