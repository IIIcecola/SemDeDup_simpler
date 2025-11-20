'''
image encoder： openai/clip-vit-large-patch14
video encoder: alibaba-pai/VideoCLIP-XL
'''
import os
import pathlib
import sys
import numpy as np
import torch
import faiss
from PIL import Image
import cv2
from torch.utils.data import Dataset, DataLoader
from torch.nn.functional import normalize
from tqdm import tqdm
import shutil
import logging
from datetime import datetime
from typing import List, Tuple
from transformers import CLIPProcessor, CLIPModel

# -------------------------- 关键：添加模型路径到Python环境（确保能导入modeling和utils） --------------------------
VIDEOCLIP_XL_LOCAL_DIR = "path/to/local/alibaba-pai/VideoCLIP-XL"
sys.path.insert(0, VIDEOCLIP_XL_LOCAL_DIR)
from modeling import VideoCLIP_XL

# -------------------------- 配置参数 --------------------------
class Config:
    def __init__(self):
        # 路径配置
        self.input_dir = "path/to/input"  
        self.output_dir = "path/to/dedup_results" 
        self.temp_dir = os.path.join(self.output_dir, "temp") 
        self.log_dir = os.path.join(self.output_dir, "logs") 
        
        # 本地模型路径
        self.local_clip_image_model = "path/to/local/openai/clip-vit-large-patch14"  
        self.local_video_clip_model = VIDEOCLIP_XL_LOCAL_DIR  
        self.video_clip_weight_path = os.path.join(self.local_video_clip_model, "VideoCLIP-XL.bin") 
        
        # 模型配置（适配VideoCLIP-XL官方要求）
        self.image_embedding_size = 768  # CLIP-ViT-L/14的嵌入维度
        self.video_embedding_size = 768  # VideoCLIP-XL的嵌入维度
        self.batch_size = 64  # 图片batch=64
        self.video_batch_size = 16  # 视频batch=16（VideoCLIP-XL显存占用较高）
        self.video_frame_count = 8  # 官方demo默认8帧
        self.video_frame_interval = None  # 禁用固定间隔，改用官方的"均匀采样到目标帧数"逻辑
        
        # 聚类配置
        self.num_clusters = None  # 自动计算
        self.kmeans_iter = 200
        self.use_cosine = True
        
        # 去重阈值
        self.image_eps = 0.08
        self.video_eps = 0.12

# -------------------------- 工具函数 --------------------------
def setup_logger(config: Config, name: str) -> logging.Logger:
    os.makedirs(config.log_dir, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(
        os.path.join(config.log_dir, f"{name}_{datetime.now().strftime('%Y%m%d')}.log")
    )
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

def get_file_paths(input_dir: str) -> Tuple[List[str], List[str]]:
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp', '.heic')
    video_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm', '.m4v')
    
    image_paths = []
    video_paths = []

    input_path = pathlib.Path(input_dir).absolute()  # 转为绝对路径的 Path 对象
    
    # 遍历图片文件（rglob 是 Path 对象的方法）
    for ext in image_extensions:
        image_paths.extend([str(p) for p in input_path.rglob(f'*{ext}')])
    
    # 遍历视频文件
    for ext in video_extensions:
        video_paths.extend([str(p) for p in input_path.rglob(f'*{ext}')])
    
    return list(sorted(set(image_paths))), list(sorted(set(video_paths)))

# -------------------------- 数据加载 --------------------------
class ImageDataset(Dataset):
    """图片数据集（CLIP官方Processor）"""
    def __init__(self, paths: List[str], processor):
        self.paths = paths
        self.processor = processor

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        try:
            img = Image.open(path).convert("RGB")
            img = self.processor(images=img, return_tensors="pt")["pixel_values"].squeeze(0)
            return img, path, idx
        except Exception as e:
            logging.warning(f"处理图片 {path} 出错: {e}")
            dummy_img = self.processor(images=Image.new('RGB', (224, 224)), return_tensors="pt")["pixel_values"].squeeze(0)
            return dummy_img * 0, path, idx

class VideoDataset(Dataset):
    """视频数据集（适配VideoCLIP-XL官方预处理逻辑）"""
    def __init__(self, paths: List[str], frame_count: int = 8):
        self.paths = paths
        self.frame_count = frame_count  # 官方固定8帧
        # 官方预处理的归一化参数
        self.v_mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
        self.v_std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)

    def __len__(self):
        return len(self.paths)

    def _normalize_frame(self, frame: np.ndarray) -> np.ndarray:
        """官方归一化逻辑：(0-255) → (0-1) → 减均值除标准差"""
        return (frame / 255.0 - self.v_mean) / self.v_std

    def _extract_frames(self, video_path: str) -> List[np.ndarray]:
        """官方帧提取逻辑：读取所有帧 → 均匀采样到目标帧数"""
        cap = cv2.VideoCapture(video_path)
        frames = []
        while cap.isOpened():
            success, frame = cap.read()
            if success:
                frames.append(frame)  # BGR格式（cv2默认）
            else:
                break
        cap.release()

        # 处理空帧或帧数不足的情况
        if len(frames) == 0:
            logging.warning(f"视频 {video_path} 无有效帧，返回占位符")
            return [np.zeros((224, 224, 3), dtype=np.uint8)] * self.frame_count

        # 均匀采样到目标帧数（官方demo逻辑：step = 总帧数//目标帧数）
        step = max(1, len(frames) // self.frame_count)
        sampled_frames = frames[::step][:self.frame_count]  # 按步长采样，取前N帧

        # 不足目标帧数时补最后一帧（官方未处理，补充避免维度错误）
        while len(sampled_frames) < self.frame_count:
            sampled_frames.append(sampled_frames[-1])

        return sampled_frames

    def __getitem__(self, idx):
        path = self.paths[idx]
        try:
            # 1. 提取并采样帧（不变）
            sampled_frames = self._extract_frames(path)  # [8, 224, 224, 3]（8帧，每帧[H,W,C]）
            
            # 2. 预处理每帧（核心：维度转换为 [C, H, W]）
            vid_tube = []
            for fr in sampled_frames:
                fr_rgb = fr[:, :, ::-1]  # BGR→RGB：[224,224,3]
                fr_resized = cv2.resize(fr_rgb, (224, 224))  # 缩放：[224,224,3]
                fr_norm = self._normalize_frame(fr_resized)  # 归一化：[224,224,3]
                fr_transposed = np.transpose(fr_norm, (2, 0, 1))  # 维度转换：[3,224,224]（C在前）
                vid_tube.append(fr_transposed)
            
            # 3. 拼接帧序列：[8, 3, 224, 224]
            vid_tube = np.stack(vid_tube, axis=0)
            # 4. 转换为Tensor（单样本4维：[C, N, H, W]，完全符合模型要求）
            vid_tensor = torch.from_numpy(vid_tube).float()  # 形状：[8, 3, 224, 224]
            return vid_tensor, path, idx
        except Exception as e:
            logging.warning(f"处理视频 {path} 出错: {e}")
            # 占位符形状与正常输出一致：[8, 3, 224, 224]
            dummy_tensor = torch.zeros((3, self.frame_count, 224, 224), dtype=torch.float32)
            return dummy_tensor, path, idx

# -------------------------- 嵌入提取 --------------------------
def extract_embeddings(model, dataloader, dataset_size: int, emb_size: int, temp_dir: str, name: str, is_video: bool = False) -> Tuple[np.memmap, np.memmap]:
    os.makedirs(temp_dir, exist_ok=True)
    emb_path = os.path.join(temp_dir, f"{name}_embeddings.npy")
    paths_path = os.path.join(temp_dir, f"{name}_paths.npy")
    
    emb_memmap = np.memmap(emb_path, dtype='float32', mode='w+', shape=(dataset_size, emb_size))
    paths_memmap = np.memmap(paths_path, dtype=f"<U512", mode='w+', shape=(dataset_size,))
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    torch.cuda.empty_cache()
    
    with torch.no_grad():
        for data_batch, paths_batch, indices in tqdm(dataloader, desc=f"提取{name}嵌入"):
            data_batch = data_batch.to(device, non_blocking=True)
            
            if is_video:
                # VideoCLIP-XL官方嵌入提取逻辑：(batch, 8, 3, 224, 224) → 视频特征
                video_features = model.vision_model.get_vid_features(data_batch)
                embeddings = video_features
            else:
                # CLIP图片嵌入提取
                embeddings = model.get_image_features(pixel_values=data_batch)
            
            # 归一化（与官方demo一致）
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
            # 写入内存映射
            emb_memmap[indices] = embeddings.cpu().numpy()
            paths_memmap[indices] = paths_batch
    
    return emb_memmap, paths_memmap

# -------------------------- 聚类（动态调整） --------------------------
def adjust_cluster_num(sample_size: int) -> int:
    if sample_size < 1000:
        return max(10, sample_size // 5)
    elif sample_size > 2_000_000:
        return 100_000
    else:
        return min(sample_size // 20, 100_000)

def run_clustering(embeddings: np.memmap, config: Config, name: str, sample_size: int) -> Tuple[np.ndarray, np.ndarray]:
    config.num_clusters = adjust_cluster_num(sample_size)
    d = config.image_embedding_size if name == "图片" else config.video_embedding_size
    spherical = config.use_cosine
    
    logging.info(f"{name}聚类配置：样本数={sample_size}，聚类数={config.num_clusters}，迭代次数={config.kmeans_iter}")
    
    kmeans = faiss.Kmeans(
        d, config.num_clusters, 
        niter=config.kmeans_iter,
        verbose=True, seed=42, 
        spherical=spherical,
        gpu=True,
        max_points_per_centroid=1000,
        min_points_per_centroid=1
    )
    
    print(f"开始{name}聚类...（聚类数：{config.num_clusters}）")
    kmeans.train(embeddings)
    
    dists, centroids = kmeans.index.search(embeddings, 1)
    return dists.squeeze(1), centroids.squeeze(1)

# -------------------------- 去重 --------------------------
def deduplicate(
    embeddings: np.memmap, paths: np.memmap, 
    cluster_labels: np.ndarray, eps: float,
    config: Config, logger: logging.Logger, name: str
) -> List[str]:
    clusters = {}
    for idx, label in enumerate(cluster_labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append((idx, embeddings[idx]))
    
    keep_indices = set()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    total_duplicates = 0
    
    for cluster_id, samples in tqdm(clusters.items(), desc=f"{name}去重"):
        if len(samples) <= 1:
            keep_indices.add(samples[0][0])
            continue
        
        indices = [s[0] for s in samples]
        embs = torch.tensor([s[1] for s in samples], device=device, dtype=torch.float32)
        sim_matrix = embs @ embs.T
        sim_matrix.fill_diagonal_(0.0)
        
        # 关键修改：计算每个样本与组内其他样本的平均相似度（而非最大相似度）
        avg_sim = sim_matrix.mean(dim=1).cpu().numpy()
        
        # 逻辑调整：
        # 1. 先保留平均相似度最低的样本（最不重复的核心样本）
        core_idx = np.argmin(avg_sim)
        to_keep = np.zeros(len(samples), dtype=bool)
        to_keep[core_idx] = True  # 强制保留核心样本
        
        # 2. 再保留与核心样本相似度≤阈值的其他样本（非重复样本）
        core_emb = embs[core_idx:core_idx+1]  # 核心样本嵌入
        other_sims = (embs @ core_emb.T).squeeze(1).cpu().numpy()  # 其他样本与核心样本的相似度
        non_dup_mask = other_sims <= (1 - eps)
        to_keep = to_keep | non_dup_mask  # 合并：保留核心样本 + 非重复样本
        
        # 计算重复数
        duplicates_in_cluster = len(samples) - sum(to_keep)
        total_duplicates += duplicates_in_cluster
        
        # 添加保留的索引
        for i, keep in enumerate(to_keep):
            if keep:
                keep_indices.add(indices[i])
    
    keep_paths = [paths[idx] for idx in sorted(keep_indices)]
    logger.info(f"{name}去重完成：原始{len(paths)}个，保留{len(keep_paths)}个，移除重复{total_duplicates}个")
    return keep_paths

# -------------------------- 主处理函数 --------------------------
def process_media(
    paths: List[str], is_image: bool, 
    config: Config, logger: logging.Logger
) -> List[str]:
    if not paths:
        logger.info(f"没有找到{'图片' if is_image else '视频'}文件")
        return []
    
    name = "图片" if is_image else "视频"
    logger.info(f"找到{len(paths)}{name}文件，开始处理...")
    sample_size = len(paths)
    
    # 1. 加载模型（图片用CLIP，视频用VideoCLIP-XL官方模型）
    if is_image:
        model = CLIPModel.from_pretrained(config.local_clip_image_model)
        processor = CLIPProcessor.from_pretrained(config.local_clip_image_model)
        dataset = ImageDataset(paths, processor)
        emb_size = config.image_embedding_size
        batch_size = config.batch_size
    else:
        # 加载VideoCLIP-XL官方模型
        model = VideoCLIP_XL()
        # 加载本地权重（避免在线下载）
        assert os.path.exists(config.video_clip_weight_path), f"VideoCLIP-XL权重文件不存在：{config.video_clip_weight_path}"
        state_dict = torch.load(config.video_clip_weight_path, map_location="cpu")
        model.load_state_dict(state_dict)
        # 视频数据集（无需processor，用官方预处理）
        dataset = VideoDataset(paths, frame_count=config.video_frame_count)
        emb_size = config.video_embedding_size
        batch_size = config.video_batch_size  # 视频单独设置batch size
    
    # 2. 数据加载
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size,
        shuffle=False, 
        num_workers=min(os.cpu_count(), 32),
        pin_memory=True,
        drop_last=False
    )
    
    # 3. 提取嵌入（视频需标记is_video=True）
    embeddings, paths_memmap = extract_embeddings(
        model, dataloader, sample_size,
        emb_size, config.temp_dir,
        "image" if is_image else "video",
        is_video=not is_image
    )
    
    # 4. 聚类
    _, cluster_labels = run_clustering(
        embeddings, config, name, sample_size
    )
    
    # 5. 去重
    eps = config.image_eps if is_image else config.video_eps
    keep_paths = deduplicate(
        embeddings, paths_memmap,
        cluster_labels, eps,
        config, logger, name
    )
    
    # 清理显存
    del model, dataset, dataloader
    if is_image:
        del processor  # 图片才有关processor
    torch.cuda.empty_cache()
    
    return keep_paths

# -------------------------- 文件复制 --------------------------
def copy_kept_files(keep_paths: List[str], output_dir: str, subdir: str):
    dest_dir = os.path.join(output_dir, subdir)
    os.makedirs(dest_dir, exist_ok=True)
    
    for src_path in tqdm(keep_paths, desc=f"复制{subdir}文件"):
        try:
            common_parent = os.path.commonpath(keep_paths)
            rel_path = os.path.relpath(src_path, common_parent)
            dest_path = os.path.join(dest_dir, rel_path)
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            shutil.copy2(src_path, dest_path, follow_symlinks=False)
        except Exception as e:
            logging.warning(f"复制文件 {src_path} 出错: {e}")

# -------------------------- 主函数 --------------------------
def main():
    config = Config()
    os.makedirs(config.output_dir, exist_ok=True)
    logger = setup_logger(config, "semantic_dedup_h100")
    
    # 检查模型路径
    assert os.path.exists(config.local_clip_image_model), f"CLIP模型目录不存在：{config.local_clip_image_model}"
    assert os.path.exists(config.local_video_clip_model), f"VideoCLIP-XL模型目录不存在：{config.local_video_clip_model}"
    assert os.path.exists(config.video_clip_weight_path), f"VideoCLIP-XL权重文件不存在：{config.video_clip_weight_path}"
    logger.info("所有本地模型路径验证通过，开始处理...")
    
    # 获取文件路径
    image_paths, video_paths = get_file_paths(config.input_dir)
    logger.info(f"总图片数: {len(image_paths)}, 总视频数: {len(video_paths)}")
    
    # 处理图片
    image_keep = process_media(image_paths, True, config, logger)
    
    # 处理视频
    video_keep = process_media(video_paths, False, config, logger)
    
    # 保存结果
    if image_keep:
        copy_kept_files(image_keep, config.output_dir, "images")
    if video_keep:
        copy_kept_files(video_keep, config.output_dir, "videos")
    
    # 清理临时文件
    shutil.rmtree(config.temp_dir, ignore_errors=True)
    
    logger.info("所有处理完成！")
    print(f"去重结果已保存到: {config.output_dir}")
    print(f"图片：原始{len(image_paths)}个 → 保留{len(image_keep)}个")
    print(f"视频：原始{len(video_paths)}个 → 保留{len(video_keep)}个")

if __name__ == "__main__":
    main()
