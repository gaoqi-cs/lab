import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import re
from utils.common_utils import default_logger
from pathlib import Path
from config.config import settings
from safetensors.torch import load_file
import torchvision.models as models
import threading
import torchvision.transforms as transforms
import gc
import json
from torchvision.models import ResNet50_Weights
from torchvision import datasets
import numpy as np
from torch.utils.data import WeightedRandomSampler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader,SubsetRandomSampler
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
import copy
from io import BytesIO
from PIL import Image
import io

class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing:float=0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, pred, target):
        #计算交叉熵损失
        n_classes=pred.size(1)
        #对标签进行one热编码
        one_hot=torch.zeros_like(pred).scatter(1, target.unsqueeze(1), 1)
        #对标签进行平滑处理
        one_hot = one_hot * (1 - self.smoothing) + self.smoothing / n_classes
        #对预测进行log_softmax
        log_prob = F.log_softmax(pred, dim=1)
        # 计算损失
        loss=-(one_hot * log_prob).sum(dim=1).mean()
        return loss
    
class CosineClassifier(nn.Module):
    
    
    #余弦分类器（适用于当前学期，实际工作可以直接使用线性分类器）
    def __init__(self,in_features:int,num_classes:int,scale:float=1.0):
        super().__init__()
        self.in_features=in_features
        self.num_classes=num_classes
        self.scale=nn.Parameter(torch.tensor(scale,dtype=torch.float))
        self.weight=nn.Parameter(torch.empty(in_features,num_classes))
        #让参数生效
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight,a=math.sqrt(5.0))
        
    def forward(self,x):
        #输入的权重进行归一化
        # weight 的形状是 [特征维度, 类别数]，应沿特征维度归一化每个类别向量。
        # 原先按 dim=1 归一化会把每个特征行归一化，导致类别方向失真，
        # 训练后很容易所有图片都偏向同一个类别。
        weight_norm=F.normalize(self.weight,dim=0,p=2)
        x_norm=F.normalize(x,dim=1,p=2)
        #计算余弦相似度
        # cos_sim=F.linear(x_norm,weight_norm)
        cos_sim=x_norm@weight_norm
        return cos_sim*self.scale
    
def adapt_timn_resnet_state_dict(state_dict):
    #将微软的模型映射为原始模型状态
    new_state_dict = {}
    for k,v in state_dict.items():
        if k.startswith("resnet.embedder.embedder.convolution"):
            #卷积层
            suffix=k.split(".")[-1]
            new_state_dict[f"conv1.{suffix}"]=v
            continue
        elif k.startswith("resnet.embedder.embedder.normalization"):
            #归一化层
            suffix=k.split(".")[-1]
            new_state_dict[f"bn1.{suffix}"]=v
            continue
        elif k.startswith("classifier"):
            #全连接层
            parts=k.split(".")
            if len(parts)>=3 and parts[1]=="1":
                new_state_dict[f"fc.{parts[-1]}"]=v
            continue 
        #提取编码器中卷积层和归一化层的权重，平移，滑动均值，滑动方差
           
        pattern=re.compile(r"""xxx""",re.VERBOSE)
        pattern = re.compile(r"""^resnet\.encoder\.stages\.(\d+)
                             \.layers\.(\d+)\.layer\.(\d+)
                             \.(convolution|normalization)
                             \.(weight|bias|running_mean|running_var)$""", 
                             re.VERBOSE)#$结束符
        m=pattern.match(k)
        if m:
            stage=int(m.group(1))
            block=int(m.group(2))
            layer_idx=int(m.group(3))
            kind=m.group(4)
            suffix=m.group(5)
            layer_name=f"layer{stage+1}"
            if kind=="convolution":
                new_key=f"{layer_name}.{block}.conv{layer_idx+1}.{suffix}"
            else:
                new_key=f"{layer_name}.{block}.bn{layer_idx+1}.{suffix}"
            new_state_dict[new_key]=v
            continue
    
        # 提取捷径卷积层和归一化层的权重，平移，滑动均值，滑动方差 
        shortcut_pattern=re.compile(r"^resnet\.encoder\.stages\.(\d+)\.layers\.(\d+)\.shortcut\.(convolution|normalization)\.(weight|bias|running_mean|running_var)$")
        m=shortcut_pattern.match(k)
        if m:
            stage=int(m.group(1))
            block=int(m.group(2))
           # layer_idx=int(m.group(3))
            kind=m.group(3)
            suffix=m.group(4)
            layer_name=f"layer{stage+1}"
            if kind=="convolution":
               new_key=f"{layer_name}.{block}.downsample.0.{suffix}"
            else:
               new_key=f"{layer_name}.{block}.downsample.1.{suffix}"
            new_state_dict[new_key]=v
            continue
    default_logger.info(f"适配后的STATE_DICT的键的数量:{len(new_state_dict)}")       
    return new_state_dict

def load_resnet50_from_local_safetensors():
    #从本地加载ResNet-50模型
    #model_path=Path(settings.RESNET50_MODEL_PATH)
    try:
        raw_state_dict=load_file(f"{settings.RESNET50_MODEL_PATH}/model.safetensors")
        default_logger.info(f"原始STATE_DICT的键的数量:{len(raw_state_dict)}")
    except Exception as e:
        raise RuntimeError(f"加载原始ResNet-50模型失败:{e}")   
    
    sample_keys=list(raw_state_dict.keys())
    # if any(k.startswith("resnet.encoder.stages") for k in sample_keys):
    #     default_logger.info(f"非标准key")
    # else:
    #     default_logger.info(f"标准key") 
    #转换为标准key
    adapted_state_dict=adapt_timn_resnet_state_dict(raw_state_dict)
    #构建标准模型
    model=models.resnet50(weights=None)
    #处理旧权重
    old_fc_weight=None
    if "fc.weight" in adapted_state_dict:
        old_fc_weight=adapted_state_dict["fc.weight"]
        del adapted_state_dict["fc.weight"]
    if "fc.bias" in adapted_state_dict:
        del adapted_state_dict["fc.bias"]
        
    #处理骨架权重
    missing_keys,unexpected_keys=model.load_state_dict(adapted_state_dict,strict=False)
    #标准模型必有的key
    essential_keys=['conv1.weight','bn1.weight','layer1.0.conv1.weight']
    # 查找必备但缺少的key
    missing_essential=[k for k in essential_keys if k  in missing_keys]
    
    if missing_essential:
        default_logger.warning(f"缺少必备的权重：{missing_essential}")
        raise RuntimeError(f"缺少必备的权重：{missing_essential}")
    # 替换fc层
    in_features=model.fc.in_features 
    num_classes=1000
    model.fc=nn.Linear(in_features, num_classes)
    if old_fc_weight is not None and old_fc_weight.shape == model.fc.weight.shape:
        model.fc.weight.data=old_fc_weight
        default_logger.info(f"旧的fc层权重已加载")
    else:
        default_logger.info(f"没有旧的fc层权重")
    return model


class ClassifyService:
    #图像分类服务（单例模式）
    _instance=None
    _lock=threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    
        return cls._instance
    
    def __init__(self):
        if hasattr(self,"_initialized") :
            #如果已经初始化过，则直接返回
            return
        #如果没有初始化，做初始化标记
        self._initialized = True
        self.device=torch.device(settings.DEVICE)
        self.model=None
        # 当前模型支持的识别列表
        self.class_names=[]
        # 标识模型是否载入成功
        self.model_ready=False
        # 图像归一化处理
        self.transforms = transforms.Compose([
            transforms.Resize((224,224)),
            #归一化：0-1
            transforms.ToTensor(),
            #归一化参数平均差/标准差
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self._load_model()
        
    def unload_model(self):
        #卸载模型
        if self.model is not None:
            #清空内存
            del self.model
            self.model = None
            gc.collect()
            if torch.cuda.is_available(): 
                #清空显存
                torch.cuda.empty_cache()
            default_logger.info(f"模型已卸载")
        else:
            default_logger.info(f"模型未加载")
    def _load_model(self):
        #加载模型
        self.unload_model()
        pth_path=Path(settings.RESNET50_FINETUNED_PTH_PATH)
        class_path=Path(settings.RESNET50_FINETUNED_CLASSNAMES_PATH)
        #尝试加载微调模型
        if pth_path.exists() and class_path.exists():
            try:
                with open(class_path, 'r',encoding='utf-8') as f:
                    class_names=json.load(f)
                num_classes=len(class_names)
                model=models.resnet50(weights=None)
                #提取分类的特征
                in_features=model.fc.in_features
                #替换分类器
                model.fc=nn.Linear(in_features,num_classes)
                #提取字典
                state_dict=torch.load(pth_path,map_location=self.device)
                # 保存字典到标准模型
                model.load_state_dict(state_dict)
                # 设置模型放入位置
                model.to(self.device)
                model.eval()
                
                self.model=model
                self.class_names=class_names
                # 加载完毕
                self.model_ready=True
                default_logger.info(f"模型已加载，支持{num_classes}个分类")
                return
            
                
            except Exception as e:
                default_logger.error(f"模型加载失败：{e}")
        default_logger.error(f"开始预训练模型")
        try:
            model=load_resnet50_from_local_safetensors()
            self.model=model
            self.class_names=self._get_imagenet_classes()
            # 加载完毕
            self.model_ready = True
            default_logger.info(f"预训练模型已加载，支持的类别有{len(self.class_names)}个分类")
            
        except Exception as e:
            default_logger.error(f"加载预训练模型失败：{e}")
            self.model = None
            self.class_names = []
            self.model_ready = False
    
    def _get_imagenet_classes(self):
        #获取ImageNet分类列表
        try:
            weights=ResNet50_Weights.IMAGENET_V1()
            return weights.meta["categories"]
        except Exception as e:
            default_logger.warning(f"获取ImageNet分类列表失败：{e}")
            return [f"class_{i}" for i in range(1000)]
     
    #  def _load_class_names_from_data(self):
    
    def _expand_fc_layer(self,model,old_num_classes,new_num_classes):
        #扩展fc层 
        fc=model.fc
        in_features=fc.in_features
        
        if old_num_classes==0:
            #如果旧分类数为0，则直接创建新分类器
            new_fc=nn.Linear(in_features,new_num_classes)
            # 保存新分类器到标准模型
            model.fc=new_fc
            default_logger.info(f"fc层已扩展为{new_num_classes}个分类")
            return model
        # 继续微调
        #保存旧权重
        old_weight=fc.weight.data
        new_fc=nn.Linear(in_features,new_num_classes)
        
        if new_num_classes>old_num_classes:
            # 如果新分类数大于旧分类数，则只复制旧分类数的权重
            new_fc.weight.data[:old_num_classes]=old_weight[:old_num_classes]
            new_fc.bias.data[:old_num_classes]=fc.bias.data[:old_num_classes]
            # 填充新分类的权重
            nn.init.kaiming_uniform_(new_fc.weight.data[old_num_classes:,:],a=math.sqrt(5))
            default_logger.info(f"fc层已扩展为{new_num_classes}个分类")
        else:
            new_fc.weight.data[:,:]=old_weight[:new_num_classes]
            new_fc.bias.data[:]=fc.bias.data[:new_num_classes]
            default_logger.info(f"fc层已截断为{new_num_classes}个分类")
            
        model.fc=new_fc
        return model
    
    
    def finetune(self,data_dir:Path, epoch=settings.FULL_EPOCHS,stop_check=None):
        # 微调模型
        default_logger.info(f"开始微调模型，支持{epoch}个epoch")
        try:
            # 定义微调参数
            # 对应的训练集
            train_transform=transforms.Compose([
                transforms.RandomResizedCrop(224,scale=(0.7,1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.3,contrast=0.3,saturation=0.3,hue=0.1),
                transforms.RandomRotation(degrees=20),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            # 对应的验证集
            val_transform=transforms.Compose([
                transforms.Resize((224,224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            # 分别创建训练集和验证集，避免修改 Subset 共用的 transform。
            train_source = datasets.ImageFolder(root=str(data_dir), transform=train_transform)
            val_source = datasets.ImageFolder(root=str(data_dir), transform=val_transform)
            # 从数据集中提取分类
            class_names = train_source.classes
            # 分类的个数
            num_class = len(class_names)
            if num_class == 0:
                default_logger.error(f"数据集{data_dir}中没有分类")
                return False
            
            class_indices = [[] for _ in range(num_class)]
            for index, target in enumerate(train_source.targets):
                class_indices[target].append(index)
            insufficient = [class_names[index] for index, items in enumerate(class_indices) if len(items) < 2]
            if insufficient:
                raise ValueError(f"每个类别至少需要 2 张有效图片，样本不足：{','.join(insufficient)}")

            # Per-class split keeps both train and validation sets non-empty.
            rng = np.random.default_rng(42)
            train_indices, val_indices = [], []
            for items in class_indices:
                rng.shuffle(items)
                val_count = max(1, int(round(len(items) * 0.2)))
                val_count = min(val_count, len(items) - 1)
                val_indices.extend(items[:val_count])
                train_indices.extend(items[val_count:])
            
            val_dataset = torch.utils.data.Subset(val_source, val_indices)
            train_dataset = torch.utils.data.Subset(train_source, train_indices)
            if not train_indices or not val_indices:
                raise ValueError("训练集或验证集为空，请为每个类别补充至少 2 张图片")
            #类别均衡采样
            train_labels = [train_source.targets[i] for i in train_indices]
            # 就算某一个样本抽出0个，也要保证该样本出现在列表，但数量为0
            class_counts=np.bincount(train_labels,minlength=num_class)
            # 检查个数为0个样本，将其改为1
            class_counts[class_counts == 0] = 1
            # 类型权重
            class_weight=1.0/torch.tensor(class_counts,dtype=torch.float)
            sample_weights=class_weight[train_labels]
            #权重随机样本
            sampler=WeightedRandomSampler(sample_weights,len(sample_weights),replacement=True)
            batch_size=settings.BATCH_SIZE
            # windows下num_workers参数需要0，否则容易引起错误导致失败
            #即使使用了WeightRandomSampler也只能提高低数量样本的抽出概率
            #并不能完全解决某样本0抽出的问题
            train_loader=DataLoader(train_dataset,batch_size=batch_size,sampler=sampler,shuffle=None,num_workers=0)
            val_loader=DataLoader(val_dataset,batch_size=batch_size,shuffle=False,num_workers=0)
                       
        except Exception as e:
            default_logger.error(f"数据集加载出错：{e}")
            return False
        pth_path=Path(settings.RESNET50_FINETUNED_PTH_PATH)
        class_path=Path(settings.RESNET50_FINETUNED_CLASSNAMES_PATH)
        old_num_class=0
        model=None
           
        #初始化模型
        # 读取上次微调后的模型（会引发灾难遗忘，可以省略的）
        
        # if not reset_model and pth_path.exists() and class_path.exists():
        #         #继续微调
        #     try:
        #         with open(class_path,'r',encoding='utf-8') as f:
        #             old_class_names=json.load(f) 
        #         old_num_class=len(old_class_names)
                
        #         model=models.resnet50(weight=None)
        #         in_feature=model.fc.in_features
        #         model.fc=CosineClassifier(in_feature,old_num_class)
        #         state_dict=torch.load(pth_path,map_location=self.device)
        #         model.load_state_dict(state_dict)
        #         default_logger.info(f"模型加载成功，分类数为{old_num_class}")
        #     except Exception as e:
        #         default_logger.error(f"模型加载出错：{e}")
        #         model=None
            
        if model is None:
            try:
                # 如果获取模型失败，则从本地加载模型
                model=load_resnet50_from_local_safetensors()
                old_num_class=0
                default_logger.info(f"原始模型加载成功，分类数为{old_num_class}")
            except Exception as e:
                default_logger.error(f"原始模型加载错误：{e}")
                raise RuntimeError(f"原始模型加载出错：{e}")
        
        #调整fc层    
        if old_num_class!=num_class:
            # 调整fc层（增加/缩减）
            model=self._expand_fc_layer(model,old_num_class,num_class)
            default_logger.info(f"调整fc层成功，分类数为{num_class}")
        else:
            default_logger.info(f"fc层分类不改变")
        # 设定模型放哪里    
        model=model.to(self.device)
        
        # 小样本场景只微调最后一个特征块和分类头，保留预训练特征的稳定性，
        # 同时让模型适应护目镜这一类实验室场景，避免只训练分类头时输出塌缩到单一类别。
        for param in model.parameters():
            param.requires_grad = False
        for param in model.layer4.parameters():
            param.requires_grad = True
        for param in model.fc.parameters():
            param.requires_grad = True
            
        # 使用自定义损失函数（标签平滑）
        criterion=LabelSmoothingCrossEntropy(smoothing=0.1)
        
        #优化器
        optimizer=optim.AdamW([
            {"params": model.layer4.parameters(), "lr": 1e-5},
            {"params": model.fc.parameters(), "lr": 1e-3},
        ], weight_decay=0.01)
        
        # 调度器
        scheduler=optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epoch,eta_min=1e-6)
        
        # 跑圈训练
        best_acc=0.0
        best_model_state=None
        
        for e in range(1,epoch+1):
            default_logger.info(f"当前epoch为第{e}轮")
            if stop_check and stop_check():
                # 提前结束
                default_logger.info(f"微调训练提前结束，当前epoch为{e}")
                return False
            # 微调
            model.train()
            # 调整
            running_loss,correct,total=0.0,0,0
            
            for images,labels in train_loader:
                images=images.to(self.device)
                labels=labels.to(self.device)
                # 优化器参数清零
                optimizer.zero_grad()
                outputs=model(images)
                # 计算损失量
                loss=criterion(outputs,labels)
                # 向后传播
                loss.backward()
                # 调整梯度(确保梯度合理范围，不能超过1.0)
                clip_grad_norm_(model.parameters(), max_norm=1.0)
                # 落实操作
                optimizer.step()
                
                running_loss += loss.item()*images.size(0)
                # 计算准确率(前一次)
                _,pred=torch.max(outputs,1)
                # 计算准确率
                total += labels.size(0)
                # 计算准确率（当前）
                correct += (pred==labels).sum().item()
            # 整体微调的损失和正确率    
            train_loss=running_loss/total
            train_acc=correct/total
            
            
            # 验证
            model.eval()
            val_loss,val_correct,val_total=0.0,0,0
            # 临时解冻
            with torch.no_grad():
                for images,labels in val_loader:
                    images=images.to(self.device)
                    labels=labels.to(self.device)
                    # 导出结果
                    outputs=model(images)
                    # 计算损失量（向前传播）
                    loss=criterion(outputs,labels)                 
                    val_loss += loss.item()*images.size(0)
                    # 计算准确率(前一次)
                    _,pred=torch.max(outputs,1)
                    # 计算准确率
                    val_total += labels.size(0)
                    # 计算准确率（当前）
                    val_correct += (pred==labels).sum().item()
            val_loss/=val_total
            val_acc = val_correct/val_total
            
            # 调整学习率
            scheduler.step()
            # 获得当前的学习率
            current_lr = optimizer.param_groups[0]['lr']
            
            if val_acc > best_acc:
                # 保存模型
                best_acc = val_acc
                best_model_state=copy.deepcopy(model.state_dict())
                default_logger.info(f"验证准确率提升，当前epoch为{e}，准确率为{val_acc:.4f}")
        # 保存模型
        if best_model_state is None:
            # 微调失败
            default_logger.info("微调失败")
            return False 
        
        
        try:
            # 保存微调后的模型权重
            torch.save(best_model_state, settings.RESNET50_FINETUNED_PTH_PATH)
            default_logger.info(f"微调后的模型保存成功：{settings.RESNET50_FINETUNED_PTH_PATH}")
            
            with open(Path(settings.RESNET50_FINETUNED_CLASSNAMES_PATH),'w',encoding='utf-8') as f:
                json.dump(class_names,f,ensure_ascii=False,indent=2)
            default_logger.info(f"微调后的模型所能识别的类型列表已保存：{settings.RESNET50_FINETUNED_CLASSNAMES_PATH}")
            # 清除微调的原模型
            del model
            # 标记资源可回收
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            # 载入微调好的模型
            self._load_model()
            default_logger.info(f"微调后的模型载入成功：{settings.RESNET50_FINETUNED_PTH_PATH}")
            return True
            
            
        except Exception as e:
            default_logger.error(f"保存微调后的模型失败: {e}")
            return False       


    def predict(self,image_bytes:bytes,top_k:int = 5)->dict:
        #预测(图像识别)
        if self.model is None:
            return{
                "top1":" 模型未加载",
                "top1_confidence":0.0,
                "top_5":[{"class":" 模型未加载","confidence":0.0}]
            }
        if not self.model_ready:
            return{
                "top1":" 模型未就绪",
                "top1_confidence":0.0,
                "top_5":[{"class":" 模型未就绪","confidence":0.0}]
            }
        if not self.class_names:
            #如果可以识别的类型不存在，说明该模型没有经过微调，或者提取类型出错
            #那么，默认返回“未知类型”
            self.class_names = ["未知错误"] * 1000
        try:
            #将图片格式化24位颜色 
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            #将图片转换为张量
            input_tensor = self.transforms(image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                outputs = self.model(input_tensor)
                #计算概率
                probs = torch.softmax(outputs,dim=1).cpu().numpy()[0]
            #取top_k个概率最高索引(先升序排序取最大的5个，然后再降序排序，确保)
            top_indices = np.argsort(probs)[-top_k:][::-1]
            top_probs = probs[top_indices]
            top5 = []
            #将索引和概率打包为元组
            for idx,prob in zip(top_indices,top_probs):
                class_name = self.class_names[idx] if idx < len(self.class_names) else f"未知类型_{idx}"
                #保存所有的可能性，包括未知类型（前五个）
                top5.append({"class":class_name,"confidence":float(prob)})
            #最高的命种率
            top1_confidence = top5[0]["confidence"] if top5 else 0.0
            #获得命中率的阈值默认0.25
            threshold = getattr(settings,"CONFIDENCE_THRESHOLD",0.25)
            if top1_confidence < threshold:
                return{
                    "top1":" 未知类型",
                    "top1_confidence":0.0,
                    "top_5":[]
                }
            
            return{
                "top1":top5[0]["class"],
                "top1_confidence":top1_confidence,
                "top_5":top5
            }
        except Exception as e:
            default_logger.error(f"预测失败: {e}")
        return {
            "top1":" 预测失败",
            "top1_confidence":0.0,
            "top_5":[{"class":" 预测失败","confidence":0.0}]
        }
        
#全局单例         
classify_service =  ClassifyService()       
                
                

        
        
            
        
                
                
                
                     
        
    
        
            
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
