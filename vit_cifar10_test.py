
import mmpretrain
from mmpretrain import inference_model
from mmcv.ops import get_compiling_cuda_version, get_compiler_version
import torch
from mmpretrain import get_model
from mmengine.config import Config
from mmpretrain import list_models
from mmpretrain import ImageClassificationInferencer
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
import matplotlib.pyplot as plt
import os
import config
from matplotlib import style
import scienceplots
import seaborn as sns
import numpy as np
# from mmpretrain.datasets import CUB


from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

class CUB200Dataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        
        # Load and prepare data
        with open(os.path.join(root_dir, 'images.txt'), 'r') as f:
            for line in f:
                image_id, image_file = line.strip().split(' ')
                self.image_paths.append(os.path.join(root_dir, 'images', image_file))
        
        with open(os.path.join(root_dir, 'image_class_labels.txt'), 'r') as f:
            for line in f:
                image_id, class_label = line.strip().split(' ')
                self.labels.append(int(class_label) - 1)  # Assuming class labels start from 1

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_name = self.image_paths[idx]
        image = Image.open(img_name).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


class OxfordPetsDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.labels = []

        # Assume the labels are in a file named 'labels.txt'
        labels_path = os.path.join(root_dir, 'annotations', 'list.txt')
        with open(labels_path, 'r') as file:
            for line in file:
                if line.startswith("#"):
                    continue
                parts = line.strip().split()
                image_path = os.path.join(root_dir, 'images', parts[0] + '.jpg')
                label = int(parts[2]) - 1  # Convert class index to 0-based
                self.images.append(image_path)
                self.labels.append(label)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)

        return image, label








def inference_test():
    print(torch.__version__, torch.cuda.is_available())
    print(mmpretrain.__version__)

    print(get_compiling_cuda_version())
    print(get_compiler_version())
    # 根据colab状态设置device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # 选择下载好的checkpoint
    model = get_model("vit-base-p32_in21k-pre_3rdparty_in1k-384px", pretrained="vit-base-p32_in21k-pre-3rdparty_ft-64xb64_in1k-384_20210928-9cea8599.pth")

    image='demo/bird.JPEG'

    result = inference_model(model, image,device=device, show=True)

    # print(result['pred_class'])
    print(result)
   
def inference_test_1():
    print(torch.__version__, torch.cuda.is_available())
    print(mmpretrain.__version__)

    print(get_compiling_cuda_version())
    print(get_compiler_version())
    # 根据colab状态设置device
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    # 选择下载好的checkpoint

    config="vit_config_cifar10.py"
    # checkpoint="vit-base-p32_in21k-pre-3rdparty_ft-64xb64_in1k-384_20210928-9cea8599.pth"

    image='demo/bird.JPEG'
    # inferencer = ImageClassificationInferencer(model=config, pretrained=checkpoint, device=device)
    inferencer = ImageClassificationInferencer(model=config, device=device)
    result = inferencer(image)[0]
    print(result['pred_class'])


def inference_test_2(order):
    print(torch.__version__, torch.cuda.is_available())
    print(mmpretrain.__version__)
    print(get_compiling_cuda_version())
    print(get_compiler_version())
    batch_size_=8

    transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Resize((384,384),interpolation=InterpolationMode.BICUBIC),
    #  transforms.Normalize(mean=(125.307, 122.961, 113.8575), std=(51.5865, 50.847, 51.255))
     ])

    # 加载训练集
    trainset = torchvision.datasets.CIFAR10(root='./data/cifar10', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size_,
                                            shuffle=True, num_workers=2)
    dataiter = iter(trainloader)
    classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    images, labels = next(dataiter)
    


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # order=30
    # config="vit_config_cifar10_bs64_p16_384_eps1e-3.py"
    config=f"gpu_vit_config_cifar10_bs64_p16_384_mode1_order_{order}.py"
    config2=f"gpu_vit_config_cifar10_bs64_p16_384_mode2_order_{order}.py"


    # image=images.numpy()[0,:,:,:]
    # inferencer = ImageClassificationInferencer(model=config, device=device)
    # print(inferencer.model)
    # result = inferencer(image)[0]
    # print(result['pred_class'])

    data_preprocessor = dict(
    num_classes=10,
    # RGB format normalization parameters
    mean=[125.307, 122.961, 113.8575],
    std=[51.5865, 50.847, 51.255],
    # loaded images are already RGB format
    to_rgb=False)
    images = images.to('cuda:0')

    model=get_model(config,pretrained=True,device=device)
    output1=model(images)
    label=torch.max(output1,1)[1]
    cls=[classes[label[i]] for i in range(label.shape[0])]
    # print(output)
    print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size_)))
    print(cls)
    # exit()


    model2=get_model(config2,pretrained=True,device=device)
    output2=model2(images)
    label2=torch.max(output2,1)[1]
    cls2=[classes[label2[i]] for i in range(label2.shape[0])]
    # print(output)
    print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size_)))
    print(cls2)

    # print(model)
    # # image=torch.rand(1,3,64,64)
    # output=model.predict(images)
    
# def cosine_sim_compare(dataset,order):
#     print(torch.__version__, torch.cuda.is_available())
#     print(mmpretrain.__version__)
#     print(get_compiling_cuda_version())
#     print(get_compiler_version())
#     batch_size_=8

#     transform = transforms.Compose(
#     [transforms.ToTensor(),
#      transforms.Resize((384,384),interpolation=InterpolationMode.BICUBIC),
#     #  transforms.Normalize(mean=(125.307, 122.961, 113.8575), std=(51.5865, 50.847, 51.255))
#      ])

#     # 加载训练集
#     if dataset=='cifar10':
#         trainset = torchvision.datasets.CIFAR10(root='./data/cifar10', train=True,
#                                                 download=True, transform=transform)
#         trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size_,
#                                                 shuffle=True, num_workers=2)
#     elif dataset=='cifar100':
#         trainset = torchvision.datasets.CIFAR100(root='./data/cifar100', train=True,
#                                                 download=True, transform=transform)
#         trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size_,
#                                                 shuffle=True, num_workers=2)
#     elif dataset=='cub':
#         trainset = CUB(data_root='./data/CUB_200_2011', split='train',transform=transform)
#         trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size_,
#                                                 shuffle=True, num_workers=2)
#     elif dataset=='oxford_iii_pets':
#         trainset = torchvision.datasets.OxfordIIITPet(root='./data/Oxford_IIII_Pets', train=True,
#                                                 download=True, transform=transform)
#         trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size_,
#                                                 shuffle=True, num_workers=2)
#     dataiter = iter(trainloader)
#     # classes = ('plane', 'car', 'bird', 'cat',
#     #        'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
#     images, _ = next(dataiter)
#     images = images.to('cuda:0')

#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     # config=f"gpu_vit_config_{dataset}_bs64_p16_384_mode1_order_{order}.py"
#     # config2=f"gpu_vit_config_{dataset}_bs64_p16_384_mode2_order_{order}.py"

#     config=f"gpu_vit_config_{dataset}_bs64_p16_384_mode1_order_{order}.py"
#     config2=f"gpu_vit_config_{dataset}_bs64_p16_384_mode2_order_{order}.py"
    

#     model=get_model(config,pretrained=True,device=device)
#     output1=model(images)

#     model2=get_model(config2,pretrained=True,device=device)
#     output2=model2(images)
 
    
def train_test():

    config=Config.fromfile('vit_config.py')
    print(config.pretty_text)


   
def cosine_sim_compare_mr(dataset,eps,repeat_id):
    print(torch.__version__, torch.cuda.is_available())
    print(mmpretrain.__version__)
    print(get_compiling_cuda_version())
    print(get_compiler_version())
    batch_size_=16

    transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Resize((384,384),interpolation=InterpolationMode.BICUBIC),
    #  transforms.Normalize(mean=(125.307, 122.961, 113.8575), std=(51.5865, 50.847, 51.255))
     ])

    # 加载训练集
    if dataset=='cifar10':
        trainset = torchvision.datasets.CIFAR10(root='./data/cifar10', train=True,
                                                download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size_,
                                                shuffle=True, num_workers=2)
    elif dataset=='cifar100':
        trainset = torchvision.datasets.CIFAR100(root='./data/cifar100', train=True,
                                                download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size_,
                                                shuffle=True, num_workers=2)
    elif dataset=='cub':
        trainset = CUB200Dataset(root_dir='./data/CUB_200_2011', transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size_,
                                                shuffle=True, num_workers=2)
    elif dataset=='oxford_iii_pets':
        trainset = OxfordPetsDataset(root_dir='./data/Oxford_IIII_Pets', transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size_,
                                                shuffle=True, num_workers=2)
    dataiter = iter(trainloader)
    # classes = ('plane', 'car', 'bird', 'cat',
    #        'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    images, _ = next(dataiter)
    # images, a,b = next(dataiter)
    images = images.to('cuda:0')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # config=f"gpu_vit_config_{dataset}_bs64_p16_384_mode1_order_{order}.py"
    # config2=f"gpu_vit_config_{dataset}_bs64_p16_384_mode2_order_{order}.py"

    file_prepre=f'test_qofd_0513/{dataset}/eps{eps}/{repeat_id}/'

    config_eps0=f"gpu_vit_config_{dataset}_bs64_p16_384_mode0_eps0.py"
    config=f"gpu_vit_config_{dataset}_bs64_p16_384_mode0_eps{eps}.py"
    config_defer=f"gpu_vit_config_{dataset}_bs64_p16_384_mode0_eps{eps}_defer.py"
    
    file_pre=file_prepre+'standard/'
    if not os.path.exists(file_pre):
    # 如果目录不存在，则创建它
        os.makedirs(file_pre)
    model_eps0=get_model(config_eps0,pretrained=True,device=device)
    output0=model_eps0(images)

    file_pre=file_prepre+'multiple_reuse/'
    if not os.path.exists(file_pre):
    # 如果目录不存在，则创建它
        os.makedirs(file_pre)
    model=get_model(config,pretrained=True,device=device)
    output1=model(images)

    file_pre=file_prepre+'single_reuse/'
    if not os.path.exists(file_pre):
    # 如果目录不存在，则创建它
        os.makedirs(file_pre)
    model_defer=get_model(config_defer,pretrained=True,device=device)
    output2=model_defer(images)


def cosine_sim(x,y):
    return torch.sum(x*y)/torch.norm(x)/torch.norm(y)

def draw_figures(dataset,eps,layer):
    cos_mr=[]
    cos_defer=[]
    file_pre='test_qofd_0513'
    repeat_number=2
    for i in range(layer):
        print(i)
        f_eps0=f'{file_pre}/{dataset}/eps{eps}/{repeat_number}/standard/layer={i}.pt'
        f_mr=f'{file_pre}/{dataset}/eps{eps}/{repeat_number}/multiple_reuse/layer={i}.pt'
        f_defer=f'{file_pre}/{dataset}/eps{eps}/{repeat_number}/single_reuse/layer={i}.pt'
        x_eps0=torch.load(f_eps0)
        x_mr=torch.load(f_mr)
        x_defer=torch.load(f_defer)
        temp1=cosine_sim(x_eps0,x_mr).detach().cpu()
        temp2=cosine_sim(x_eps0,x_defer).detach().cpu()
        cos_mr.append(temp1)
        cos_defer.append(temp2)
    
    plt.plot(range(layer),cos_mr,label='multiple')
    plt.scatter(range(layer),cos_mr)
    plt.plot(range(layer),cos_defer,label='Single')
    plt.scatter(range(layer),cos_defer)
    plt.xlabel('layer')
    plt.ylabel('cosine similarity')
    plt.legend(loc=1)
    plt.show()

def compute_cosine_similarity(dataset,eps_range,layer,repeat_range):
    cos_mr=[]
    cos_defer=[]
    file_pre='test_qofd_0513'
    for eps in eps_range:
        cos_mr1=[]
        cos_defer1=[]
        for repeat_number in repeat_range:
            temp_mr=[]
            temp_single=[]
            for i in range(layer):
                print(i)
                f_eps0=f'{file_pre}/{dataset}/eps{eps}/{repeat_number}/standard/layer={i}.pt'
                f_mr=f'{file_pre}/{dataset}/eps{eps}/{repeat_number}/multiple_reuse/layer={i}.pt'
                f_defer=f'{file_pre}/{dataset}/eps{eps}/{repeat_number}/single_reuse/layer={i}.pt'
                x_eps0=torch.load(f_eps0)
                x_mr=torch.load(f_mr)
                x_defer=torch.load(f_defer)
                temp1=cosine_sim(x_eps0,x_mr).detach().cpu()
                temp2=cosine_sim(x_eps0,x_defer).detach().cpu()
                temp_mr.append(temp1)
                temp_single.append(temp2)
            cos_mr1.append(temp_mr)
            cos_defer1.append(temp_single)
        cos_mr.append(cos_mr1)
        cos_defer.append(cos_defer1)
    return np.array(cos_mr),np.array(cos_defer)

def draw_figure_new(cos_mr,cos_defer,eps_range,title):
    style.use('seaborn-v0_8-darkgrid')
    # plt.style.use(['science', 'grid'])
    plt.style.use(['science', 'grid'])
    # 创建图表
    plt.figure(figsize=(8, 6))

    step_range=range(12)
    x_axis=cos_mr.shape[-1]
    # color_lib=['blue','green','red']
    color_lib=['r','g','b','c','m','k','y','purple']
    for r in range(cos_mr.shape[0]):
        cos_ave_origin = [np.average(cos_mr[r,:, i]) for i in range(x_axis)]
        cos_std_origin = [np.std(cos_mr[r,:, i]) for i in range(x_axis)]

        plt.errorbar(step_range, cos_ave_origin, yerr=cos_std_origin, fmt='o', color=color_lib[r], 
                    ecolor=color_lib[r], elinewidth=2, capsize=3, capthick=2)
        plt.scatter(step_range, cos_ave_origin, color=color_lib[r], s=40)  # s 控制点的大小
        plt.plot(step_range, cos_ave_origin,label=f'Multiple, $\delta$={eps_range[r]}',lw=3,color=color_lib[r])

        cos_ave_defer = [np.average(cos_defer[r,:, i]) for i in range(x_axis)]
        cos_std_defer = [np.std(cos_defer[r,:, i]) for i in range(x_axis)]

        plt.errorbar(step_range, cos_ave_defer, yerr=cos_std_defer, fmt='o', color=color_lib[r], 
                    ecolor=color_lib[r], elinewidth=2, capsize=3, capthick=2)
        plt.scatter(step_range, cos_ave_defer, color=color_lib[r], s=40)  # s 控制点的大小
        plt.plot(step_range, cos_ave_defer,label=f'Single, $\delta$={eps_range[r]}',lw=3,color=color_lib[r],linestyle='--')
    
    font_size=22
    plt.xlabel('Layer Number', fontsize=font_size)
    # plt.xscale('log')
    plt.legend(fontsize=15,loc=3)
    plt.ylabel('Cosine Similarity', fontsize=font_size)
    plt.tick_params(axis='both', labelsize=font_size)  # 主刻度标签大小
    plt.xlim(-2,12)
    plt.title(title,fontsize=font_size)
    plt.savefig(f'{dataset}_forward.png')
    plt.savefig(f'{dataset}_forward.pdf')
    plt.show()

if __name__=="__main__":

    # print(list_models())
    
    # inference_test_1()
    # order_range=[5,10,20,30,40,50]
    # dataset='cub'
    # dataset='oxford_iii_pets'
    # is_save=True
    dataset='oxford_iii_pets'
    title='Oxford-IIIT Pets'
    eps_range=['2e-3','3e-3','4e-3']
    eps_range1=['0.002','0.003','0.004']
    layer=12
    repeat_range=range(1,11)
    # draw_figures(dataset,'5e-3',layer=12)
    # exit()
    cos_mr,cos_single=compute_cosine_similarity(dataset,eps_range,layer,repeat_range)
    print(cos_mr,cos_single)
    draw_figure_new(cos_mr,cos_single,eps_range1,title)
    exit()

    # file_pre='test_qofd_0513'
    # filename_block=f'{file_pre}/{dataset}/{dataset}_eps={0.1}.pt'
    # filename_block2=f'{file_pre}/{dataset}/{dataset}_eps={0}.pt'
    # a1=torch.load(filename_block)
    # a2=torch.load(filename_block2)
    # x=cosine_sim(a1,a2)
    # print(x)
    # exit()
    # order_range=[2,5,10,20,30,40,50]
    # eps_range=['1e-1','1e-2','1e-3']
    eps_range=['2e-3']
    repeat_id=1
    for order in eps_range:
        cosine_sim_compare_mr(dataset,order,repeat_id)


    exit()


    inference_test()

    # predict = inference_model('vit-base-p32_in21k-pre_3rdparty_in1k-384px', 'demo/bird.JPEG')
    # print(predict['pred_class'])
    # print(predict['pred_score'])
    # train_test()
    # exit()


    # model = get_model('vit-base-p32_in21k-pre_3rdparty_in1k-384px', pretrained=True)
    # # inputs = torch.rand(1, 3, 224, 224)
    # inputs = torch.rand(1, 3, 224, 224)
    # out = model(inputs)
    # print(type(out))
    # # To extract features.
    # feats = model.extract_feat(inputs)
    # print(type(feats))