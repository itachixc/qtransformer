
import mmpretrain
from mmpretrain import inference_model
from mmcv.ops import get_compiling_cuda_version, get_compiler_version
import torch
from mmpretrain import get_model
from mmengine.config import Config
from mmpretrain import list_models
from mmpretrain import ImageClassificationInferencer
from mmpretrain.models.utils import LinearSampling
import torch.nn as nn
import numpy as np 

def inference_test():
    print(torch.__version__, torch.cuda.is_available())
    print(mmpretrain.__version__)

    print(get_compiling_cuda_version())
    print(get_compiler_version())
    # 根据colab状态设置device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    b=list_models('vit')
    print(b)
    # exit()
    # 选择下载好的checkpoint
    # model = get_model("vit-base-p32_in21k-pre_3rdparty_in1k-384px", pretrained=True)
    model = get_model("vit-base-p16_in21k-pre_3rdparty_in1k-384px", pretrained=True)
    
    # model = get_model("vit-base-p32_in21k-pre_3rdparty_in1k-384px", pretrained="vit-base-p32_in21k-pre-3rdparty_ft-64xb64_in1k-384_20210928-9cea8599.pth")
    # model = get_model("vit_config_sampling.py", pretrained=False)
    torch.save(model,'vit-base-p16_in21k-pre_3rdparty_in1k-384px.pth')
    print(model)
    exit()

    image='demo/bird.JPEG'

    result = inference_model(model, image,device=device, show=True)

    # print(result['pred_class'])
    # print(result)
   
def inference_test_1():
    print(torch.__version__, torch.cuda.is_available())
    print(mmpretrain.__version__)

    print(get_compiling_cuda_version())
    print(get_compiler_version())
    # 根据colab状态设置device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # 选择下载好的checkpoint

    # config="vit_config_small.py"
    config="vit_config_sampling.py"
    # config="vit_config_sampling_1.py"
    # checkpoint='vit-base-p16_3rdparty_pt-64xb64_in1k-224_20210928-02284250.pth'
    checkpoint='imagenet21k_ViT-B_16.npz'
    # checkpoint="vit-base-p32_in21k-pre-3rdparty_ft-64xb64_in1k-384_20210928-9cea8599.pth"

    # image='demo/bird.JPEG'
    image='output.png'
    # inferencer = ImageClassificationInferencer(model=config, pretrained=checkpoint, device=device)
    inferencer = ImageClassificationInferencer(model=config,pretrained=checkpoint, device=device)
    result = inferencer(image)[0]
    print(result['pred_class'])


def train_test():

    config=Config.fromfile('vit_config.py')
    print(config.pretty_text)


def linear_sampling_test():
    m1=LinearSampling(in_features=2,out_features=3,sampling_error=0.1)
    # m2=LinearSampling(in_features=2,out_features=3,sampling_error=0.)
    x=torch.rand(1,4,2,requires_grad=True)
    y=m1(x)
    loss=y.sum()
    loss.backward()
    print(x.grad)
    print(m1.weight.grad)
    print(m1.bias.grad)

    # y2=m2(x)
    # loss2=y2.sum()
    # loss2.backward()
    # print(m2.weight.grad)
    # print(m2.bias.grad)
def npz_to_pth():
    numpy_weights = np.load('imagenet21k_ViT-B_16.npz')
    # 创建一个空的 PyTorch 模型
    # model = YourModel()  # 用你的模型类初始化

    # # 将 NumPy 字典中的权重复制到 PyTorch 模型的 state_dict 中
    # model_state_dict = model.state_dict()

    # for key in model_state_dict.keys():
    #     # 从 NumPy 数组中复制权重
    #     model_state_dict[key] = torch.from_numpy(numpy_weights[key])

    # # 将 state_dict 设置到 PyTorch 模型中
    # model.load_state_dict(model_state_dict)
    # # tensor_weights = torch.from_numpy(numpy_weights['weight_array']).float()
    # torch.save(tensor_weights, 'imagenet21k_ViT-B_16.pth')

if __name__=="__main__":
    # npz_to_pth()
    # exit()

    # print(list_models())
    # inference_test()
    # exit()
    # c=torch.tensor([1.0,2,3])
    # d=torch.softmax(c,dim=-1)
    # # g=torch.sum(c,dim=-1)
    # f=c.repeat(3,2,1)
    # print(f)

    a=torch.rand(2,3,4)
    b=torch.unsqueeze(a,dim=1)
    print(a.size(),b.size())


    # linear_sampling_test()
    exit()
    m=nn.Linear(3,4)
    # inference_test_1()
    x=torch.rand(2,4,3)
    y=m(x)
    print(y,y.shape)

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