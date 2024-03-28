
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


def inference_test_2():
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
    trainset = torchvision.datasets.CIFAR10(root='./data/cifar10', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size_,
                                            shuffle=True, num_workers=2)
    dataiter = iter(trainloader)
    classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    images, labels = next(dataiter)
    


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    config="vit_config_cifar10_bs64_p16_384_eps1e-3.py"

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
    model=get_model(config,pretrained=True,device=device)
    # print(model)
    # # image=torch.rand(1,3,64,64)
    # output=model.predict(images)
    images = images.to('cuda:0')
    output=model(images)
    label=torch.max(output,1)[1]
    cls=[classes[label[i]] for i in range(label.shape[0])]
    # print(output)
    print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size_)))
    print(cls)

def train_test():

    config=Config.fromfile('vit_config.py')
    print(config.pretty_text)

if __name__=="__main__":

    # print(list_models())

    # inference_test_1()
    inference_test_2()

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