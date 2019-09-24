from models.mlp import MLP
from models.mlp_extractor import MLPExtractor
from models.mlp_selfconfid import MLPSelfConfid
from models.mlp_selfconfid_cloning import MLPSelfConfidCloning
from models.mlp_oodconfid import MLPOODConfid
from models.small_convnet_mnist import SmallConvNetMNIST
from models.small_convnet_mnist_extractor import SmallConvNetMNISTExtractor
from models.small_convnet_mnist_selfconfid_classic import SmallConvNetMNISTSelfConfidClassic
from models.small_convnet_mnist_selfconfid_cloning import SmallConvNetMNISTSelfConfidCloning
from models.small_convnet_mnist_oodconfid import SmallConvNetMNISTOODConfid
from models.small_convnet_svhn import SmallConvNetSVHN
from models.small_convnet_svhn_extractor import SmallConvNetSVHNExtractor
from models.small_convnet_svhn_selfconfid_classic import SmallConvNetSVHNSelfConfidClassic
from models.small_convnet_svhn_selfconfid_cloning import SmallConvNetSVHNSelfConfidCloning
from models.small_convnet_svhn_selfconfid_1layers import SmallConvNetSVHNSelfConfid1Layers
from models.small_convnet_svhn_selfconfid_2layers import SmallConvNetSVHNSelfConfid2Layers
from models.small_convnet_svhn_selfconfid_3layers import SmallConvNetSVHNSelfConfid3Layers
from models.small_convnet_svhn_selfconfid_4layers import SmallConvNetSVHNSelfConfid4Layers
from models.small_convnet_svhn_selfconfid_6layers import SmallConvNetSVHNSelfConfid6Layers
from models.small_convnet_svhn_selfconfid_7layers import SmallConvNetSVHNSelfConfid7Layers
from models.small_convnet_svhn_oodconfid import SmallConvNetSVHNOODConfid
from models.vgg16 import VGG16
from models.vgg16_extractor import VGG16Extractor
from models.vgg16_selfconfid_classic import VGG16SelfConfidClassic
from models.vgg16_selfconfid_cloning import VGG16SelfConfidCloning
from models.vgg16_oodconfid import VGG16OODConfid
from models.segnet import Segnet
from models.segnet_extractor import SegnetExtractor
from models.segnet_selfconfid import SegnetSelfConfid
from models.segnet_selfconfid_cloning import SegnetSelfConfidCloning
from models.segnet_oodconfid import SegNetOODConfid


def get_model(config_args, device):
    """
        Return a new instance of model
    """
    
    # Available models
    model_factory = {
        "mlp": MLP,
        "mlp_extractor": MLPExtractor,
        "mlp_selfconfid": MLPSelfConfid,
        "mlp_selfconfid_cloning": MLPSelfConfidCloning,
        "mlp_oodconfid": MLPOODConfid,
        'small_convnet_mnist': SmallConvNetMNIST,
        'small_convnet_mnist_extractor': SmallConvNetMNISTExtractor,
        'small_convnet_mnist_selfconfid_classic': SmallConvNetMNISTSelfConfidClassic,
        'small_convnet_mnist_selfconfid_cloning': SmallConvNetMNISTSelfConfidCloning,
        "small_convnet_mnist_oodconfid": SmallConvNetMNISTOODConfid,
        'small_convnet_svhn_conv': SmallConvNetSVHN,
        'small_convnet_svhn_extractor': SmallConvNetSVHNExtractor,
        'small_convnet_svhn_selfconfid_classic': SmallConvNetSVHNSelfConfidClassic,
        'small_convnet_svhn_selfconfid_cloning': SmallConvNetSVHNSelfConfidCloning,
        'small_convnet_svhn_selfconfid_1layers': SmallConvNetSVHNSelfConfid1Layers,
        'small_convnet_svhn_selfconfid_2layers': SmallConvNetSVHNSelfConfid2Layers,
        'small_convnet_svhn_selfconfid_3layers': SmallConvNetSVHNSelfConfid3Layers,
        'small_convnet_svhn_selfconfid_4layers': SmallConvNetSVHNSelfConfid4Layers,
        'small_convnet_svhn_selfconfid_6layers': SmallConvNetSVHNSelfConfid6Layers,
        'small_convnet_svhn_selfconfid_7layers': SmallConvNetSVHNSelfConfid7Layers,
        "small_convnet_svhn_oodconfid": SmallConvNetSVHNOODConfid,
        'vgg16': VGG16,
        'vgg16_extractor': VGG16Extractor,
        'vgg16_selfconfid_classic': VGG16SelfConfidClassic,
        'vgg16_selfconfid_cloning': VGG16SelfConfidCloning,
        "vgg16_oodconfid": VGG16OODConfid,
        "segnet": Segnet,
        "segnet_extractor": SegnetExtractor,
        "segnet_selfconfid": SegnetSelfConfid,
        "segnet_selfconfid_cloning": SegnetSelfConfidCloning,
        "segnet_oodconfid": SegNetOODConfid
    }
    
    if config_args['model']['name'].lower() not in model_factory:
        raise Exception("Model {} non existing".format(config_args['model_name']))
        
    return model_factory[config_args['model']['name']](config_args=config_args, device=device)
