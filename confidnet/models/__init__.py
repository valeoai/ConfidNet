from confidnet.models import MLP
from confidnet.models import MLPExtractor
from confidnet.models import MLPSelfConfid
from confidnet.models import MLPSelfConfidCloning
from confidnet.models import MLPOODConfid
from confidnet.models import SmallConvNetMNIST
from confidnet.models.small_convnet_mnist_extractor import SmallConvNetMNISTExtractor
from confidnet.models.small_convnet_mnist_selfconfid_classic import SmallConvNetMNISTSelfConfidClassic
from confidnet.models import SmallConvNetMNISTSelfConfidCloning
from confidnet.models import SmallConvNetMNISTOODConfid
from confidnet.models import SmallConvNetSVHN
from confidnet.models import SmallConvNetSVHNExtractor
from confidnet.models import SmallConvNetSVHNSelfConfidClassic
from confidnet.models import SmallConvNetSVHNSelfConfidCloning
from confidnet.models.small_convnet_svhn_selfconfid_1layers import SmallConvNetSVHNSelfConfid1Layers
from confidnet.models.small_convnet_svhn_selfconfid_2layers import SmallConvNetSVHNSelfConfid2Layers
from confidnet.models.small_convnet_svhn_selfconfid_3layers import SmallConvNetSVHNSelfConfid3Layers
from confidnet.models import SmallConvNetSVHNSelfConfid4Layers
from confidnet.models import SmallConvNetSVHNSelfConfid6Layers
from confidnet.models import SmallConvNetSVHNSelfConfid7Layers
from confidnet.models.small_convnet_svhn_oodconfid import SmallConvNetSVHNOODConfid
from confidnet.models import VGG16
from confidnet.models import VGG16Extractor
from confidnet.models.vgg16_selfconfid_classic import VGG16SelfConfidClassic
from confidnet.models import VGG16SelfConfidCloning
from confidnet.models import VGG16OODConfid
from confidnet.models.segnet import Segnet
from confidnet.models import SegnetExtractor
from confidnet.models import SegnetSelfConfid
from confidnet.models.segnet_selfconfid_cloning import SegnetSelfConfidCloning
from confidnet.models import SegNetOODConfid


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
