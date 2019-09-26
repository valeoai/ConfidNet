from confidnet.models.mlp import MLP
from confidnet.models.mlp_extractor import MLPExtractor
from confidnet.models.mlp_selfconfid import MLPSelfConfid
from confidnet.models.mlp_selfconfid_cloning import MLPSelfConfidCloning
from confidnet.models.mlp_oodconfid import MLPOODConfid
from confidnet.models.small_convnet_mnist import SmallConvNetMNIST
from confidnet.models.small_convnet_mnist_extractor import SmallConvNetMNISTExtractor
from confidnet.models.small_convnet_mnist_selfconfid_classic import SmallConvNetMNISTSelfConfidClassic
from confidnet.models.small_convnet_mnist_selfconfid_cloning import SmallConvNetMNISTSelfConfidCloning
from confidnet.models.small_convnet_mnist_oodconfid import SmallConvNetMNISTOODConfid
from confidnet.models.small_convnet_svhn import SmallConvNetSVHN
from confidnet.models.small_convnet_svhn_extractor import SmallConvNetSVHNExtractor
from confidnet.models.small_convnet_svhn_selfconfid_classic import SmallConvNetSVHNSelfConfidClassic
from confidnet.models.small_convnet_svhn_selfconfid_cloning import SmallConvNetSVHNSelfConfidCloning
from confidnet.models.small_convnet_svhn_oodconfid import SmallConvNetSVHNOODConfid
from confidnet.models.vgg16 import VGG16
from confidnet.models.vgg16_extractor import VGG16Extractor
from confidnet.models.vgg16_selfconfid_classic import VGG16SelfConfidClassic
from confidnet.models.vgg16_selfconfid_cloning import VGG16SelfConfidCloning
from confidnet.models.vgg16_oodconfid import VGG16OODConfid
from confidnet.models.segnet import Segnet
from confidnet.models.segnet_extractor import SegnetExtractor
from confidnet.models.segnet_selfconfid import SegnetSelfConfid
from confidnet.models.segnet_selfconfid_cloning import SegnetSelfConfidCloning
from confidnet.models.segnet_oodconfid import SegNetOODConfid


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
        'small_convnet_svhn': SmallConvNetSVHN,
        'small_convnet_svhn_extractor': SmallConvNetSVHNExtractor,
        'small_convnet_svhn_selfconfid_classic': SmallConvNetSVHNSelfConfidClassic,
        'small_convnet_svhn_selfconfid_cloning': SmallConvNetSVHNSelfConfidCloning,
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

    return model_factory[config_args['model']['name']](config_args=config_args, device=device)
