from confidnet.loaders import usualtorch_loader as dload


def get_loader_class(name):
    """
        Return a new instance of dataset loader
    """
    
    # Available models
    data_loader_factory = {
        'cifar10': dload.CIFAR10Loader,
        'cifar100': dload.CIFAR100Loader,
        'mnist': dload.MNISTLoader,
        'svhn': dload.SVHNLoader,
        'camvid': dload.CamVidLoader
    }

    return data_loader_factory[name]
