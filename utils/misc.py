import os
import csv
import torch


def one_hot_embedding(labels, num_classes):
    """Embedding labels to one-hot form.

    Args:
      labels: (LongTensor) class labels, sized [N,].
      num_classes: (int) number of classes.

    Returns:
      (tensor) encoded labels, sized [N, #classes].
    """
    y = torch.eye(num_classes) 
    return y[labels] 


def csv_writter(path, dic):       
    # Check if the file already exists
    if os.path.isfile(path):
        append_mode = True
        rw_mode = 'a'
    else:
        append_mode = False
        rw_mode = 'w'
        
    # Write dic
    with open(path, rw_mode) as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        # Do not write header in append mode
        if append_mode is False:
            writer.writerow(dic.keys())
        writer.writerow([elem['string'] for elem in dic.values()])


def print_dict(logs_dict):
    str_print = ''
    for metr_name in logs_dict:
        str_print+='{}={},  '.format(metr_name, logs_dict[metr_name]['string'])
    print (str_print)
