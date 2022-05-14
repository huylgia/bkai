dataset_type = 'IcdarDataset'
data_root = '/content/detection_mmocr'

train1 = dict(
    type=dataset_type,
    ann_file=f'{data_root}/instances_train_images.json',
    img_prefix=f'{data_root}/imgs',
    pipeline=None)

train2 = dict(
    type=dataset_type,
    ann_file=f'{data_root}/instances_test_image.json',
    img_prefix=f'{data_root}/imgs',
    pipeline=None)

train3 = dict(
    type=dataset_type,
    ann_file=f'{data_root}/instances_unseen_test_images.json',
    img_prefix=f'{data_root}/imgs',
    pipeline=None)
    
test = dict(
    type=dataset_type,
    ann_file=f'{data_root}/instances_BK.json',
    img_prefix=f'{data_root}/imgs',
    pipeline=None)

train_list = [train1, train2, train3]

test_list = [test]
