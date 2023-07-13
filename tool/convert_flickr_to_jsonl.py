if __name__ == '__main__':
    from mllm.dataset.single_image_dataset.flickr import FlickrParser

    filenames = [
        r'D:\home\dataset\flickr30kentities\train.txt',
        r'D:\home\dataset\flickr30kentities\val.txt',
        r'D:\home\dataset\flickr30kentities\test.txt',
    ]
    dump_filenames = [
        'train.jsonl',
        'val.jsonl',
        'test.jsonl',
    ]
    annotation_dir = r'D:\home\dataset\flickr30kentities'

    for filename, dumpname in zip(filenames, dump_filenames):
        fp = FlickrParser(filename=filename, annotation_dir=annotation_dir)
        fp.dump(dumpname)
