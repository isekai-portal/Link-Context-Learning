BUSINESS_TEST = dict(
    type='BusinessDataset',
    filename=r'/mnt/lustre/fanweichen2/Research/MLLM/dataset/easy/qr_code_dt/all.txt',
    filename_positive=r'/mnt/lustre/fanweichen2/Research/MLLM/dataset/easy/qr_code_dt/positive.txt',
    filename_negative=r'/mnt/lustre/fanweichen2/Research/MLLM/dataset/easy/qr_code_dt/negative.txt',
    image_folder_positive=r'sdc:s3://MultiModal/Benchmark/simple_semantics/WA/dataset/qrcode_det_test/',
    image_folder_negative=r'sdc:s3://MultiModal/Benchmark/simple_semantics/WA/dataset/qrcode_det_test/',
    image_folder=r'sdc:s3://MultiModal/Benchmark/simple_semantics/WA/dataset/qrcode_det_test/',
    label='QR_code',
    template_file=r"{{fileDirname}}/template/ICL.json",
)
