ISEKAI_PAIR = dict(
    type='ISEKAIEval2Way',
    filename=r'/mnt/lustre/share_data/taiyan/dataset/isekai/ISEKAI-pair.json',
    image_folder=r'/mnt/lustre/share_data/zhangzhao2/VG/ISEKAI/ISEKAI-20/',
    template_file=r"{{fileDirname}}/template/ICL.json",
    policy="policy_2way"
)
ISEKAI_10 = dict(
    type='ISEKAIEval2Way',
    filename=r'/mnt/lustre/share_data/taiyan/dataset/isekai/ISEKAI-10.json',
    image_folder=r'/mnt/lustre/share_data/zhangzhao2/VG/ISEKAI/ISEKAI-20/',
    template_file=r"{{fileDirname}}/template/ICL.json",
    policy="policy_2way"
)