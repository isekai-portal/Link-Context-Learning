from .flickr import FlickrParser, FlickrDataset
from .rec import RECDataset, RECComputeMetrics
from .reg import REGDataset, GCDataset
from .caption import CaptionDataset
from .instr import InstructDataset
from .gqa import GQADataset, GQAComputeMetrics
from .clevr import ClevrDataset
from .point_qa import Point_QA_local, Point_QA_twice, V7W_POINT, PointQAComputeMetrics
from .gpt_gen import GPT4Gen
from .vcr import VCRDataset, VCRPredDataset
from .vqav2 import VQAv2Dataset
from .vqaex import VQAEXDataset
from .pure_vqa import PureVQADataset
from .pope import POPEVQADataset
from .v3det import V3DetDataset
from .lcl import LCLDataset
from .imagenet1k import ImageNet1kDatasetTrain, ImageNetTest100Eval2Way
from .isekai import ISEKAIEval2Way